#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from common_eval.journal_adapter import JournalAdapter
from common_eval.model_compat import GraphBatch
from common_eval.proceeding_adapter import ProceedingAdapter
from common_eval.qm9_data import (
    PROPERTY_CONVERSIONS,
    PROPERTY_NATIVE_UNITS,
    PROPERTY_SOURCE_UNITS,
    batch_molecules,
    construct_selected_graphs,
    load_excluded_cids,
    make_split,
    read_property_values,
    read_sdf_cids,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
PROCEEDING_MODEL_DIR = WORKSPACE / "proceeding" / "model"
JOURNAL_MODEL_DIR = SCRIPT_DIR / "model"
OUTPUT_MARKER = ".common_split_evaluation_output"
OUTPUT_MARKER_CONTENT = "mol-infer-gnn common-split evaluation output\n"
BATCH_CONSISTENCY_RTOL = 1e-5
BATCH_CONSISTENCY_ATOL = 1e-4
BATCH_CONSISTENCY_HARD_THRESHOLD = 1e-2
SMOKE_MODEL_IDS = {
    "proceeding_homo_3lay_h16",
    "journal_homo_3lay_h16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic common-split reevaluation of proceeding and journal QM9 checkpoints."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "audit" / "common_split_full_18models",
        help="New directory for audit outputs; must not already exist.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alternate-batch-size", type=int, default=127)
    parser.add_argument(
        "--batch-consistency-atol",
        type=float,
        default=BATCH_CONSISTENCY_ATOL,
        help="Absolute tolerance in eV for predictions from different batch sizes (default: 1e-4).",
    )
    parser.add_argument(
        "--batch-consistency-rtol",
        type=float,
        default=BATCH_CONSISTENCY_RTOL,
        help="Relative tolerance for predictions from different batch sizes (default: 1e-5).",
    )
    parser.add_argument(
        "--batch-consistency-mode",
        choices=("strict", "warn", "off"),
        default=None,
        help="Batch consistency policy. Default: strict with --smoke-only, warn for a full run.",
    )
    parser.add_argument(
        "--batch-consistency-hard-threshold",
        type=float,
        default=BATCH_CONSISTENCY_HARD_THRESHOLD,
        help="Always stop when max absolute difference exceeds this value in eV (default: 1e-2).",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help=(
            "Replace an existing output directory only when it contains this script's "
            "dedicated safety marker. Unmarked directories are never deleted."
        ),
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Evaluate only proceeding HOMO GNN16 and journal HOMO 3-layer/h16. Without this flag, all 18 models are evaluated.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the selected model inventory and exit without creating output or evaluating checkpoints.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    protected_paths = {Path.home().resolve(), SCRIPT_DIR.resolve(), WORKSPACE.resolve()}
    if path in protected_paths or len(path.parts) < 4:
        raise ValueError(f"Refusing unsafe output directory: {path}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Use --overwrite-output only for a directory created by this evaluator."
            )
        marker = path / OUTPUT_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8") != OUTPUT_MARKER_CONTENT:
            raise ValueError(
                f"Refusing to delete unmarked output directory: {path}. "
                f"Expected safety marker {marker}."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
    (path / OUTPUT_MARKER).write_text(OUTPUT_MARKER_CONTENT, encoding="utf-8")


def checkpoint_inventory(selected_model_ids: set[str] | None = None) -> list[dict[str, object]]:
    selected_model_ids = selected_model_ids or set()
    rows: list[dict[str, object]] = []
    for prop, prefix in (("homo", "3_homo"), ("lumo", "4_lumo"), ("gap", "5_gap")):
        for hidden in (16, 32):
            path = PROCEEDING_MODEL_DIR / f"{prefix}_GNN{hidden}.pth"
            model_id = f"proceeding_{prop}_3lay_h{hidden}"
            rows.append({
                "model_id": model_id,
                "era": "proceeding",
                "property": prop,
                "layers": 3,
                "hidden_dim": hidden,
                "checkpoint": str(path),
                "config": "",
                "sha256": sha256(path),
                "target_space": "native_eV",
                "selection_status": "selected" if model_id in selected_model_ids else "not_selected",
            })
        for layers in (2, 3):
            for hidden in (16, 32):
                stem = f"{prefix}_{layers}lay_h{hidden}_nodropout"
                path = JOURNAL_MODEL_DIR / f"{stem}.pth"
                config = JOURNAL_MODEL_DIR / f"{stem}_config.txt"
                model_id = f"journal_{prop}_{layers}lay_h{hidden}"
                rows.append({
                    "model_id": model_id,
                    "era": "journal",
                    "property": prop,
                    "layers": layers,
                    "hidden_dim": hidden,
                    "checkpoint": str(path),
                    "config": str(config),
                    "sha256": sha256(path),
                    "target_space": "train_standardized",
                    "selection_status": "selected" if model_id in selected_model_ids else "not_selected",
                })
    return rows


def select_inventory(inventory: list[dict[str, object]], smoke_only: bool) -> list[dict[str, object]]:
    selected = [
        row for row in inventory
        if not smoke_only or str(row["model_id"]) in SMOKE_MODEL_IDS
    ]
    expected_count = 2 if smoke_only else 18
    if len(selected) != expected_count:
        raise AssertionError(f"Expected {expected_count} selected models, found {len(selected)}")
    era_counts = {
        era: sum(row["era"] == era for row in selected)
        for era in ("proceeding", "journal")
    }
    property_counts = {
        prop: sum(row["property"] == prop for row in selected)
        for prop in ("homo", "lumo", "gap")
    }
    if smoke_only:
        if era_counts != {"proceeding": 1, "journal": 1} or property_counts != {"homo": 2, "lumo": 0, "gap": 0}:
            raise AssertionError(f"Invalid smoke selection: eras={era_counts}, properties={property_counts}")
    else:
        if era_counts != {"proceeding": 6, "journal": 12}:
            raise AssertionError(f"Invalid full era counts: {era_counts}")
        if property_counts != {"homo": 6, "lumo": 6, "gap": 6}:
            raise AssertionError(f"Invalid full property counts: {property_counts}")
    return selected


def selection_summary(selected: list[dict[str, object]], smoke_only: bool) -> dict[str, object]:
    return {
        "mode": "smoke_only" if smoke_only else "full_18models",
        "selected_count": len(selected),
        "era_counts": {
            era: sum(row["era"] == era for row in selected)
            for era in ("proceeding", "journal")
        },
        "property_counts": {
            prop: sum(row["property"] == prop for row in selected)
            for prop in ("homo", "lumo", "gap")
        },
        "models": [row["model_id"] for row in selected],
    }


def adapter_from_inventory(row: dict[str, object]) -> ProceedingAdapter | JournalAdapter:
    if row["era"] == "proceeding":
        return ProceedingAdapter(
            checkpoint_path=Path(str(row["checkpoint"])),
            property_name=str(row["property"]),
            hidden_dim=int(row["hidden_dim"]),
        )
    return JournalAdapter(
        checkpoint_path=Path(str(row["checkpoint"])),
        config_path=Path(str(row["config"])),
        property_name=str(row["property"]),
    )


def load_checkpoint_strict(model: torch.nn.Module, path: Path) -> tuple[list[str], list[str]]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError(f"{path} is not a tensor state_dict")
    result = model.load_state_dict(state, strict=True)
    return list(result.missing_keys), list(result.unexpected_keys)


def evaluate(
    model: torch.nn.Module,
    adapter: ProceedingAdapter | JournalAdapter,
    molecule_graphs,
    ft_graph: GraphBatch,
    test_cids: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.to(device)
    model.eval()
    if model.training:
        raise AssertionError("model.eval() did not disable training mode")
    ft_graph = GraphBatch(
        x=ft_graph.x.to(device),
        edge_index=ft_graph.edge_index.to(device),
        batch=ft_graph.batch.to(device),
    )
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        ft_output = model.encode_fringe_trees(ft_graph)
        for start in range(0, len(test_cids), batch_size):
            cids = test_cids[start:start + batch_size]
            batch = batch_molecules([molecule_graphs[cid] for cid in cids])
            batch = GraphBatch(
                x=batch.x.to(device),
                edge_index=batch.edge_index.to(device),
                batch=batch.batch.to(device),
                pos=batch.pos.to(device) if batch.pos is not None else None,
            )
            standardized_or_native = model.forward_interior(batch, ft_output).reshape(-1)
            outputs.append(adapter.to_native(standardized_or_native).cpu())
    return torch.cat(outputs).numpy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = y_true.astype(np.float64)
    pred = y_pred.astype(np.float64)
    mae = float(np.mean(np.abs(true - pred)))
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - float(np.sum((true - pred) ** 2)) / denominator
    return {
        "mae_eV": mae,
        "mae_meV": mae * 1000.0,
        "r2": r2,
        "prediction_min_eV": float(np.min(pred)),
        "prediction_max_eV": float(np.max(pred)),
        "prediction_mean_eV": float(np.mean(pred)),
    }


def main() -> int:
    args = parse_args()
    base_inventory = checkpoint_inventory()
    selected = select_inventory(base_inventory, smoke_only=args.smoke_only)
    selected_ids = {str(row["model_id"]) for row in selected}
    inventory = checkpoint_inventory(selected_ids)
    selected = select_inventory(inventory, smoke_only=args.smoke_only)
    selected_summary = selection_summary(selected, smoke_only=args.smoke_only)
    if args.list_models:
        print(json.dumps(selected_summary, ensure_ascii=False, indent=2))
        return 0
    if args.batch_size <= 0 or args.alternate_batch_size <= 0:
        raise SystemExit("Batch sizes must be positive")
    if args.batch_consistency_atol < 0 or args.batch_consistency_rtol < 0:
        raise SystemExit("Batch consistency tolerances must be non-negative")
    if args.batch_consistency_hard_threshold <= 0:
        raise SystemExit("--batch-consistency-hard-threshold must be positive")
    batch_consistency_mode = args.batch_consistency_mode or (
        "strict" if args.smoke_only else "warn"
    )
    output_dir = args.output_dir.resolve()
    prepare_output_directory(output_dir, overwrite=args.overwrite_output)
    predictions_dir = output_dir / "per_molecule_predictions"
    predictions_dir.mkdir()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    sdf_path = SCRIPT_DIR / "qm9_all.sdf"
    values_path = SCRIPT_DIR / "qm9_complete_values.csv"
    exclusion_path = SCRIPT_DIR / "qm9_uncharacterized.csv"
    excluded = load_excluded_cids(exclusion_path)
    all_cids = read_sdf_cids(sdf_path)
    filtered_cids = [cid for cid in all_cids if cid not in excluded]
    train_cids, validation_cids, test_cids = make_split(filtered_cids, seed=42)
    if len(test_cids) != 10_831:
        raise AssertionError(f"Unexpected test size: {len(test_cids)}")

    split_rows: list[dict[str, object]] = []
    memberships = (
        ("train", train_cids),
        ("validation", validation_cids),
        ("test", test_cids),
    )
    filtered_index = {cid: i for i, cid in enumerate(filtered_cids)}
    for partition, cids in memberships:
        for position, cid in enumerate(cids):
            split_rows.append({
                "filtered_dataset_index": filtered_index[cid],
                "molecule_id": cid,
                "partition": partition,
                "partition_position": position,
            })
    write_csv(
        output_dir / "qm9_split_seed42.csv",
        split_rows,
        ["filtered_dataset_index", "molecule_id", "partition", "partition_position"],
    )
    test_cid_hash = hashlib.sha256("\n".join(test_cids).encode("utf-8")).hexdigest()
    write_json(output_dir / "qm9_split_manifest.json", {
        "sdf": str(sdf_path),
        "sdf_sha256": sha256(sdf_path),
        "values": str(values_path),
        "values_sha256": sha256(values_path),
        "exclusion": str(exclusion_path),
        "exclusion_sha256": sha256(exclusion_path),
        "raw_count": len(all_cids),
        "excluded_count": len(excluded),
        "filtered_count": len(filtered_cids),
        "split_seed": 42,
        "train_n": len(train_cids),
        "validation_n": len(validation_cids),
        "test_n": len(test_cids),
        "test_cid_order_sha256": test_cid_hash,
        "first_test_cids": test_cids[:5],
        "last_test_cids": test_cids[-5:],
        "torch_version": torch.__version__,
    })

    write_csv(
        output_dir / "checkpoint_inventory.csv",
        inventory,
        list(inventory[0].keys()),
    )

    selected_properties = tuple(
        prop for prop in ("homo", "lumo", "gap")
        if any(row["property"] == prop for row in selected)
    )
    values = read_property_values(values_path, selected_properties)
    train_transforms: dict[str, tuple[float, float]] = {}
    y_true_by_property: dict[str, np.ndarray] = {}
    property_scales: dict[str, dict[str, object]] = {}
    for prop in selected_properties:
        y_train = torch.tensor([values[cid][prop] for cid in train_cids], dtype=torch.float32)
        train_transforms[prop] = (float(y_train.mean().item()), float(y_train.std().item()))
        y_true_by_property[prop] = np.asarray(
            [values[cid][prop] for cid in test_cids], dtype=np.float32
        )
        y_test64 = y_true_by_property[prop].astype(np.float64)
        property_scales[prop] = {
            "source_unit": PROPERTY_SOURCE_UNITS[prop],
            "source_to_native_factor": PROPERTY_CONVERSIONS[prop],
            "native_unit": PROPERTY_NATIVE_UNITS[prop],
            "train_mean_native": train_transforms[prop][0],
            "train_std_native": train_transforms[prop][1],
            "test_min_native": float(np.min(y_test64)),
            "test_max_native": float(np.max(y_test64)),
            "test_mean_native": float(np.mean(y_test64)),
            "test_std_native_population": float(np.std(y_test64, ddof=0)),
        }

    adapters: list[tuple[str, ProceedingAdapter | JournalAdapter]] = []
    for row in selected:
        adapter = adapter_from_inventory(row)
        if isinstance(adapter, JournalAdapter):
            train_mean, train_std = train_transforms[adapter.property_name]
            if adapter.y_mean != train_mean or adapter.y_std != train_std:
                raise AssertionError(
                    f"Journal transform mismatch for {row['model_id']}: "
                    f"config=({adapter.y_mean}, {adapter.y_std}), "
                    f"recalculated=({train_mean}, {train_std})"
                )
        adapters.append((str(row["model_id"]), adapter))

    start_graph_build = time.time()
    molecule_graphs, ft_graph = construct_selected_graphs(sdf_path, test_cids, SCRIPT_DIR)
    graph_build_seconds = time.time() - start_graph_build

    validation_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    transform_rows: list[dict[str, object]] = []
    prediction_by_id: dict[str, np.ndarray] = {}
    run_details: dict[str, object] = {}

    for model_id, adapter in adapters:
        prop = adapter.property_name
        y_true = y_true_by_property[prop]
        train_mean, train_std = train_transforms[prop]
        before_hash = sha256(adapter.checkpoint_path)
        model = adapter.build_model()
        missing, unexpected = load_checkpoint_strict(model, adapter.checkpoint_path)
        model.eval()
        dropout_disabled = not model.training and not model.GNN_2L_int.training and not model.GNN_FT.training
        if missing or unexpected or not dropout_disabled:
            raise AssertionError(f"Model validation failed for {model_id}")

        started = time.time()
        pred_primary = evaluate(
            model, adapter, molecule_graphs, ft_graph, test_cids, args.batch_size, device
        )
        pred_repeat = evaluate(
            model, adapter, molecule_graphs, ft_graph, test_cids, args.batch_size, device
        )
        pred_alternate = (
            evaluate(
                model, adapter, molecule_graphs, ft_graph, test_cids,
                args.alternate_batch_size, device,
            )
            if batch_consistency_mode != "off" else None
        )
        elapsed = time.time() - started
        repeat_exact = bool(np.array_equal(pred_primary, pred_repeat))
        repeat_max_diff = float(np.max(np.abs(pred_primary - pred_repeat)))
        batch_bitwise_equal = (
            bool(np.array_equal(pred_primary, pred_alternate))
            if pred_alternate is not None else None
        )
        batch_allclose = (
            bool(np.allclose(
                pred_primary,
                pred_alternate,
                rtol=args.batch_consistency_rtol,
                atol=args.batch_consistency_atol,
            ))
            if pred_alternate is not None else None
        )
        batch_max_diff = (
            float(np.max(np.abs(pred_primary - pred_alternate)))
            if pred_alternate is not None else None
        )
        batch_warning = bool(batch_allclose is False and batch_consistency_mode == "warn")
        if not repeat_exact:
            raise AssertionError(f"Repeated evaluation differs for {model_id}: {repeat_max_diff}")
        if batch_max_diff is not None and batch_max_diff > args.batch_consistency_hard_threshold:
            raise AssertionError(
                f"Batch-size evaluation exceeds hard threshold for {model_id}: "
                f"max_abs_diff={batch_max_diff} eV, "
                f"hard_threshold={args.batch_consistency_hard_threshold} eV"
            )
        if batch_allclose is False and batch_consistency_mode == "strict":
            raise AssertionError(
                f"Batch-size evaluation exceeds tolerance for {model_id}: "
                f"max_abs_diff={batch_max_diff} eV, "
                f"rtol={args.batch_consistency_rtol}, atol={args.batch_consistency_atol} eV"
            )
        after_hash = sha256(adapter.checkpoint_path)
        if before_hash != after_hash:
            raise AssertionError(f"Checkpoint changed during evaluation: {adapter.checkpoint_path}")

        model_metrics = metrics(y_true, pred_primary)
        prediction_by_id[model_id] = pred_primary
        metric_rows.append({
            "model_id": model_id,
            "era": adapter.era,
            "property": prop,
            "test_n": len(test_cids),
            "batch_size": args.batch_size,
            "alternate_batch_size": args.alternate_batch_size,
            "metrics_prediction_source": "primary_batch_size",
            **model_metrics,
            "checkpoint_sha256": before_hash,
            "test_cid_order_sha256": test_cid_hash,
            "elapsed_seconds_three_passes": elapsed,
        })
        validation_rows.append({
            "model_id": model_id,
            "strict_load": True,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "model_eval": not model.training,
            "dropout_disabled": dropout_disabled,
            "repeat_exact": repeat_exact,
            "repeat_max_abs_diff": repeat_max_diff,
            "batch_size_bitwise_equal": batch_bitwise_equal,
            "batch_size_allclose": batch_allclose,
            "batch_consistency_mode": batch_consistency_mode,
            "batch_consistency_warning": batch_warning,
            "batch_consistency_rtol": args.batch_consistency_rtol,
            "batch_consistency_atol_eV": args.batch_consistency_atol,
            "batch_consistency_hard_threshold_eV": args.batch_consistency_hard_threshold,
            "batch_size_max_abs_diff": batch_max_diff,
            "checkpoint_hash_before": before_hash,
            "checkpoint_hash_after": after_hash,
            "checkpoint_unchanged": before_hash == after_hash,
            "test_n": len(test_cids),
            "cid_order_sha256": test_cid_hash,
        })
        metadata = adapter.transform_metadata
        transform_rows.append({
            "model_id": model_id,
            "target_space": metadata["target_space"],
            "inverse_applied": metadata["inverse_applied"],
            "config_mean": metadata["mean"],
            "config_std": metadata["std"],
            "recalculated_train_mean": train_mean if adapter.era == "journal" else "",
            "recalculated_train_std": train_std if adapter.era == "journal" else "",
            "config_matches_recalculation": (
                adapter.y_mean == train_mean and adapter.y_std == train_std
                if isinstance(adapter, JournalAdapter) else "not_applicable"
            ),
            "std_definition": "torch.std correction=1 on train n=110000" if adapter.era == "journal" else "not_applicable",
            "formula": metadata["formula"],
        })
        prediction_path = predictions_dir / f"{model_id}.csv.gz"
        with gzip.open(prediction_path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["partition_position", "molecule_id", "y_true_eV", "y_pred_eV", "abs_error_eV"])
            for position, (cid, true, pred) in enumerate(zip(test_cids, y_true, pred_primary)):
                writer.writerow([position, cid, float(true), float(pred), abs(float(true) - float(pred))])
        run_details[model_id] = {
            "prediction_file": str(prediction_path),
            "prediction_rows": len(pred_primary),
            "cid_order_sha256": test_cid_hash,
        }

    if len(metric_rows) != len(selected):
        raise AssertionError(
            f"Metrics row count mismatch: selected={len(selected)}, metrics={len(metric_rows)}"
        )

    write_csv(output_dir / "checkpoint_validation.csv", validation_rows, list(validation_rows[0].keys()))
    write_csv(output_dir / "target_transform_audit.csv", transform_rows, list(transform_rows[0].keys()))
    write_csv(output_dir / "common_split_metrics.csv", metric_rows, list(metric_rows[0].keys()))

    paired: list[dict[str, object]] = []
    for prop in selected_properties:
        y_true = y_true_by_property[prop]
        for hidden in (16, 32):
            old_id = f"proceeding_{prop}_3lay_h{hidden}"
            new_id = f"journal_{prop}_3lay_h{hidden}"
            if old_id not in prediction_by_id or new_id not in prediction_by_id:
                continue
            old_abs = np.abs(y_true - prediction_by_id[old_id])
            new_abs = np.abs(y_true - prediction_by_id[new_id])
            paired.append({
                "comparison": f"{new_id}_minus_{old_id}",
                "property": prop,
                "hidden_dim": hidden,
                "n": len(test_cids),
                "mean_abs_error_difference_eV": float(np.mean(new_abs - old_abs)),
                "median_abs_error_difference_eV": float(np.median(new_abs - old_abs)),
                "journal_lower_abs_error_n": int(np.sum(new_abs < old_abs)),
                "proceeding_lower_abs_error_n": int(np.sum(old_abs < new_abs)),
                "ties_n": int(np.sum(old_abs == new_abs)),
                "cid_order_sha256": test_cid_hash,
            })
    if not paired:
        raise AssertionError("No matched proceeding/journal pairs were available")
    write_csv(output_dir / "paired_model_comparison.csv", paired, list(paired[0].keys()))

    result_filename = "smoke_test_results.json" if args.smoke_only else "full_evaluation_results.json"
    write_json(output_dir / result_filename, {
        "status": "PASS",
        "scope": selected_summary["mode"],
        "selection": selected_summary,
        "historical_metric_reproduction": False,
        "evaluation_definition": "deterministic raw-checkpoint common-split reevaluation without EMA",
        "test_n": len(test_cids),
        "test_cid_order_sha256": test_cid_hash,
        "graph_build_seconds": graph_build_seconds,
        "models": run_details,
        "metrics": metric_rows,
        "validations": validation_rows,
        "target_transforms": transform_rows,
    })
    write_json(output_dir / "evaluation_manifest.json", {
        "script": str(Path(__file__).resolve()),
        "scope": selected_summary["mode"],
        "selection": selected_summary,
        "models_evaluated": selected_summary["models"],
        "models_evaluated_count": len(selected),
        "metrics_row_count": len(metric_rows),
        "models_not_evaluated": 18 - len(selected),
        "property_scales": property_scales,
        "ema": False,
        "model_eval": True,
        "torch_inference_mode": True,
        "device": str(device),
        "batch_size": args.batch_size,
        "alternate_batch_size": args.alternate_batch_size,
        "metrics_prediction_source": "primary_batch_size",
        "batch_consistency": {
            "mode": batch_consistency_mode,
            "default_mode_rule": "strict for smoke; warn for full",
            "rtol": args.batch_consistency_rtol,
            "atol_eV": args.batch_consistency_atol,
            "hard_threshold_eV": args.batch_consistency_hard_threshold,
            "models": [
                {
                    "model_id": row["model_id"],
                    "bitwise_equal": row["batch_size_bitwise_equal"],
                    "allclose": row["batch_size_allclose"],
                    "max_abs_diff_eV": row["batch_size_max_abs_diff"],
                    "warning": row["batch_consistency_warning"],
                }
                for row in validation_rows
            ],
        },
        "output_overwrite_requested": args.overwrite_output,
        "output_safety_marker": OUTPUT_MARKER,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "output_directory": str(output_dir),
    })
    print(json.dumps({
        "status": "PASS",
        "scope": selected_summary["mode"],
        "models_evaluated_count": len(selected),
        "metrics_row_count": len(metric_rows),
        "output_dir": str(output_dir),
        "metrics": metric_rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
