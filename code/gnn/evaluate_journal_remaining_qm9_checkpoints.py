#!/usr/bin/env python3
"""Common-split evaluation of the journal properties not covered by HOMO/LUMO/GAP.

The default run evaluates 24 checkpoints (six properties x four architectures).
The already completed 12 journal HOMO/LUMO/GAP rows are imported from the formal
18-model evaluation and are not recomputed.  The combined metrics table therefore
contains all 36 journal checkpoints while retaining explicit provenance per row.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common_eval.journal_adapter import JournalAdapter, read_config
from common_eval.model_compat import GraphBatch
from common_eval.qm9_data import (
    PROPERTY_CONVERSIONS,
    PROPERTY_NATIVE_UNITS,
    PROPERTY_SOURCE_UNITS,
    construct_selected_graphs,
    load_excluded_cids,
    make_split,
    read_property_values,
    read_sdf_cids,
)
from evaluate_old_new_qm9_checkpoints import (
    BATCH_CONSISTENCY_ATOL,
    BATCH_CONSISTENCY_HARD_THRESHOLD,
    BATCH_CONSISTENCY_RTOL,
    JOURNAL_MODEL_DIR,
    SCRIPT_DIR,
    evaluate,
    load_checkpoint_strict,
    prepare_output_directory,
    sha256,
    write_csv,
    write_json,
)


PROPERTY_PREFIXES = {
    "mu": "1_mu",
    "alpha": "2_alpha",
    "homo": "3_homo",
    "lumo": "4_lumo",
    "gap": "5_gap",
    "r2": "6_r2",
    "zpve": "7_zpve",
    "cv": "12_cv",
    "u0_atom": "13_u0_atom",
}
ALREADY_EVALUATED_PROPERTIES = {"homo", "lumo", "gap"}
REMAINING_PROPERTIES = tuple(
    prop for prop in PROPERTY_PREFIXES if prop not in ALREADY_EVALUATED_PROPERTIES
)
ALL_PROPERTIES = tuple(PROPERTY_PREFIXES)
DEFAULT_FORMAL_HLG_DIR = SCRIPT_DIR / "audit" / "common_split_full_18models_cuda_v3"
TRANSFORM_CONSISTENCY_RTOL = 1e-6
TRANSFORM_CONSISTENCY_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the 24 remaining journal QM9 checkpoints on the fixed seed-42 "
            "test split and combine their metrics with the formal HOMO/LUMO/GAP results."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "audit" / "journal_remaining_24_common_split",
        help="New audit output directory (default: code/learning/audit/journal_remaining_24_common_split).",
    )
    parser.add_argument(
        "--existing-hlg-dir",
        type=Path,
        default=DEFAULT_FORMAL_HLG_DIR,
        help="Formal 18-model result directory whose 12 journal HOMO/LUMO/GAP rows are referenced.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alternate-batch-size", type=int, default=127)
    parser.add_argument(
        "--batch-consistency-atol",
        type=float,
        default=BATCH_CONSISTENCY_ATOL,
        help="Absolute allclose tolerance in each property's native unit (default: 1e-4).",
    )
    parser.add_argument(
        "--batch-consistency-rtol",
        type=float,
        default=BATCH_CONSISTENCY_RTOL,
        help="Relative allclose tolerance (default: 1e-5).",
    )
    parser.add_argument(
        "--batch-consistency-mode",
        choices=("strict", "warn", "off"),
        default=None,
        help="Batch consistency policy (default: warn for this full evaluator).",
    )
    parser.add_argument(
        "--batch-consistency-hard-threshold",
        type=float,
        default=BATCH_CONSISTENCY_HARD_THRESHOLD,
        help="Always stop above this max difference in native units (default: 1e-2).",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace only a marked output directory previously created by this evaluator.",
    )
    parser.add_argument(
        "--include-evaluated-properties",
        action="store_true",
        help="Evaluate all 36 journal checkpoints, including HOMO/LUMO/GAP, instead of the default 24.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the selected inventory and exit without creating outputs or loading checkpoint weights.",
    )
    return parser.parse_args()


def checkpoint_inventory(selected_ids: set[str] | None = None) -> list[dict[str, object]]:
    selected_ids = selected_ids or set()
    rows: list[dict[str, object]] = []
    for prop, prefix in PROPERTY_PREFIXES.items():
        configs = sorted(JOURNAL_MODEL_DIR.glob(f"{prefix}_*nodropout*_config.txt"))
        if len(configs) != 4:
            raise AssertionError(f"Expected four {prop} configs, found {len(configs)}")
        for config_path in configs:
            checkpoint_path = config_path.with_name(config_path.name.replace("_config.txt", ".pth"))
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            config = read_config(config_path)
            layers = int(config["num_layers_IN"])
            hidden = int(config["hidden_channels_IN"])
            model_id = f"journal_{prop}_{layers}lay_h{hidden}"
            rows.append({
                "model_id": model_id,
                "era": "journal",
                "property": prop,
                "layers": layers,
                "hidden_dim": hidden,
                "native_unit": PROPERTY_NATIVE_UNITS[prop],
                "checkpoint": str(checkpoint_path.resolve()),
                "config": str(config_path.resolve()),
                "checkpoint_sha256": sha256(checkpoint_path),
                "target_space": "train_standardized",
                "selection_status": (
                    "selected" if model_id in selected_ids else
                    "formal_hlg_reference" if prop in ALREADY_EVALUATED_PROPERTIES else
                    "not_selected"
                ),
            })
    model_ids = [str(row["model_id"]) for row in rows]
    if len(rows) != 36 or len(set(model_ids)) != 36:
        raise AssertionError(f"Expected 36 unique journal checkpoints, found {len(rows)} rows/{len(set(model_ids))} IDs")
    return rows


def select_inventory(inventory: list[dict[str, object]], include_hlg: bool) -> list[dict[str, object]]:
    selected = [
        row for row in inventory
        if include_hlg or row["property"] not in ALREADY_EVALUATED_PROPERTIES
    ]
    expected = 36 if include_hlg else 24
    expected_properties = ALL_PROPERTIES if include_hlg else REMAINING_PROPERTIES
    counts = Counter(str(row["property"]) for row in selected)
    if len(selected) != expected or counts != Counter({prop: 4 for prop in expected_properties}):
        raise AssertionError(f"Invalid selection: expected={expected}, actual={len(selected)}, property_counts={dict(counts)}")
    return selected


def selection_summary(selected: list[dict[str, object]], include_hlg: bool) -> dict[str, object]:
    counts = Counter(str(row["property"]) for row in selected)
    return {
        "mode": "all_36_journal" if include_hlg else "remaining_24_journal",
        "selected_count": len(selected),
        "property_counts": {prop: counts.get(prop, 0) for prop in ALL_PROPERTIES},
        "models": [row["model_id"] for row in selected],
        "homo_lumo_gap_recomputed": include_hlg,
    }


def native_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = y_true.astype(np.float64)
    pred = y_pred.astype(np.float64)
    mae = float(np.mean(np.abs(true - pred)))
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    if denominator == 0.0:
        raise ZeroDivisionError("Cannot calculate R2 for a constant test target")
    return {
        "mae_native": mae,
        "r2": 1.0 - float(np.sum((true - pred) ** 2)) / denominator,
        "prediction_min_native": float(np.min(pred)),
        "prediction_max_native": float(np.max(pred)),
        "prediction_mean_native": float(np.mean(pred)),
    }


def load_formal_hlg_metrics(result_dir: Path, expected_test_hash: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    metrics_path = result_dir / "common_split_metrics.csv"
    manifest_path = result_dir / "evaluation_manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Formal HOMO/LUMO/GAP outputs not found in {result_dir}; "
            "pass --existing-hlg-dir with the completed 18-model result directory"
        )
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    journal_rows = [row for row in source_rows if row["era"] == "journal"]
    counts = Counter(row["property"] for row in journal_rows)
    if len(source_rows) != 18 or len(journal_rows) != 12 or counts != Counter({"homo": 4, "lumo": 4, "gap": 4}):
        raise AssertionError(
            f"Formal result must contain 18 total rows and 12 journal H/L/G rows; "
            f"found {len(source_rows)}/{len(journal_rows)}, counts={dict(counts)}"
        )
    if any(row["test_cid_order_sha256"] != expected_test_hash for row in journal_rows):
        raise AssertionError("Formal HOMO/LUMO/GAP test CID order differs from this run")
    combined_rows: list[dict[str, object]] = []
    for row in journal_rows:
        model_id = row["model_id"]
        architecture = model_id.rsplit("_", 2)
        layers = int(architecture[-2].removesuffix("lay"))
        hidden = int(architecture[-1].removeprefix("h"))
        combined_rows.append({
            "model_id": model_id,
            "era": "journal",
            "property": row["property"],
            "layers": layers,
            "hidden_dim": hidden,
            "test_n": int(row["test_n"]),
            "native_unit": "eV",
            "mae_native": float(row["mae_eV"]),
            "r2": float(row["r2"]),
            "prediction_min_native": float(row["prediction_min_eV"]),
            "prediction_max_native": float(row["prediction_max_eV"]),
            "prediction_mean_native": float(row["prediction_mean_eV"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "test_cid_order_sha256": row["test_cid_order_sha256"],
            "evaluation_source": "formal_18_model_run_reference",
            "newly_evaluated": False,
            "metrics_prediction_source": "primary_batch_size_in_formal_run",
            "elapsed_seconds_three_passes": row["elapsed_seconds_three_passes"],
        })
    reference = {
        "directory": str(result_dir.resolve()),
        "metrics_file": str(metrics_path.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "manifest_file": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "journal_rows_imported": len(journal_rows),
        "properties": ["homo", "lumo", "gap"],
        "recomputed": False,
    }
    return combined_rows, reference


def main() -> int:
    args = parse_args()
    base_inventory = checkpoint_inventory()
    selected = select_inventory(base_inventory, args.include_evaluated_properties)
    selected_ids = {str(row["model_id"]) for row in selected}
    inventory = checkpoint_inventory(selected_ids)
    selected = select_inventory(inventory, args.include_evaluated_properties)
    summary = selection_summary(selected, args.include_evaluated_properties)
    if args.list_models:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.batch_size <= 0 or args.alternate_batch_size <= 0:
        raise SystemExit("Batch sizes must be positive")
    if args.batch_consistency_atol < 0 or args.batch_consistency_rtol < 0:
        raise SystemExit("Batch consistency tolerances must be non-negative")
    if args.batch_consistency_hard_threshold <= 0:
        raise SystemExit("--batch-consistency-hard-threshold must be positive")
    batch_consistency_mode = args.batch_consistency_mode or "warn"

    output_dir = args.output_dir.resolve()
    prepare_output_directory(output_dir, args.overwrite_output)
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
    if (len(train_cids), len(validation_cids), len(test_cids)) != (110_000, 10_000, 10_831):
        raise AssertionError("Unexpected split sizes")
    test_cid_hash = hashlib.sha256("\n".join(test_cids).encode("utf-8")).hexdigest()

    write_csv(output_dir / "checkpoint_inventory.csv", inventory, list(inventory[0].keys()))
    write_json(output_dir / "qm9_split_manifest.json", {
        "sdf": str(sdf_path.resolve()),
        "sdf_sha256": sha256(sdf_path),
        "values": str(values_path.resolve()),
        "values_sha256": sha256(values_path),
        "exclusion": str(exclusion_path.resolve()),
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
    })

    selected_properties = ALL_PROPERTIES if args.include_evaluated_properties else REMAINING_PROPERTIES
    # Scale metadata covers the complete 36-row journal table, including the
    # formally referenced HOMO/LUMO/GAP rows, without evaluating those models.
    values = read_property_values(values_path, ALL_PROPERTIES)
    transforms: dict[str, tuple[float, float]] = {}
    y_true_by_property: dict[str, np.ndarray] = {}
    property_scales: dict[str, dict[str, object]] = {}
    for prop in ALL_PROPERTIES:
        y_train = torch.tensor([values[cid][prop] for cid in train_cids], dtype=torch.float32)
        transforms[prop] = (float(y_train.mean().item()), float(y_train.std().item()))
        y_true_by_property[prop] = np.asarray([values[cid][prop] for cid in test_cids], dtype=np.float32)
        y_test64 = y_true_by_property[prop].astype(np.float64)
        property_scales[prop] = {
            "source_unit": PROPERTY_SOURCE_UNITS[prop],
            "source_to_native_factor": PROPERTY_CONVERSIONS[prop],
            "native_unit": PROPERTY_NATIVE_UNITS[prop],
            "train_mean_native": transforms[prop][0],
            "train_std_native": transforms[prop][1],
            "test_min_native": float(np.min(y_test64)),
            "test_max_native": float(np.max(y_test64)),
            "test_mean_native": float(np.mean(y_test64)),
            "test_std_native_population": float(np.std(y_test64, ddof=0)),
        }

    adapters: list[tuple[dict[str, object], JournalAdapter]] = []
    for row in selected:
        adapter = JournalAdapter(
            checkpoint_path=Path(str(row["checkpoint"])),
            config_path=Path(str(row["config"])),
            property_name=str(row["property"]),
        )
        train_mean, train_std = transforms[adapter.property_name]
        mean_allclose = bool(np.isclose(
            adapter.y_mean, train_mean,
            rtol=TRANSFORM_CONSISTENCY_RTOL, atol=TRANSFORM_CONSISTENCY_ATOL,
        ))
        std_allclose = bool(np.isclose(
            adapter.y_std, train_std,
            rtol=TRANSFORM_CONSISTENCY_RTOL, atol=TRANSFORM_CONSISTENCY_ATOL,
        ))
        if not mean_allclose or not std_allclose:
            raise AssertionError(
                f"Transform mismatch for {row['model_id']}: "
                f"config=({adapter.y_mean}, {adapter.y_std}), recalculated=({train_mean}, {train_std}), "
                f"rtol={TRANSFORM_CONSISTENCY_RTOL}, atol={TRANSFORM_CONSISTENCY_ATOL}"
            )
        adapters.append((row, adapter))

    graph_started = time.time()
    molecule_graphs, ft_graph = construct_selected_graphs(sdf_path, test_cids, SCRIPT_DIR)
    graph_seconds = time.time() - graph_started

    metric_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    transform_rows: list[dict[str, object]] = []
    run_details: dict[str, object] = {}
    for row, adapter in adapters:
        model_id = str(row["model_id"])
        prop = adapter.property_name
        unit = PROPERTY_NATIVE_UNITS[prop]
        y_true = y_true_by_property[prop]
        before_hash = sha256(adapter.checkpoint_path)
        model = adapter.build_model()
        missing, unexpected = load_checkpoint_strict(model, adapter.checkpoint_path)
        model.eval()
        dropout_disabled = not model.training and not model.GNN_2L_int.training and not model.GNN_FT.training
        if missing or unexpected or not dropout_disabled:
            raise AssertionError(f"Checkpoint/model validation failed for {model_id}")

        started = time.time()
        primary = evaluate(model, adapter, molecule_graphs, ft_graph, test_cids, args.batch_size, device)
        repeat = evaluate(model, adapter, molecule_graphs, ft_graph, test_cids, args.batch_size, device)
        alternate = (
            evaluate(
                model, adapter, molecule_graphs, ft_graph, test_cids,
                args.alternate_batch_size, device,
            )
            if batch_consistency_mode != "off" else None
        )
        elapsed = time.time() - started
        repeat_exact = bool(np.array_equal(primary, repeat))
        repeat_max = float(np.max(np.abs(primary - repeat)))
        batch_bitwise = (
            bool(np.array_equal(primary, alternate)) if alternate is not None else None
        )
        batch_allclose = (
            bool(np.allclose(
                primary, alternate,
                rtol=args.batch_consistency_rtol,
                atol=args.batch_consistency_atol,
            ))
            if alternate is not None else None
        )
        batch_max = (
            float(np.max(np.abs(primary - alternate))) if alternate is not None else None
        )
        batch_warning = bool(batch_allclose is False and batch_consistency_mode == "warn")
        if not repeat_exact:
            raise AssertionError(f"Repeated evaluation differs for {model_id}: {repeat_max} {unit}")
        if batch_max is not None and batch_max > args.batch_consistency_hard_threshold:
            raise AssertionError(
                f"Batch-size evaluation exceeds hard threshold for {model_id}: "
                f"max_abs_diff={batch_max} {unit}, "
                f"hard_threshold={args.batch_consistency_hard_threshold} {unit}"
            )
        if batch_allclose is False and batch_consistency_mode == "strict":
            raise AssertionError(
                f"Batch-size evaluation exceeds tolerance for {model_id}: "
                f"max_abs_diff={batch_max} {unit}, rtol={args.batch_consistency_rtol}, "
                f"atol={args.batch_consistency_atol} {unit}"
            )
        after_hash = sha256(adapter.checkpoint_path)
        if before_hash != after_hash:
            raise AssertionError(f"Checkpoint changed during evaluation: {adapter.checkpoint_path}")

        result = native_metrics(y_true, primary)
        metric_rows.append({
            "model_id": model_id,
            "era": "journal",
            "property": prop,
            "layers": row["layers"],
            "hidden_dim": row["hidden_dim"],
            "test_n": len(test_cids),
            "native_unit": unit,
            **result,
            "checkpoint_sha256": before_hash,
            "test_cid_order_sha256": test_cid_hash,
            "evaluation_source": "current_remaining_property_run",
            "newly_evaluated": True,
            "metrics_prediction_source": "primary_batch_size",
            "elapsed_seconds_three_passes": elapsed,
        })
        validation_rows.append({
            "model_id": model_id,
            "property": prop,
            "native_unit": unit,
            "checkpoint": str(adapter.checkpoint_path.resolve()),
            "config": str(adapter.config_path.resolve()),
            "strict_load": True,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "model_eval": not model.training,
            "dropout_disabled": dropout_disabled,
            "repeat_exact": repeat_exact,
            "repeat_max_abs_diff_native": repeat_max,
            "batch_size_bitwise_equal": batch_bitwise,
            "batch_size_allclose": batch_allclose,
            "batch_consistency_mode": batch_consistency_mode,
            "batch_consistency_warning": batch_warning,
            "batch_consistency_rtol": args.batch_consistency_rtol,
            "batch_consistency_atol_native": args.batch_consistency_atol,
            "batch_consistency_hard_threshold_native": args.batch_consistency_hard_threshold,
            "batch_size_max_abs_diff_native": batch_max,
            "checkpoint_hash_before": before_hash,
            "checkpoint_hash_after": after_hash,
            "checkpoint_unchanged": before_hash == after_hash,
            "test_n": len(test_cids),
            "cid_order_sha256": test_cid_hash,
        })
        train_mean, train_std = transforms[prop]
        mean_exact = adapter.y_mean == train_mean
        std_exact = adapter.y_std == train_std
        mean_allclose = bool(np.isclose(
            adapter.y_mean, train_mean,
            rtol=TRANSFORM_CONSISTENCY_RTOL, atol=TRANSFORM_CONSISTENCY_ATOL,
        ))
        std_allclose = bool(np.isclose(
            adapter.y_std, train_std,
            rtol=TRANSFORM_CONSISTENCY_RTOL, atol=TRANSFORM_CONSISTENCY_ATOL,
        ))
        transform_rows.append({
            "model_id": model_id,
            "property": prop,
            "source_unit": PROPERTY_SOURCE_UNITS[prop],
            "source_to_native_factor": PROPERTY_CONVERSIONS[prop],
            "native_unit": unit,
            "target_space": adapter.target_space,
            "inverse_applied": True,
            "config_mean_native": adapter.y_mean,
            "config_std_native": adapter.y_std,
            "recalculated_train_mean_native": train_mean,
            "recalculated_train_std_native": train_std,
            "mean_abs_difference_native": abs(adapter.y_mean - train_mean),
            "std_abs_difference_native": abs(adapter.y_std - train_std),
            "mean_exact": mean_exact,
            "std_exact": std_exact,
            "mean_allclose": mean_allclose,
            "std_allclose": std_allclose,
            "config_transform_allclose": mean_allclose and std_allclose,
            "transform_consistency_rtol": TRANSFORM_CONSISTENCY_RTOL,
            "transform_consistency_atol_native": TRANSFORM_CONSISTENCY_ATOL,
            "train_n": len(train_cids),
            "std_definition": "torch.std correction=1 on train n=110000",
            "formula": "y_native = y_standardized * train_std + train_mean",
        })

        prediction_path = predictions_dir / f"{model_id}.csv.gz"
        with gzip.open(prediction_path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "partition_position", "molecule_id", "property", "native_unit",
                "y_true_native", "y_pred_native", "abs_error_native",
            ])
            for position, (cid, true, pred) in enumerate(zip(test_cids, y_true, primary)):
                writer.writerow([position, cid, prop, unit, float(true), float(pred), abs(float(true) - float(pred))])
        run_details[model_id] = {
            "prediction_file": str(prediction_path.resolve()),
            "prediction_rows": len(primary),
            "cid_order_sha256": test_cid_hash,
        }

    if len(metric_rows) != len(selected):
        raise AssertionError(f"Expected {len(selected)} new metric rows, found {len(metric_rows)}")

    formal_reference: dict[str, object] | None = None
    if args.include_evaluated_properties:
        combined_metrics = metric_rows
    else:
        formal_rows, formal_reference = load_formal_hlg_metrics(args.existing_hlg_dir.resolve(), test_cid_hash)
        combined_metrics = formal_rows + metric_rows
    combined_counts = Counter(str(row["property"]) for row in combined_metrics)
    if len(combined_metrics) != 36 or combined_counts != Counter({prop: 4 for prop in ALL_PROPERTIES}):
        raise AssertionError(f"Combined journal metrics are incomplete: rows={len(combined_metrics)}, counts={dict(combined_counts)}")
    combined_metrics.sort(key=lambda row: (ALL_PROPERTIES.index(str(row["property"])), int(row["layers"]), int(row["hidden_dim"])))

    write_csv(output_dir / "journal_all_properties_metrics.csv", combined_metrics, list(combined_metrics[0].keys()))
    write_csv(output_dir / "checkpoint_validation.csv", validation_rows, list(validation_rows[0].keys()))
    write_csv(output_dir / "target_transform_audit.csv", transform_rows, list(transform_rows[0].keys()))
    manifest = {
        "status": "PASS",
        "script": str(Path(__file__).resolve()),
        "scope": summary["mode"],
        "selection": summary,
        "evaluation_definition": "deterministic raw-checkpoint seed-42 common-split reevaluation without EMA",
        "new_models_evaluated_count": len(metric_rows),
        "combined_journal_metrics_count": len(combined_metrics),
        "combined_property_counts": {prop: combined_counts[prop] for prop in ALL_PROPERTIES},
        "formal_homo_lumo_gap_reference": formal_reference,
        "existing_homo_lumo_gap_recomputed": args.include_evaluated_properties,
        "ema": False,
        "raw_checkpoint_weights": True,
        "model_eval": True,
        "torch_inference_mode": True,
        "split_seed": 42,
        "train_n": len(train_cids),
        "validation_n": len(validation_cids),
        "test_n": len(test_cids),
        "test_cid_order_sha256": test_cid_hash,
        "property_scales": property_scales,
        "device": str(device),
        "batch_size": args.batch_size,
        "alternate_batch_size": args.alternate_batch_size,
        "metrics_prediction_source": "primary_batch_size",
        "batch_consistency": {
            "mode": batch_consistency_mode,
            "default_mode_rule": "warn for full journal-property evaluation",
            "rtol": args.batch_consistency_rtol,
            "atol_native": args.batch_consistency_atol,
            "hard_threshold_native": args.batch_consistency_hard_threshold,
            "models": [
                {
                    "model_id": row["model_id"],
                    "native_unit": row["native_unit"],
                    "bitwise_equal": row["batch_size_bitwise_equal"],
                    "allclose": row["batch_size_allclose"],
                    "max_abs_diff_native": row["batch_size_max_abs_diff_native"],
                    "warning": row["batch_consistency_warning"],
                }
                for row in validation_rows
            ],
        },
        "transform_consistency": {
            "comparison": "config mean/std versus float32 recalculation on the fixed train split",
            "rtol": TRANSFORM_CONSISTENCY_RTOL,
            "atol_native": TRANSFORM_CONSISTENCY_ATOL,
            "all_models_allclose": all(row["config_transform_allclose"] for row in transform_rows),
            "all_models_bitwise_exact": all(row["mean_exact"] and row["std_exact"] for row in transform_rows),
            "inverse_transform_source": "checkpoint-associated config mean/std",
        },
        "graph_build_seconds": graph_seconds,
        "models": run_details,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "output_directory": str(output_dir),
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps({
        "status": "PASS",
        "scope": summary["mode"],
        "new_models_evaluated_count": len(metric_rows),
        "combined_journal_metrics_count": len(combined_metrics),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
