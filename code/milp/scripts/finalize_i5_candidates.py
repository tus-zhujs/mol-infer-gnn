#!/usr/bin/env python3
"""Recheck I5/I5-prime incumbents and prepare the production DFT manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / "src")]

from mol_infer_gnn_experiments.canonical import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from mol_infer_gnn_experiments.phase2.verification import (  # noqa: E402
    PREDICTOR_TOLERANCE,
)
from src.config import GNNModelLoader  # noqa: E402
from src_inverter.inverter import create_inverter  # noqa: E402
from src_milp.seed_graph import create_seed_graph  # noqa: E402

CAMPAIGN_ID = "phase2-v2-960-sequential-thread0-20260727"
CONDITIONS = REPOSITORY / "data/phase2/conditions/production-conditions.jsonl"
ASSET_ROOT = REPOSITORY / "data/phase1"
FINAL_STATUSES = (
    "ACCEPTED_CANDIDATE",
    "REJECTED_GRAPH_MISMATCH",
    "REJECTED_GNN_MISMATCH",
    "REJECTED_TARGET_MISS",
    "REJECTED_RECONSTRUCTION_FAILURE",
    "REJECTED_MISSING_EVIDENCE",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"), overwrite=True)


def completed_attempt(campaign_root: Path, condition_id: str) -> Path | None:
    root = campaign_root / "runs" / CAMPAIGN_ID / condition_id
    attempts = sorted(root.glob("attempt-[0-9][0-9][0-9][0-9]"))
    completed = [path for path in attempts if (path / "COMPLETED.json").is_file()]
    return completed[-1] if completed else None


def graph_value(molecule: Chem.Mol) -> dict[str, Any]:
    return {
        "atoms": [atom.GetSymbol() for atom in molecule.GetAtoms()],
        "bonds": sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) + 1,
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) + 1,
                int(round(bond.GetBondTypeAsDouble())),
            )
            for bond in molecule.GetBonds()
        ),
    }


def retained_graph_sha256(molecule: Chem.Mol) -> str:
    return hashlib.sha256(
        json.dumps(
            graph_value(molecule), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def raw_labeled_graph(molecule: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()
    for atom in molecule.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            element=atom.GetSymbol(),
            formal_charge=atom.GetFormalCharge(),
        )
    for bond in molecule.GetBonds():
        order = bond.GetBondTypeAsDouble()
        if not float(order).is_integer():
            raise ValueError(f"raw SDF contains noninteger bond order: {order}")
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_order=int(order),
        )
    return graph


def strict_graph_isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    return nx.is_isomorphic(
        left,
        right,
        node_match=nx.algorithms.isomorphism.categorical_node_match(
            ["element", "formal_charge"], [None, None]
        ),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match(
            "bond_order", None
        ),
    )


def indexed_graph_equal(left: nx.Graph, right: nx.Graph) -> bool:
    return (
        set(left.nodes) == set(right.nodes)
        and all(left.nodes[index] == right.nodes[index] for index in left.nodes)
        and {
            (min(first, second), max(first, second), data["bond_order"])
            for first, second, data in left.edges(data=True)
        }
        == {
            (min(first, second), max(first, second), data["bond_order"])
            for first, second, data in right.edges(data=True)
        }
    )


def _topology_isomorphic(
    left: nx.Graph, right: nx.Graph, *, labeled_nodes: bool
) -> bool:
    node_match = (
        nx.algorithms.isomorphism.categorical_node_match(
            ["element", "formal_charge"], [None, None]
        )
        if labeled_nodes
        else None
    )
    return nx.is_isomorphic(left, right, node_match=node_match)


def mismatch_cause(
    raw_graph: nx.Graph,
    intended_graph: nx.Graph,
    raw_molecule: Chem.Mol,
    sanitized_molecule: Chem.Mol | None,
    *,
    prior_indexed_match: bool,
) -> str:
    if strict_graph_isomorphic(raw_graph, intended_graph):
        if not indexed_graph_equal(raw_graph, intended_graph):
            return "atom indexing/order only"
        sanitized_changed = False
        if sanitized_molecule is not None:
            sanitized_changed = any(
                bond.GetIsAromatic()
                or not float(bond.GetBondTypeAsDouble()).is_integer()
                or not raw_graph.has_edge(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                )
                or int(round(bond.GetBondTypeAsDouble()))
                != raw_graph.edges[
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                ]["bond_order"]
                for bond in sanitized_molecule.GetBonds()
            )
        if sanitized_changed and not prior_indexed_match:
            return "aromaticity/sanitization conversion"
        return "another cause"
    if _topology_isomorphic(raw_graph, intended_graph, labeled_nodes=True):
        return "bond-order difference"
    raw_heavy = raw_graph.subgraph(
        [index for index, data in raw_graph.nodes(data=True) if data["element"] != "H"]
    )
    intended_heavy = intended_graph.subgraph(
        [index for index, data in intended_graph.nodes(data=True) if data["element"] != "H"]
    )
    if strict_graph_isomorphic(raw_heavy, intended_heavy):
        return "hydrogen representation"
    if _topology_isomorphic(raw_graph, intended_graph, labeled_nodes=False):
        return "atom/charge difference"
    return "actual connectivity difference"


class MilpGraphReconstructor:
    """Recreate the literal intended SDF from retained structural MILP values."""

    def __init__(self) -> None:
        seed_graph = create_seed_graph("2LMM_GNN", "I5")
        instance_root = REPOSITORY / "data/phase2/instances/canonical"
        seed_graph.read_seed_graph(instance_root / "I5.txt")
        seed_graph.read_fringe_trees(instance_root / "I5-fringe.txt")
        seed_graph.set_up_seed_graph()
        seed_graph.set_up_variables()
        self.seed_graph = seed_graph
        self.variables: dict[str, Any] = {}
        seen: set[int] = set()

        def collect(value: Any) -> None:
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            name = getattr(value, "name", None)
            if isinstance(name, str) and hasattr(value, "varValue"):
                self.variables[name] = value
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    collect(key)
                    collect(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    collect(item)

        collect(seed_graph.__dict__)

    def reconstruct(self, attempt: Path, output_prefix: Path) -> Path:
        dump_path = attempt / "solver/full-precision-variables.jsonl"
        if not dump_path.is_file():
            raise FileNotFoundError(dump_path)
        values = {
            record["name"]: float(record["value"])
            for record in (
                json.loads(line)
                for line in dump_path.read_text(encoding="utf-8").splitlines()
                if line
            )
        }
        for variable in self.variables.values():
            variable.varValue = None
        for name, variable in self.variables.items():
            if name in values:
                variable.varValue = values[name]
        with redirect_stdout(io.StringIO()):
            self.seed_graph._print_sdf(str(output_prefix))
        return output_prefix.with_suffix(".sdf")


def canonical_graph_identity(molecule: Chem.Mol) -> tuple[str, str]:
    explicit = Chem.Mol(molecule)
    smiles = Chem.MolToSmiles(
        explicit,
        canonical=True,
        isomericSmiles=False,
        allBondsExplicit=True,
        allHsExplicit=True,
    )
    value = {"canonical_explicit_smiles": smiles}
    return smiles, hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class DirectReplay:
    def __init__(self) -> None:
        inventory = read_json(ASSET_ROOT / "asset-inventory.json")
        configs = read_json(ASSET_ROOT / "config-inventory.json")
        self.checkpoints = {
            record["logical_asset_id"]: record
            for record in inventory["checkpoint_records"]
        }
        self.configs = {
            record["logical_asset_id"]: record for record in configs
        }
        self.inverters: dict[str, Any] = {}

    def inverter(self, checkpoint_id: str) -> Any:
        if checkpoint_id not in self.inverters:
            checkpoint = self.checkpoints[checkpoint_id]
            config = self.configs[checkpoint_id]
            loader = GNNModelLoader.from_dict(
                {
                    "name": checkpoint_id,
                    "model_filename": str(
                        ASSET_ROOT / checkpoint["destination_path"]
                    ),
                    "config_filename": str(ASSET_ROOT / config["destination_path"]),
                }
            )
            inverter = create_inverter(loader.type, checkpoint_id)
            inverter.set_up_inverter(loader)
            inverter.GNN.eval()
            self.inverters[checkpoint_id] = inverter
        return self.inverters[checkpoint_id]

    def predict(self, checkpoint_id: str, sdf_path: Path) -> tuple[float, float]:
        inverter = self.inverter(checkpoint_id)
        standardized = float(inverter.inspection(str(sdf_path.resolve())))
        physical = standardized * float(inverter.y_std) + float(inverter.y_mean)
        return standardized, physical


def evidence_value(path: Path, *keys: str) -> Any:
    if not path.is_file():
        return None
    value: Any = read_json(path)
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def classify_incumbent(
    condition: dict[str, Any],
    attempt: Path,
    replay: DirectReplay,
    reconstructor: MilpGraphReconstructor,
    reconstruction_root: Path,
) -> tuple[dict[str, Any], Chem.Mol | None]:
    sdf_path = attempt / "candidate/candidate.sdf"
    structural_path = attempt / "verification/structural-reconstruction.json"
    predictor_path = attempt / "verification/predictor-equivalence.json"
    variable_path = attempt / "verification/variable-audit.json"
    row_path = attempt / "verification/linear-row-audit.json"
    raw_path = attempt / "solver/raw-termination.json"
    status = read_json(attempt / "status.json")
    raw = read_json(raw_path) if raw_path.is_file() else {}
    row: dict[str, Any] = {
        "condition_id": condition["condition_id"],
        "specification": "I5" if condition["population"] == "baseline" else "I5-prime",
        "population": condition["population"],
        "property": condition["target_property"],
        "architecture": f"MP{condition['model_depth']}-{condition['hidden_width']}",
        "architecture_id": condition["architecture_id"],
        "checkpoint_id": condition["checkpoint_id"],
        "target_range_index": condition["range_index"],
        "target_lower": condition["target_lower_inclusive"],
        "target_upper": condition["target_upper_inclusive"],
        "original_solver_status": raw.get("native_status"),
        "original_termination_reason": raw.get("termination_reason"),
        "original_operational_status": status.get("normalized_status"),
        "solver_runtime": status.get("solver_runtime_seconds"),
        "incumbent_available": True,
        "direct_gnn_prediction": None,
        "direct_physical_prediction": None,
        "milp_prediction": evidence_value(
            predictor_path, "milp_standardized_prediction"
        )
        or status.get("milp_standardized_prediction"),
        "direct_milp_discrepancy": None,
        "direct_target_membership": False,
        "rdkit_readable": False,
        "connected": False,
        "sanitization_valid": False,
        "prior_graph_adjacency_exact_match": None,
        "graph_check": False,
        "raw_indexed_graph_match": False,
        "raw_graph_isomorphic": False,
        "graph_mismatch_cause": "",
        "raw_graph_sha256": "",
        "intended_graph_sha256": "",
        "maximum_integrality_residual": evidence_value(
            variable_path, "maximum_integrality_residual"
        ),
        "maximum_scaled_row_residual": evidence_value(
            row_path, "maximum_scaled_violation"
        ),
        "final_scientific_status": "REJECTED_MISSING_EVIDENCE",
        "warnings": "",
        "sdf_path": str(sdf_path.resolve()),
        "attempt_path": str(attempt.resolve()),
        "graph_id": "",
    }
    if not sdf_path.is_file() or not structural_path.is_file() or row["milp_prediction"] is None:
        return row, None
    raw_supplier = Chem.SDMolSupplier(
        str(sdf_path), removeHs=False, sanitize=False
    )
    raw_molecule = raw_supplier[0] if len(raw_supplier) else None
    if raw_molecule is None:
        row["final_scientific_status"] = "REJECTED_RECONSTRUCTION_FAILURE"
        return row, None
    row["rdkit_readable"] = True
    sanitized_molecule = Chem.Mol(raw_molecule)
    try:
        Chem.SanitizeMol(sanitized_molecule)
        row["sanitization_valid"] = True
    except Exception as exc:
        row["warnings"] = f"RDKit sanitization failed: {type(exc).__name__}: {exc}"
        row["final_scientific_status"] = "REJECTED_RECONSTRUCTION_FAILURE"
        return row, raw_molecule
    row["connected"] = len(Chem.GetMolFrags(sanitized_molecule)) == 1
    if not row["connected"]:
        row["final_scientific_status"] = "REJECTED_RECONSTRUCTION_FAILURE"
        return row, sanitized_molecule
    structural = read_json(structural_path)
    row["prior_graph_adjacency_exact_match"] = structural.get(
        "adjacency_exact_match"
    )
    intended_path = reconstructor.reconstruct(
        attempt,
        reconstruction_root / f"{condition['condition_id']}-intended",
    )
    intended_supplier = Chem.SDMolSupplier(
        str(intended_path), removeHs=False, sanitize=False
    )
    intended_molecule = (
        intended_supplier[0] if len(intended_supplier) else None
    )
    if intended_molecule is None:
        row["final_scientific_status"] = "REJECTED_MISSING_EVIDENCE"
        return row, sanitized_molecule
    raw_graph = raw_labeled_graph(raw_molecule)
    intended_graph = raw_labeled_graph(intended_molecule)
    indexed_match = indexed_graph_equal(raw_graph, intended_graph)
    graph_matches = strict_graph_isomorphic(raw_graph, intended_graph)
    cause = mismatch_cause(
        raw_graph,
        intended_graph,
        raw_molecule,
        sanitized_molecule,
        prior_indexed_match=bool(structural.get("adjacency_exact_match")),
    )
    row["raw_indexed_graph_match"] = indexed_match
    row["raw_graph_isomorphic"] = graph_matches
    row["graph_mismatch_cause"] = cause
    row["raw_graph_sha256"] = hashlib.sha256(
        canonical_json_bytes(nx.node_link_data(raw_graph, edges="edges"))
    ).hexdigest()
    row["intended_graph_sha256"] = hashlib.sha256(
        canonical_json_bytes(nx.node_link_data(intended_graph, edges="edges"))
    ).hexdigest()
    row["graph_check"] = graph_matches
    if not graph_matches:
        row["final_scientific_status"] = "REJECTED_GRAPH_MISMATCH"
        return row, sanitized_molecule
    try:
        direct, physical = replay.predict(condition["checkpoint_id"], sdf_path)
    except Exception as exc:
        row["warnings"] = f"direct replay failed: {type(exc).__name__}: {exc}"
        row["final_scientific_status"] = "REJECTED_MISSING_EVIDENCE"
        return row, sanitized_molecule
    milp = float(row["milp_prediction"])
    discrepancy = abs(direct - milp)
    lower = float(condition["target_lower_inclusive"])
    upper = float(condition["target_upper_inclusive"])
    member = lower <= physical <= upper
    row.update(
        {
            "direct_gnn_prediction": direct,
            "direct_physical_prediction": physical,
            "direct_milp_discrepancy": discrepancy,
            "direct_target_membership": member,
        }
    )
    warnings = []
    scaled = row["maximum_scaled_row_residual"]
    if scaled is not None and float(scaled) > 1e-7:
        warnings.append(f"scaled_row_residual={scaled}")
    row["warnings"] = ";".join(warnings)
    if discrepancy > PREDICTOR_TOLERANCE:
        row["final_scientific_status"] = "REJECTED_GNN_MISMATCH"
    elif not member:
        row["final_scientific_status"] = "REJECTED_TARGET_MISS"
    else:
        row["final_scientific_status"] = "ACCEPTED_CANDIDATE"
    return row, sanitized_molecule


def prepare_mmff_geometry(molecule: Chem.Mol) -> Chem.Mol:
    base = Chem.Mol(molecule)
    base.RemoveAllConformers()
    prepared = Chem.AddHs(base, addCoords=False)
    params = AllChem.ETKDGv3()
    params.randomSeed = 1
    params.useRandomCoords = True
    params.ignoreSmoothingFailures = True
    # Some accepted graphs contain strained unsaturated rings.  Disabling the
    # empirical basic-knowledge rejection retains ETKDGv3 distance geometry
    # while allowing those graphs to receive a deterministic starting pose.
    params.useBasicKnowledge = False
    if AllChem.EmbedMolecule(prepared, params) != 0:
        raise RuntimeError("ETKDGv3 conformer generation failed")
    if not AllChem.MMFFHasAllMoleculeParams(prepared):
        raise RuntimeError("MMFF parameters unavailable")
    result = AllChem.MMFFOptimizeMolecule(prepared, maxIters=1000)
    if result < 0:
        raise RuntimeError("MMFF optimization failed")
    return prepared


def write_sdf(path: Path, records: list[tuple[Chem.Mol, dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    writer = Chem.SDWriter(str(temporary))
    try:
        for molecule, properties in records:
            value = Chem.Mol(molecule)
            for key, item in properties.items():
                value.SetProp(key, str(item))
            writer.write(value)
    finally:
        writer.close()
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=REPOSITORY / "results/final"
    )
    arguments = parser.parse_args()
    campaign_root = arguments.campaign_root.resolve(strict=True)
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = [
        json.loads(line)
        for line in CONDITIONS.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line)["instance_id"] == "I5"
    ]
    replay = DirectReplay()
    reconstructor = MilpGraphReconstructor()
    rows: list[dict[str, Any]] = []
    accepted: list[tuple[dict[str, Any], Chem.Mol]] = []
    with tempfile.TemporaryDirectory(prefix="phase2-final-replay-") as work:
        prior = Path.cwd()
        os.chdir(work)
        try:
            for index, condition in enumerate(conditions, start=1):
                attempt = completed_attempt(campaign_root, condition["condition_id"])
                if attempt is None:
                    raise RuntimeError(
                        f"missing completed attempt: {condition['condition_id']}"
                    )
                status = read_json(attempt / "status.json")
                sdf_path = attempt / "candidate/candidate.sdf"
                if sdf_path.is_file() or status.get("incumbent_available"):
                    row, molecule = classify_incumbent(
                        condition,
                        attempt,
                        replay,
                        reconstructor,
                        Path(work),
                    )
                    if row["final_scientific_status"] == "ACCEPTED_CANDIDATE" and molecule is not None:
                        accepted.append((row, molecule))
                else:
                    raw_path = attempt / "solver/raw-termination.json"
                    raw = read_json(raw_path) if raw_path.is_file() else {}
                    row = {
                        "condition_id": condition["condition_id"],
                        "specification": "I5" if condition["population"] == "baseline" else "I5-prime",
                        "population": condition["population"],
                        "property": condition["target_property"],
                        "architecture": f"MP{condition['model_depth']}-{condition['hidden_width']}",
                        "architecture_id": condition["architecture_id"],
                        "checkpoint_id": condition["checkpoint_id"],
                        "target_range_index": condition["range_index"],
                        "target_lower": condition["target_lower_inclusive"],
                        "target_upper": condition["target_upper_inclusive"],
                        "original_solver_status": raw.get("native_status"),
                        "original_termination_reason": raw.get("termination_reason"),
                        "original_operational_status": status.get("normalized_status"),
                        "solver_runtime": status.get("solver_runtime_seconds"),
                        "incumbent_available": False,
                        "direct_gnn_prediction": None,
                        "direct_physical_prediction": None,
                        "milp_prediction": None,
                        "direct_milp_discrepancy": None,
                        "direct_target_membership": False,
                        "rdkit_readable": False,
                        "connected": False,
                        "sanitization_valid": False,
                        "prior_graph_adjacency_exact_match": None,
                        "graph_check": False,
                        "raw_indexed_graph_match": False,
                        "raw_graph_isomorphic": False,
                        "graph_mismatch_cause": "",
                        "raw_graph_sha256": "",
                        "intended_graph_sha256": "",
                        "maximum_integrality_residual": None,
                        "maximum_scaled_row_residual": None,
                        "final_scientific_status": "NO_INCUMBENT",
                        "warnings": "",
                        "sdf_path": "",
                        "attempt_path": str(attempt.resolve()),
                        "graph_id": "",
                    }
                rows.append(row)
                if index % 20 == 0 or index == len(conditions):
                    print(f"[RECHECK] {index}/{len(conditions)} conditions", flush=True)
        finally:
            os.chdir(prior)

    graph_members: dict[str, list[tuple[dict[str, Any], Chem.Mol, str]]] = defaultdict(list)
    for row, molecule in accepted:
        smiles, digest = canonical_graph_identity(molecule)
        graph_members[digest].append((row, molecule, smiles))
    graph_ids = {
        digest: f"graph-{index:04d}-{digest[:12]}"
        for index, digest in enumerate(sorted(graph_members), start=1)
    }
    for digest, members in graph_members.items():
        for row, _molecule, _smiles in members:
            row["graph_id"] = graph_ids[digest]

    fields = [
        "condition_id", "specification", "population", "property", "architecture",
        "architecture_id", "checkpoint_id", "target_range_index", "target_lower",
        "target_upper", "original_solver_status", "original_termination_reason",
        "original_operational_status", "solver_runtime", "incumbent_available",
        "direct_gnn_prediction", "direct_physical_prediction", "milp_prediction",
        "direct_milp_discrepancy", "direct_target_membership", "rdkit_readable",
        "connected", "sanitization_valid", "prior_graph_adjacency_exact_match",
        "raw_indexed_graph_match", "raw_graph_isomorphic",
        "graph_mismatch_cause", "raw_graph_sha256", "intended_graph_sha256",
        "graph_check", "maximum_integrality_residual",
        "maximum_scaled_row_residual", "final_scientific_status", "warnings",
        "graph_id", "sdf_path", "attempt_path",
    ]
    write_csv(output_root / "i5_i5prime_condition_summary.csv", rows, fields)
    incumbent_rows = [row for row in rows if row["incumbent_available"]]
    write_csv(output_root / "i5_i5prime_candidate_recheck.csv", incumbent_rows, fields)
    prior_graph_mismatches = [
        row
        for row in incumbent_rows
        if row["prior_graph_adjacency_exact_match"] is False
    ]
    write_csv(
        output_root / "i5_i5prime_graph_mismatch_diagnosis.csv",
        prior_graph_mismatches,
        fields,
    )
    rejected = [
        row for row in incumbent_rows if row["final_scientific_status"] != "ACCEPTED_CANDIDATE"
    ]
    write_csv(output_root / "i5_i5prime_rejected_candidates.csv", rejected, fields)
    write_sdf(
        output_root / "i5_i5prime_accepted_candidates.sdf",
        [
            (
                molecule,
                {
                    "condition_id": row["condition_id"],
                    "specification": row["specification"],
                    "graph_id": row["graph_id"],
                    "direct_physical_prediction": row["direct_physical_prediction"],
                },
            )
            for row, molecule in accepted
        ],
    )

    index_rows = []
    unique_records = []
    manifest_entries = []
    input_root = output_root / "dft_inputs"
    for digest in sorted(graph_members):
        members = graph_members[digest]
        graph_id = graph_ids[digest]
        representative_row, representative_molecule, smiles = members[0]
        source_ids = sorted(row["condition_id"] for row, _mol, _smi in members)
        specifications = sorted({row["specification"] for row, _mol, _smi in members})
        graph_root = input_root / graph_id
        graph_root.mkdir(parents=True, exist_ok=True)
        input_sdf = graph_root / "input_graph.sdf"
        mmff_sdf = graph_root / "mmff_start.sdf"
        write_sdf(input_sdf, [(representative_molecule, {"graph_id": graph_id})])
        mmff = prepare_mmff_geometry(representative_molecule)
        write_sdf(mmff_sdf, [(mmff, {"graph_id": graph_id, "random_seed": 1})])
        index_rows.append(
            {
                "graph_id": graph_id,
                "canonical_graph_sha256": digest,
                "canonical_explicit_smiles": smiles,
                "source_condition_count": len(source_ids),
                "source_condition_ids": ";".join(source_ids),
                "specification_membership": ";".join(specifications),
                "i5_condition_count": sum(row["specification"] == "I5" for row, _mol, _smi in members),
                "i5prime_condition_count": sum(row["specification"] == "I5-prime" for row, _mol, _smi in members),
                "input_sdf": str(input_sdf.resolve()),
                "input_sdf_sha256": sha256_file(input_sdf),
                "mmff_start_sdf": str(mmff_sdf.resolve()),
                "mmff_start_sdf_sha256": sha256_file(mmff_sdf),
            }
        )
        unique_records.append((representative_molecule, {"graph_id": graph_id, "source_condition_ids": ";".join(source_ids), "specification_membership": ";".join(specifications)}))
        for engine in ("psi4", "pyscf-geometric"):
            manifest_entries.append(
                {
                    "dft_id": f"{graph_id}-{engine}",
                    "graph_id": graph_id,
                    "engine": engine,
                    "canonical_graph_sha256": digest,
                    "source_condition_ids": source_ids,
                    "specification_membership": specifications,
                    "input_graph_sdf": str(input_sdf.resolve()),
                    "input_graph_sdf_sha256": sha256_file(input_sdf),
                    "mmff_start_sdf": str(mmff_sdf.resolve()),
                    "mmff_start_sdf_sha256": sha256_file(mmff_sdf),
                }
            )
    index_fields = list(index_rows[0]) if index_rows else ["graph_id"]
    write_csv(output_root / "dft_candidate_index.csv", index_rows, index_fields)
    write_sdf(output_root / "dft_unique_graphs.sdf", unique_records)
    protocol = {
        "conformer": "RDKit ETKDGv3",
        "random_seed": 1,
        "etkdg_use_random_coordinates": True,
        "etkdg_use_basic_knowledge": False,
        "hydrogens": "explicit-retained",
        "force_field": "MMFF",
        "preoptimization": {"method": "B3LYP", "basis": "3-21G"},
        "final_optimization": {"method": "B3LYP", "basis": "6-31G(2df,p)"},
        "frequency": {"method": "B3LYP", "basis": "6-31G(2df,p)", "imaginary_threshold_cm-1": -20.0},
        "properties": ["HOMO", "LUMO", "Gap", "ZPVE_eV"],
        "engines": {
            "psi4": {
                "environment": "psi4-zpve",
                "psi4_version": "1.10.1",
            },
            "pyscf-geometric": {
                "environment": "pyscf-zpve",
                "pyscf_version": "2.13.1",
                "geometric_version": "1.1.1",
            },
        },
    }
    manifest = {
        "schema_version": "phase2-production-dft-manifest/1",
        "manifest_id": "phase2-i5-i5prime-dft",
        "campaign_id": CAMPAIGN_ID,
        "source_condition_count": len(conditions),
        "accepted_condition_count": len(accepted),
        "unique_graph_count": len(graph_members),
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        "candidate_index_sha256": sha256_file(output_root / "dft_candidate_index.csv"),
        "entries": manifest_entries,
    }
    atomic_write_json(output_root / "dft_manifest.json", manifest, overwrite=True)
    final_counts = Counter(row["final_scientific_status"] for row in incumbent_rows)
    original_counts = Counter(row["original_operational_status"] for row in rows)
    accepted_specs = Counter(row["specification"] for row, _mol in accepted)
    unique_specs = Counter()
    shared = 0
    for members in graph_members.values():
        specifications = {row["specification"] for row, _mol, _smi in members}
        for specification in specifications:
            unique_specs[specification] += 1
        if specifications == {"I5", "I5-prime"}:
            shared += 1
    recovered = sum(
        row["original_operational_status"] == "POSTSOLVE_REJECTED"
        and row["final_scientific_status"] == "ACCEPTED_CANDIDATE"
        for row in incumbent_rows
    )
    print(f"Total I5 conditions: {sum(row['specification'] == 'I5' for row in rows)}")
    print(f"Total I5-prime conditions: {sum(row['specification'] == 'I5-prime' for row in rows)}")
    print(f"Solver incumbents: {len(incumbent_rows)}")
    print(f"Original VERIFIED_CANDIDATE: {original_counts['VERIFIED_CANDIDATE']}")
    print(f"Original POSTSOLVE_REJECTED: {original_counts['POSTSOLVE_REJECTED']}")
    print(f"Final ACCEPTED_CANDIDATE: {final_counts['ACCEPTED_CANDIDATE']}")
    for status in FINAL_STATUSES[1:]:
        print(f"{status}: {final_counts[status]}")
    print(f"Accepted I5 conditions: {accepted_specs['I5']}")
    print(f"Accepted I5-prime conditions: {accepted_specs['I5-prime']}")
    print(f"Unique I5 graphs: {unique_specs['I5']}")
    print(f"Unique I5-prime graphs: {unique_specs['I5-prime']}")
    print(f"Shared graphs: {shared}")
    print(f"Recovered POSTSOLVE_REJECTED: {recovered}")
    print(f"DFT manifest: {(output_root / 'dft_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
