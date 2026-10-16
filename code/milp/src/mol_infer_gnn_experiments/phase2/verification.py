"""Full-precision numerical, structural, and predictor verification."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rdkit import Chem

from ..canonical import atomic_write_json, sha256_file
from ..errors import ContractError
from ..schema import validate_named
from ..status import NormalizedStatus
from .formulation import Formulation

INTEGRALITY_TOLERANCE = 1e-9
VARIABLE_BOUND_TOLERANCE = 1e-9
SCALED_ROW_TOLERANCE = 1e-7
PREDICTOR_TOLERANCE = 1e-5


def _float(value: float) -> dict[str, Any]:
    return {"decimal": repr(value), "hex": value.hex(), "value": value}


def audit_variables(problem: Any) -> dict[str, Any]:
    missing = []
    nonfinite = []
    maximum_integrality = 0.0
    maximum_bound = 0.0
    integral_count = 0
    for variable in problem.variables():
        value = variable.value()
        if value is None:
            missing.append(variable.name)
            continue
        value = float(value)
        if not math.isfinite(value):
            nonfinite.append(variable.name)
            continue
        if variable.cat == "Integer":
            integral_count += 1
            maximum_integrality = max(
                maximum_integrality, abs(value - round(value))
            )
        if variable.lowBound is not None:
            maximum_bound = max(
                maximum_bound, max(0.0, float(variable.lowBound) - value)
            )
        if variable.upBound is not None:
            maximum_bound = max(
                maximum_bound, max(0.0, value - float(variable.upBound))
            )
    return {
        "variable_count": len(problem.variables()),
        "integer_variable_count": integral_count,
        "missing_variables": missing,
        "nonfinite_variables": nonfinite,
        "maximum_integrality_residual": _float(maximum_integrality),
        "integrality_tolerance": INTEGRALITY_TOLERANCE,
        "maximum_variable_bound_violation": _float(maximum_bound),
        "variable_bound_tolerance": VARIABLE_BOUND_TOLERANCE,
        "required_variables_passed": not missing and not nonfinite,
        "integrality_passed": maximum_integrality <= INTEGRALITY_TOLERANCE,
        "variable_bounds_passed": maximum_bound <= VARIABLE_BOUND_TOLERANCE,
    }


def audit_linear_rows(problem: Any) -> dict[str, Any]:
    maximum_native = 0.0
    maximum_scaled = 0.0
    worst_row = None
    for name, constraint in problem.constraints.items():
        native_value = float(constraint.value())
        if constraint.sense == -1:
            violation = max(0.0, native_value)
        elif constraint.sense == 1:
            violation = max(0.0, -native_value)
        else:
            violation = abs(native_value)
        activity_magnitude = sum(
            abs(float(coefficient) * float(variable.value()))
            for variable, coefficient in constraint.items()
        )
        scale = max(1.0, abs(float(constraint.constant)), activity_magnitude)
        scaled = violation / scale
        if scaled > maximum_scaled:
            maximum_scaled = scaled
            worst_row = name
        maximum_native = max(maximum_native, violation)
    return {
        "row_count": len(problem.constraints),
        "maximum_native_violation": _float(maximum_native),
        "maximum_scaled_violation": _float(maximum_scaled),
        "scaled_row_tolerance": SCALED_ROW_TOLERANCE,
        "worst_row": worst_row,
        "passed": maximum_scaled <= SCALED_ROW_TOLERANCE,
    }


def _graph_evidence(seed_graph: Any, sdf_path: Path) -> dict[str, Any]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    molecule = supplier[0] if len(supplier) else None
    if molecule is None:
        return {"passed": False, "reason": "RDKit rejected reconstructed SDF"}
    atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    bonds = sorted(
        (
            min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) + 1,
            max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) + 1,
            int(round(bond.GetBondTypeAsDouble())),
        )
        for bond in molecule.GetBonds()
    )
    legacy_bonds = sorted(
        (first, second, int(seed_graph.graph_adj[(first, second)]))
        for first in range(1, seed_graph.graph_ind + 1)
        for second in range(first + 1, seed_graph.graph_ind + 1)
        if seed_graph.graph_adj[(first, second)] > 0
    )
    graph_value = {"atoms": atoms, "bonds": bonds}
    graph_bytes = json.dumps(
        graph_value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    import hashlib

    expected_heavy_atoms = int(round(seed_graph.n_G.value()))
    passed = (
        molecule.GetNumHeavyAtoms() == expected_heavy_atoms
        and molecule.GetNumAtoms() == seed_graph.graph_ind
        and bonds == legacy_bonds
        and len(Chem.GetMolFrags(molecule)) == 1
    )
    return {
        "total_atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "expected_heavy_atom_count_from_n_G": expected_heavy_atoms,
        "bond_count": molecule.GetNumBonds(),
        "legacy_adjacency_bond_count": len(legacy_bonds),
        "connected_component_count": len(Chem.GetMolFrags(molecule)),
        "adjacency_exact_match": bonds == legacy_bonds,
        "canonical_graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
        "sdf_sha256": sha256_file(sdf_path),
        "passed": passed,
    }


def verify_incumbent(
    *,
    formulation: Formulation,
    attempt: Path,
    raw_termination_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and retain every acceptance-critical value and artifact hash."""
    evidence_root = attempt / "verification"
    candidate_root = attempt / "candidate"
    candidate_root.mkdir(exist_ok=True)
    variable_evidence = audit_variables(formulation.problem)
    row_evidence = audit_linear_rows(formulation.problem)
    variable_path = evidence_root / "variable-audit.json"
    row_path = evidence_root / "linear-row-audit.json"
    atomic_write_json(variable_path, variable_evidence)
    atomic_write_json(row_path, row_evidence)

    sdf_prefix = candidate_root / "candidate"
    formulation.seed_graph._print_sdf(str(sdf_prefix))
    sdf_path = candidate_root / "candidate.sdf"
    graph_evidence = _graph_evidence(formulation.seed_graph, sdf_path)
    graph_path = evidence_root / "structural-reconstruction.json"
    atomic_write_json(graph_path, graph_evidence)

    milp_standardized = float(formulation.assignment.output.value())
    direct_standardized = float(
        formulation.inverter.inspection(str(sdf_path))
    )
    discrepancy = abs(milp_standardized - direct_standardized)
    direct_target = (
        direct_standardized * formulation.inverter.y_std
        + formulation.inverter.y_mean
    )
    lower = float(formulation.condition["target_lower_inclusive"])
    upper = float(formulation.condition["target_upper_inclusive"])
    predictor_evidence = {
        "milp_standardized_prediction": _float(milp_standardized),
        "direct_standardized_prediction": _float(direct_standardized),
        "absolute_standardized_discrepancy": _float(discrepancy),
        "predictor_tolerance": PREDICTOR_TOLERANCE,
        "direct_target_prediction": _float(direct_target),
        "target_lower_inclusive": _float(lower),
        "target_upper_inclusive": _float(upper),
        "predictor_equivalence_passed": discrepancy <= PREDICTOR_TOLERANCE,
        "target_membership_passed": lower <= direct_target <= upper,
    }
    predictor_path = evidence_root / "predictor-equivalence.json"
    atomic_write_json(predictor_path, predictor_evidence)

    provenance_evidence = {
        "condition_id": formulation.condition["condition_id"],
        "checkpoint_id": formulation.condition["checkpoint_id"],
        "checkpoint_sha256": formulation.condition["checkpoint_sha256"],
        "sdf_sha256": sha256_file(sdf_path),
        "raw_termination_sha256": raw_termination_sha256,
    }
    provenance_path = evidence_root / "verification-provenance.json"
    atomic_write_json(provenance_path, provenance_evidence)
    check_values = {
        "required_variables": variable_evidence["required_variables_passed"],
        "integrality": variable_evidence["integrality_passed"],
        "variable_bounds": variable_evidence["variable_bounds_passed"],
        "linear_constraints": row_evidence["passed"],
        "structural_reconstruction": graph_evidence["passed"],
        "graph_identity": graph_evidence["adjacency_exact_match"],
        "predictor_equivalence": predictor_evidence[
            "predictor_equivalence_passed"
        ],
        "target_membership": predictor_evidence["target_membership_passed"],
        "provenance": True,
    }
    evidence_paths = {
        "required_variables": variable_path,
        "integrality": variable_path,
        "variable_bounds": variable_path,
        "linear_constraints": row_path,
        "structural_reconstruction": graph_path,
        "graph_identity": graph_path,
        "predictor_equivalence": predictor_path,
        "target_membership": predictor_path,
        "provenance": provenance_path,
    }
    passed = all(check_values.values())
    record = {
        "schema_version": "verification-record/1",
        "condition_id": formulation.condition["condition_id"],
        "attempt_id": attempt.name,
        "raw_termination_sha256": raw_termination_sha256,
        "has_incumbent": True,
        "checks": [
            {
                "check_id": check_id,
                "mandatory": True,
                "passed": check_values[check_id],
                "evidence_sha256": sha256_file(evidence_paths[check_id]),
            }
            for check_id in check_values
        ],
        "all_mandatory_checks_passed": passed,
        "normalized_status": (
            NormalizedStatus.VERIFIED_CANDIDATE.value
            if passed
            else NormalizedStatus.POSTSOLVE_REJECTED.value
        ),
        "candidate_artifact_sha256": sha256_file(sdf_path),
    }
    validate_named(record, "verification-record")
    atomic_write_json(evidence_root / "verification-record.json", record)
    return record, predictor_evidence
