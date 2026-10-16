"""The single production structural and GNN-MILP formulation path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ContractError
from .specification import (
    BASELINE_I5,
    GUIDED_I5_PRIME,
    StructuralSpecification,
    apply_design_c_constraints,
)


@dataclass
class Formulation:
    problem: Any
    seed_graph: Any
    inverter: Any
    assignment: Any
    condition: dict[str, Any]
    specification: StructuralSpecification


def specification_for(condition: dict[str, Any]) -> StructuralSpecification:
    population = condition["population"]
    if population == "baseline":
        return BASELINE_I5
    if population == "guided":
        return GUIDED_I5_PRIME
    raise ContractError(f"unknown Phase 2 population: {population}")


def _asset_records(asset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = json.loads(
        (asset_root / "asset-inventory.json").read_text(encoding="utf-8")
    )
    configs = json.loads(
        (asset_root / "config-inventory.json").read_text(encoding="utf-8")
    )
    return (
        {
            record["logical_asset_id"]: record
            for record in inventory["checkpoint_records"]
        },
        {record["logical_asset_id"]: record for record in configs},
    )


def build_common_formulation(
    *,
    repository_root: Path,
    condition: dict[str, Any],
) -> Formulation:
    """Build baseline or guided through the same implementation and inputs."""
    import pulp
    from src.config import AssignmentLoader, GNNModelLoader
    from src_inverter.inverter import create_inverter
    from src_milp.assignment import check_assignment, create_assignment
    from src_milp.seed_graph import create_seed_graph

    root = repository_root.resolve(strict=True)
    asset_root = root / "data/phase1"
    checkpoint_records, config_records = _asset_records(asset_root)
    checkpoint_id = condition["checkpoint_id"]
    checkpoint = checkpoint_records[checkpoint_id]
    config = config_records[checkpoint_id]
    instance_id = condition["instance_id"]
    instance_root = root / "data/phase2/instances"
    instance_subdirectory = (
        "diagnostic" if condition.get("diagnostic_instance") else "canonical"
    )
    instance_root = instance_root / instance_subdirectory

    seed_graph = create_seed_graph("2LMM_GNN", instance_id)
    seed_graph.read_seed_graph(instance_root / f"{instance_id}.txt")
    seed_graph.read_fringe_trees(instance_root / f"{instance_id}-fringe.txt")
    seed_graph.set_up_seed_graph()
    seed_graph.set_up_variables()
    problem = pulp.LpProblem(name=condition["condition_id"])
    problem = seed_graph.add_constraints_C2(problem)

    specification = specification_for(condition)
    canonical_interior = {
        pattern: seed_graph.ac_int[pattern]
        for pattern in (
            seed_graph.Gamma_int_ac_less
            + seed_graph.Gamma_int_ac_equal
        )
    }
    problem = apply_design_c_constraints(
        problem,
        specification=specification,
        interior_totals=canonical_interior,
        leaf_oriented=seed_graph.ac_lf,
    )

    model_loader = GNNModelLoader.from_dict(
        {
            "name": checkpoint_id,
            "model_filename": str(asset_root / checkpoint["destination_path"]),
            "config_filename": str(asset_root / config["destination_path"]),
        }
    )
    inverter = create_inverter(model_loader.type, checkpoint_id)
    inverter.set_up_inverter(model_loader)
    assignment_loader = AssignmentLoader.from_dict(
        {
            "type": "GNN",
            "seed_graph": instance_id,
            "learning_model": checkpoint_id,
            # Canonical manifests preserve exact decimal strings.  Conversion
            # occurs only at the legacy numerical formulation boundary.
            "lb": float(condition["target_lower_inclusive"]),
            "ub": float(condition["target_upper_inclusive"]),
        }
    )
    assignment = create_assignment(
        sg=seed_graph,
        inv=inverter,
        asgml=assignment_loader,
    )
    check_assignment(assignment)
    problem = assignment.connect_C1_C2(problem)
    return Formulation(
        problem=problem,
        seed_graph=seed_graph,
        inverter=inverter,
        assignment=assignment,
        condition=condition,
        specification=specification,
    )
