"""Restart-safe, immutable-attempt Phase 2 condition execution."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from ..attempts import (
    AttemptState,
    claim_attempt,
    classify_attempt,
    complete_attempt,
)
from ..canonical import atomic_write_bytes, atomic_write_json, sha256_file
from ..schema import validate_named
from ..status import normalize_termination
from .formulation import build_common_formulation
from .solver import (
    canonical_cplex_solver,
    detect_termination,
    effective_parameter_record,
)
from .verification import verify_incumbent

_REQUIRED = [
    "input/condition.json",
    "formulation/model.lp",
    "solver/cplex.commands",
    "solver/cplex.log",
    "solver/stdout.txt",
    "solver/stderr.txt",
    "solver/raw-termination.json",
    "solver/effective-parameters.json",
    "status.json",
]


def _artifact(path: Path, attempt: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    media_type = {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".lp": "application/lp",
        ".sol": "application/xml",
        ".sdf": "chemical/x-mdl-sdfile",
        ".txt": "text/plain",
        ".log": "text/plain",
        ".commands": "text/plain",
    }.get(suffix, "application/octet-stream")
    return {
        "path": path.relative_to(attempt).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "media_type": media_type,
        "required": True,
    }


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _environment_record(executable: Path) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torch-geometric", "pulp", "rdkit", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpuinfo = Path("/proc/cpuinfo")
    cpu_records: list[dict[str, str]] = []
    if cpuinfo.is_file():
        for block in cpuinfo.read_text(encoding="utf-8").split("\n\n"):
            record = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    record[key.strip()] = value.strip()
            if record:
                cpu_records.append(record)
    physical_cores = {
        (record.get("physical id"), record.get("core id"))
        for record in cpu_records
        if record.get("physical id") is not None
        and record.get("core id") is not None
    }
    model_names = sorted(
        {
            record["model name"]
            for record in cpu_records
            if record.get("model name")
        }
    )
    affinity_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "cplex_executable": str(executable),
        "cplex_executable_sha256": sha256_file(executable),
        "host_cpu": {
            "logical_cpu_count_os": os.cpu_count(),
            "logical_cpu_count_proc": len(cpu_records) or None,
            "affinity_cpu_count": affinity_count,
            "physical_core_count": len(physical_cores) or None,
            "model_names": model_names,
            "machine": platform.machine(),
        },
        "requested_solver_threads": 0,
        "solver_threads_policy": "automatic",
        "deterministic_parallel_mode": 1,
    }


def _input_hash_record(
    repository_root: Path, condition: dict[str, Any]
) -> dict[str, Any]:
    inventory = json.loads(
        (repository_root / "data/phase1/asset-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    configs = json.loads(
        (repository_root / "data/phase1/config-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = next(
        record
        for record in inventory["checkpoint_records"]
        if record["logical_asset_id"] == condition["checkpoint_id"]
    )
    config = next(
        record
        for record in configs
        if record["logical_asset_id"] == condition["checkpoint_id"]
    )
    instance_subdirectory = (
        "diagnostic" if condition.get("diagnostic_instance") else "canonical"
    )
    instance_root = (
        repository_root / "data/phase2/instances" / instance_subdirectory
    )
    instance_id = condition["instance_id"]
    files = {
        "checkpoint": repository_root
        / "data/phase1"
        / checkpoint["destination_path"],
        "normalization_config": repository_root
        / "data/phase1"
        / config["destination_path"],
        "structural_instance": instance_root / f"{instance_id}.txt",
        "fringe_library": instance_root / f"{instance_id}-fringe.txt",
    }
    target_range_hash = condition.get("target_range_manifest_sha256")
    if target_range_hash is not None:
        target_range_manifest = (
            repository_root
            / "data/phase2/target_ranges/target-range-manifest.json"
        )
        if sha256_file(target_range_manifest) != target_range_hash:
            raise ValueError("condition target-range manifest hash mismatch")
        files["target_range_manifest"] = target_range_manifest
    return {
        "schema_version": "phase2-attempt-input-hashes/1",
        "assets": {
            name: {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in files.items()
        },
        "split_id": inventory["split_id"],
        "normalization": checkpoint["normalization"],
    }


def _select_attempt(
    *,
    run_root: Path,
    campaign_id: str,
    condition_id: str,
    retry_after_attempt: str | None = None,
    retry_reason: str | None = None,
) -> tuple[Path | None, int, str | None, str | None]:
    condition_root = run_root / campaign_id / condition_id
    prior = sorted(condition_root.glob("attempt-[0-9][0-9][0-9][0-9]"))
    completed = [
        path
        for path in prior
        if classify_attempt(path) is AttemptState.COMPLETED
    ]
    if completed:
        latest = completed[-1]
        if retry_after_attempt != latest.name:
            return latest, 0, None, None
        return (
            None,
            len(prior) + 1,
            latest.name,
            retry_reason or "explicit invalidation retry",
        )
    number = len(prior) + 1
    if prior:
        return (
            None,
            number,
            prior[-1].name,
            (
                "restart after preserved "
                f"{classify_attempt(prior[-1]).value.lower()} attempt"
            ),
        )
    return None, number, None, None


def run_condition(
    *,
    repository_root: Path,
    run_root: Path,
    campaign_id: str,
    condition: dict[str, Any],
    executable: Path,
    time_limit_seconds: float = 3600,
    resume: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Run once, or return the immutable completed result on resume."""
    if condition.get("schema_version") == "phase2-production-condition/1":
        validate_named(condition, "phase2-production-condition")
    repository_root = repository_root.resolve(strict=True)
    executable = executable.resolve(strict=True)
    condition_id = condition["condition_id"]
    if resume:
        completed, number, retry_of, retry_reason = _select_attempt(
            run_root=run_root,
            campaign_id=campaign_id,
            condition_id=condition_id,
            retry_after_attempt=condition.get("retry_after_attempt"),
            retry_reason=condition.get("retry_reason"),
        )
        if completed is not None:
            return completed, json.loads(
                (completed / "status.json").read_text(encoding="utf-8")
            )
    else:
        number, retry_of, retry_reason = 1, None, None
    attempt = claim_attempt(
        run_root,
        campaign_id,
        condition_id,
        number,
        required_artifacts=_REQUIRED,
        retry_of=retry_of,
        retry_reason=retry_reason,
    )
    atomic_write_json(attempt / "input/condition.json", condition)
    atomic_write_json(
        attempt / "input/input-hashes.json",
        _input_hash_record(repository_root, condition),
    )
    atomic_write_json(
        attempt / "provenance/environment.json",
        _environment_record(executable),
    )
    parameters = effective_parameter_record(time_limit_seconds)
    atomic_write_json(attempt / "solver/effective-parameters.json", parameters)

    build_started = time.monotonic()
    formulation = build_common_formulation(
        repository_root=repository_root,
        condition=condition,
    )
    formulation_seconds = time.monotonic() - build_started
    solver = canonical_cplex_solver(
        executable=executable,
        attempt=attempt,
        time_limit_seconds=time_limit_seconds,
    )
    formulation.problem.solve(solver)
    atomic_write_bytes(attempt / "solver/stdout.txt", solver.captured_stdout)
    atomic_write_bytes(attempt / "solver/stderr.txt", solver.captured_stderr)
    atomic_write_bytes(attempt / "solver/cplex.commands", solver.command_bytes)
    log_path = attempt / "solver/cplex.log"
    if not log_path.exists():
        atomic_write_bytes(log_path, b"")
    combined = (
        solver.captured_stdout.decode("utf-8", errors="replace")
        + "\n"
        + solver.captured_stderr.decode("utf-8", errors="replace")
        + "\n"
        + log_path.read_text(encoding="utf-8", errors="replace")
    )
    termination = detect_termination(
        combined, process_exit_code=solver.process_exit_code
    )
    header = solver.solution_header or {}
    has_incumbent = solver.solution_header is not None and any(
        variable.value() is not None for variable in formulation.problem.variables()
    )
    raw = {
        "schema_version": "raw-solver-termination/1",
        "engine": "CPLEX",
        "engine_version": "22.1.1",
        "native_status": header.get(
            "solutionStatusString", termination.value
        ),
        "native_code": header.get("solutionStatusValue"),
        "termination_reason": termination.value,
        "has_incumbent": has_incumbent,
        "incumbent_count": 1 if has_incumbent else 0,
        "best_bound": _optional_float(header.get("bestObjective")),
        "relative_gap": _optional_float(header.get("MIPRelativeGap")),
        "wall_seconds": float(solver.solver_runtime_seconds or 0.0),
        "cpu_seconds": 0.0,
        "exception": (
            None
            if solver.parse_failure is None
            else {
                "class": "SolutionParseError",
                "message": solver.parse_failure,
                "traceback_sha256": None,
            }
        ),
    }
    validate_named(raw, "raw-solver-termination")
    raw_path = attempt / "solver/raw-termination.json"
    atomic_write_json(raw_path, raw)

    predictor = None
    verification = None
    if has_incumbent:
        variable_dump = b"".join(
            json.dumps(
                {
                    "name": variable.name,
                    "value": repr(float(variable.value())),
                    "ieee754_hex": float(variable.value()).hex(),
                    "category": variable.cat,
                    "lower_bound": variable.lowBound,
                    "upper_bound": variable.upBound,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            for variable in formulation.problem.variables()
        )
        atomic_write_bytes(
            attempt / "solver/full-precision-variables.jsonl", variable_dump
        )
        verification, predictor = verify_incumbent(
            formulation=formulation,
            attempt=attempt,
            raw_termination_sha256=sha256_file(raw_path),
        )
    normalized = normalize_termination(
        termination,
        has_incumbent=has_incumbent,
        verification_passed=(
            verification["all_mandatory_checks_passed"]
            if verification is not None
            else None
        ),
    )
    status = {
        "schema_version": "phase2-attempt-status/1",
        "campaign_id": campaign_id,
        "condition_id": condition_id,
        "attempt_id": attempt.name,
        "excluded_from_final_statistics": bool(
            condition.get("excluded_from_final_statistics", False)
        ),
        "normalized_status": normalized.value,
        "raw_termination_reason": termination.value,
        "process_exit_code": solver.process_exit_code,
        "incumbent_available": has_incumbent,
        "pulp_status": __import__("pulp").LpStatus.get(
            formulation.problem.status, str(formulation.problem.status)
        ),
        "solver_runtime_seconds": solver.solver_runtime_seconds,
        "formulation_runtime_seconds": formulation_seconds,
        "verification_passed": (
            verification["all_mandatory_checks_passed"]
            if verification is not None
            else None
        ),
        "direct_standardized_prediction": (
            predictor["direct_standardized_prediction"]["value"]
            if predictor is not None
            else None
        ),
        "milp_standardized_prediction": (
            predictor["milp_standardized_prediction"]["value"]
            if predictor is not None
            else None
        ),
        "direct_milp_absolute_discrepancy": (
            predictor["absolute_standardized_discrepancy"]["value"]
            if predictor is not None
            else None
        ),
    }
    atomic_write_json(attempt / "status.json", status)
    artifacts = [
        _artifact(path, attempt)
        for path in sorted(attempt.rglob("*"))
        if path.is_file()
        and path.name
        not in {"ATTEMPT.json", "artifact-manifest.json", "COMPLETED.json"}
    ]
    manifest = {
        "schema_version": "artifact-manifest/1",
        "manifest_id": f"{condition_id}-{attempt.name}-artifacts",
        "root_kind": "attempt",
        "artifacts": artifacts,
    }
    complete_attempt(attempt, manifest)
    return attempt, status


def run_predeclared_smoke(
    *,
    repository_root: Path,
    smoke_config_path: Path,
    run_root: Path,
    executable: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    value = json.loads(smoke_config_path.read_text(encoding="utf-8"))
    if not value.get("predeclared") or value.get("production_campaign"):
        raise ValueError("smoke configuration must be predeclared and diagnostic")
    return [
        run_condition(
            repository_root=repository_root,
            run_root=run_root,
            campaign_id=value["campaign_id"],
            condition=condition,
            executable=executable,
            time_limit_seconds=condition.get(
                "diagnostic_time_limit_seconds",
                value["time_limit_seconds"],
            ),
        )
        for condition in value["conditions"]
    ]
