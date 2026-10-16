#!/usr/bin/env python3
"""Prepare and sequentially run the canonical 960-condition Phase 2 campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / "src")]

from mol_infer_gnn_experiments.attempts import (  # noqa: E402
    AttemptState,
    classify_attempt,
)
from mol_infer_gnn_experiments.canonical import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from mol_infer_gnn_experiments.phase2.runner import run_condition  # noqa: E402
from mol_infer_gnn_experiments.phase2.solver import (  # noqa: E402
    effective_parameter_record,
)
from mol_infer_gnn_experiments.phase2.specification import (  # noqa: E402
    BASELINE_I5,
    GUIDED_I5_PRIME,
)
from mol_infer_gnn_experiments.phase2.verification import (  # noqa: E402
    INTEGRALITY_TOLERANCE,
    PREDICTOR_TOLERANCE,
    SCALED_ROW_TOLERANCE,
)

TARGET_MANIFEST = REPOSITORY / "data/phase2/target_ranges/target-range-manifest.json"
CONDITION_MANIFEST = (
    REPOSITORY / "data/phase2/conditions/production-conditions.jsonl"
)
CONDITION_SET_MANIFEST = (
    REPOSITORY / "data/phase2/conditions/production-manifest.json"
)
SOLVER_PROFILE = (
    REPOSITORY / "configs/solver/cplex-22.1.1-deterministic-v1.json"
)
# Publication packaging only: replace the original machine-specific executable path.
CPLEX = Path(os.environ.get("CPLEX_EXECUTABLE", "cplex"))
TARGET_MANIFEST_SHA256 = (
    "39daaaf4897ab0746763d13fb1db099a263cdf34cc353264ba3305e67c52542b"
)
CONDITION_MANIFEST_SHA256 = (
    "b0dfbb9f67cc999f4e92121ab51a1d0bc512f6db44ebffb29715c80b57d12e42"
)
CAMPAIGN_ID = "phase2-v2-960-sequential-thread0-20260727"
TMUX_SESSION = "molgnn-phase2-v2"
VALID_STATUSES = {
    "VERIFIED_CANDIDATE",
    "INFEASIBLE_CERTIFIED",
    "TIMEOUT_NO_CANDIDATE",
    "SOLVER_ERROR",
    "MEMORY_ERROR",
    "POSTSOLVE_REJECTED",
    "INFRASTRUCTURE_ERROR",
}
PAUSE_STATUSES = {
    "SOLVER_ERROR",
    "MEMORY_ERROR",
    "POSTSOLVE_REJECTED",
    "INFRASTRUCTURE_ERROR",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def canonical_specification(specification: Any) -> dict[str, Any]:
    def patterns(values: Any) -> list[list[Any]]:
        return [list(pattern) for pattern in sorted(values)]

    return {
        "specification_id": specification.specification_id,
        "enabled": specification.enabled,
        "allowed_interior": patterns(specification.allowed_interior),
        "interior_upper_bounds": [
            {"pattern": list(pattern), "upper_bound": upper}
            for pattern, upper in sorted(
                specification.interior_upper_bounds.items()
            )
        ],
        "leaf_upper_bounds": [
            {"pattern": list(pattern), "upper_bound": upper}
            for pattern, upper in sorted(
                specification.leaf_upper_bounds.items()
            )
        ],
    }


def specification_record() -> dict[str, Any]:
    result = {}
    for name, specification in (
        ("baseline", BASELINE_I5),
        ("guided", GUIDED_I5_PRIME),
    ):
        value = canonical_specification(specification)
        result[name] = {
            "value": value,
            "sha256": sha256_bytes(canonical_json_bytes(value)),
        }
    return result


def cpu_record() -> dict[str, Any]:
    records = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for block in cpuinfo.read_text(encoding="utf-8").split("\n\n"):
            record = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    record[key.strip()] = value.strip()
            if record:
                records.append(record)
    physical = {
        (record.get("physical id"), record.get("core id"))
        for record in records
        if record.get("physical id") is not None
        and record.get("core id") is not None
    }
    return {
        "logical_cpu_count_os": os.cpu_count(),
        "logical_cpu_count_proc": len(records) or None,
        "affinity_cpu_count": (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "physical_core_count": len(physical) or None,
        "model_names": sorted(
            {
                record["model name"]
                for record in records
                if record.get("model name")
            }
        ),
        "machine": platform.machine(),
    }


def memory_record() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[f"{key.lower()}_bytes"] = int(value.split()[0]) * 1024
    return values


def conditions() -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in CONDITION_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(values) != 960 or len({item["condition_id"] for item in values}) != 960:
        raise RuntimeError("canonical condition count or uniqueness failed")
    if sum(item["population"] == "baseline" for item in values) != 800:
        raise RuntimeError("canonical baseline count failed")
    if sum(item["population"] == "guided" for item in values) != 160:
        raise RuntimeError("canonical guided count failed")
    if any(
        item["target_range_manifest_sha256"] != TARGET_MANIFEST_SHA256
        for item in values
    ):
        raise RuntimeError("condition target-range manifest link failed")
    return values


def prepare(campaign_root: Path, acceptance_attempt: Path) -> None:
    campaign_root = campaign_root.resolve()
    if campaign_root.exists():
        raise RuntimeError(f"campaign root already exists: {campaign_root}")
    if sha256_file(TARGET_MANIFEST) != TARGET_MANIFEST_SHA256:
        raise RuntimeError("canonical target-range manifest hash mismatch")
    if sha256_file(CONDITION_MANIFEST) != CONDITION_MANIFEST_SHA256:
        raise RuntimeError("canonical 960-condition manifest hash mismatch")
    values = conditions()
    profile = read_json(SOLVER_PROFILE)
    if profile["threads"] != 0 or profile["parallel_mode"] != "deterministic":
        raise RuntimeError("sequential automatic-thread solver profile mismatch")
    if not (acceptance_attempt / "COMPLETED.json").is_file():
        raise RuntimeError("short CPLEX acceptance evidence is incomplete")
    toy = read_json(acceptance_attempt / "toy-solve.json")
    if not toy["toy_solve_passed"] or toy["parameters"]["threads"] != 0:
        raise RuntimeError("short CPLEX acceptance failed")

    storage = shutil.disk_usage(REPOSITORY)
    package_versions = {}
    for name in ("torch", "torch-geometric", "pulp", "rdkit", "numpy"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    preflight = {
        "schema_version": "phase2-production-preflight/1",
        "captured_at": now(),
        "hostname": socket.gethostname(),
        "repository_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_worktree_porcelain": git(
            "status", "--porcelain=v1", "--untracked-files=no"
        ),
        "host_cpu": cpu_record(),
        "memory": memory_record(),
        "storage": {
            "total_bytes": storage.total,
            "used_bytes": storage.used,
            "available_bytes": storage.free,
        },
        "packages": package_versions,
        "cplex": {
            "executable": str(CPLEX),
            "executable_sha256": sha256_file(CPLEX),
            "version": toy["cplex_version"],
            "license_status": "usable",
            "acceptance_attempt": acceptance_attempt.relative_to(
                REPOSITORY
            ).as_posix(),
            "acceptance_attempt_sha256": sha256_file(
                acceptance_attempt / "artifact-manifest.json"
            ),
        },
        "checkpoint_inventory_sha256": sha256_file(
            REPOSITORY / "data/phase1/asset-inventory.json"
        ),
        "checkpoint_parity_sha256": sha256_file(
            REPOSITORY / "data/phase1/live-parity-result.json"
        ),
        "specifications": specification_record(),
        "target_range_manifest_sha256": TARGET_MANIFEST_SHA256,
        "condition_manifest_sha256": CONDITION_MANIFEST_SHA256,
        "solver_profile_sha256": sha256_file(SOLVER_PROFILE),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    if preflight["tracked_worktree_porcelain"]:
        raise RuntimeError("production requires a clean tracked worktree")
    preflight_hash = atomic_write_json(
        campaign_root / "production-preflight.json", preflight
    )
    condition_hashes = [
        {
            "condition_id": item["condition_id"],
            "sha256": sha256_bytes(canonical_json_bytes(item)),
        }
        for item in values
    ]
    runner_command = (
        f"cd {REPOSITORY} && PYTHONPATH=.:src {REPOSITORY}/.venv/bin/python "
        f"{REPOSITORY}/scripts/run_phase2_production.py run "
        f"--campaign-root {campaign_root}"
    )
    stdout_path = campaign_root / "campaign.stdout.log"
    stderr_path = campaign_root / "campaign.stderr.log"
    tmux_command = (
        f"tmux new-session -d -s {TMUX_SESSION} "
        f"\"{runner_command} >> {stdout_path} 2>> {stderr_path}\""
    )
    manifest = {
        "schema_version": "phase2-production-execution-manifest/1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": now(),
        "immutable": True,
        "repository_commit": preflight["repository_commit"],
        "target_range_manifest": {
            "path": TARGET_MANIFEST.relative_to(REPOSITORY).as_posix(),
            "sha256": TARGET_MANIFEST_SHA256,
        },
        "condition_manifest": {
            "path": CONDITION_MANIFEST.relative_to(REPOSITORY).as_posix(),
            "sha256": CONDITION_MANIFEST_SHA256,
            "condition_count": 960,
            "baseline_count": 800,
            "guided_count": 160,
            "ordered_condition_hashes": condition_hashes,
        },
        "condition_set_manifest_sha256": sha256_file(
            CONDITION_SET_MANIFEST
        ),
        "preflight_sha256": preflight_hash,
        "specifications": preflight["specifications"],
        "checkpoint_inventory_sha256": preflight[
            "checkpoint_inventory_sha256"
        ],
        "checkpoint_parity_sha256": preflight["checkpoint_parity_sha256"],
        "solver_profile": effective_parameter_record(3600),
        "solver_profile_sha256": preflight["solver_profile_sha256"],
        "maximum_concurrent_conditions": 1,
        "cplex_threads": 0,
        "deterministic_parallel_mode": 1,
        "production_time_limit_seconds": 3600,
        "rolling_canary_condition_count": 4,
        "tmux_session_name": TMUX_SESSION,
        "runner_command": runner_command,
        "tmux_launch_command": tmux_command,
        "campaign_stdout_path": stdout_path.relative_to(
            REPOSITORY
        ).as_posix(),
        "campaign_stderr_path": stderr_path.relative_to(
            REPOSITORY
        ).as_posix(),
    }
    manifest_path = campaign_root / "campaign-execution-manifest.json"
    manifest_hash = atomic_write_json(manifest_path, manifest)
    atomic_write_bytes(
        campaign_root / "campaign-execution-manifest.sha256",
        (
            f"{manifest_hash}  campaign-execution-manifest.json\n"
        ).encode("ascii"),
    )
    atomic_write_json(
        campaign_root / "launch-plan.json",
        {
            "schema_version": "phase2-production-launch-plan/1",
            "campaign_id": CAMPAIGN_ID,
            "campaign_manifest_sha256": manifest_hash,
            "runner_command": runner_command,
            "tmux_launch_command": tmux_command,
            "worker_count": 1,
            "cplex_threads": 0,
            "created_at": now(),
        },
    )
    for path in (
        campaign_root / "production-preflight.json",
        manifest_path,
        campaign_root / "campaign-execution-manifest.sha256",
        campaign_root / "launch-plan.json",
    ):
        path.chmod(0o444)
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "campaign_root": str(campaign_root),
                "campaign_execution_manifest_sha256": manifest_hash,
                "tmux_launch_command": tmux_command,
            },
            sort_keys=True,
        )
    )


def required_artifacts(incumbent: bool) -> list[str]:
    paths = [
        "ATTEMPT.json",
        "COMPLETED.json",
        "artifact-manifest.json",
        "input/condition.json",
        "input/input-hashes.json",
        "formulation/model.lp",
        "solver/cplex.commands",
        "solver/cplex.log",
        "solver/effective-parameters.json",
        "solver/raw-termination.json",
        "solver/stderr.txt",
        "solver/stdout.txt",
        "status.json",
        "provenance/environment.json",
    ]
    if incumbent:
        paths.extend(
            [
                "candidate/candidate.sdf",
                "solver/full-precision-variables.jsonl",
                "verification/linear-row-audit.json",
                "verification/predictor-equivalence.json",
                "verification/structural-reconstruction.json",
                "verification/variable-audit.json",
                "verification/verification-provenance.json",
                "verification/verification-record.json",
            ]
        )
    return paths


def validate_attempt(
    attempt: Path,
    status: dict[str, Any],
    condition: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    normalized = status.get("normalized_status")
    if normalized not in VALID_STATUSES:
        failures.append("invalid normalized status")
    if normalized in PAUSE_STATUSES:
        failures.append(f"pause status: {normalized}")
    raw = read_json(attempt / "solver/raw-termination.json")
    incumbent = bool(status.get("incumbent_available"))
    if bool(raw["has_incumbent"]) != incumbent:
        failures.append("raw/status incumbent mismatch")
    expected = {
        "TIMEOUT_NO_CANDIDATE": ("TIME_LIMIT", False),
        "INFEASIBLE_CERTIFIED": ("INTEGER_INFEASIBLE", False),
        "VERIFIED_CANDIDATE": (None, True),
    }
    if normalized in expected:
        reason, expected_incumbent = expected[normalized]
        if reason is not None and raw["termination_reason"] != reason:
            failures.append("termination/status mismatch")
        if incumbent != expected_incumbent:
            failures.append("incumbent/status mismatch")

    missing = [
        relative
        for relative in required_artifacts(incumbent)
        if not (attempt / relative).is_file()
    ]
    if missing:
        failures.append(f"missing mandatory evidence: {missing}")
    if classify_attempt(attempt) is not AttemptState.COMPLETED:
        failures.append("attempt completion state is not COMPLETED")
    artifact_manifest_path = attempt / "artifact-manifest.json"
    completion_path = attempt / "COMPLETED.json"
    if artifact_manifest_path.is_file() and completion_path.is_file():
        artifact_manifest = read_json(artifact_manifest_path)
        completion = read_json(completion_path)
        if completion["artifact_manifest_sha256"] != sha256_file(
            artifact_manifest_path
        ):
            failures.append("completion manifest hash mismatch")
        for artifact in artifact_manifest["artifacts"]:
            path = attempt / artifact["path"]
            if (
                not path.is_file()
                or path.stat().st_size != artifact["size_bytes"]
                or sha256_file(path) != artifact["sha256"]
            ):
                failures.append(
                    f"artifact integrity failure: {artifact['path']}"
                )
                break

    commands_path = attempt / "solver/cplex.commands"
    if commands_path.is_file():
        commands = commands_path.read_text(encoding="utf-8").splitlines()
        optimize = commands.index("optimize") if "optimize" in commands else -1
        for command in (
            "set threads 0",
            "set mip tolerances mipgap 0.0",
            "set mip tolerances absmipgap 0.0",
            "set mip tolerances integrality 1e-9",
            "set simplex tolerances feasibility 1e-8",
            "set emphasis numerical yes",
            "set parallel 1",
            "set preprocessing presolve yes",
            "set emphasis mip 1",
            "set paramdisplay yes",
        ):
            if commands.count(command) != 1 or commands.index(command) >= optimize:
                failures.append(f"command stream failure: {command}")
    parameters_path = attempt / "solver/effective-parameters.json"
    if parameters_path.is_file():
        parameters = read_json(parameters_path)
        if (
            parameters["threads"] != 0
            or parameters["timeLimit"] != 3600
            or parameters["parallel_mode"] != 1
        ):
            failures.append("effective solver parameter mismatch")
    environment_path = attempt / "provenance/environment.json"
    if environment_path.is_file():
        environment = read_json(environment_path)
        if (
            environment["requested_solver_threads"] != 0
            or not environment["host_cpu"]["logical_cpu_count_os"]
        ):
            failures.append("host CPU/thread provenance mismatch")
    log_path = attempt / "solver/cplex.log"
    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if "CPXPARAM_Parallel" not in log:
            failures.append("effective CPLEX parameter display missing")

    metrics = {
        "maximum_integrality_residual": None,
        "maximum_scaled_row_residual": None,
        "direct_milp_discrepancy": None,
    }
    if incumbent:
        verification = read_json(
            attempt / "verification/verification-record.json"
        )
        variable = read_json(attempt / "verification/variable-audit.json")
        rows = read_json(attempt / "verification/linear-row-audit.json")
        prediction = read_json(
            attempt / "verification/predictor-equivalence.json"
        )
        metrics = {
            "maximum_integrality_residual": variable[
                "maximum_integrality_residual"
            ]["value"],
            "maximum_scaled_row_residual": rows[
                "maximum_scaled_violation"
            ]["value"],
            "direct_milp_discrepancy": prediction[
                "absolute_standardized_discrepancy"
            ]["value"],
        }
        if (
            not verification["all_mandatory_checks_passed"]
            or not status["verification_passed"]
        ):
            failures.append("post-solve verification failed")
        if metrics["direct_milp_discrepancy"] > PREDICTOR_TOLERANCE:
            failures.append("direct/MILP discrepancy exceeds tolerance")
        if metrics["maximum_integrality_residual"] > INTEGRALITY_TOLERANCE:
            failures.append("integrality residual exceeds tolerance")
        if metrics["maximum_scaled_row_residual"] > SCALED_ROW_TOLERANCE:
            failures.append("scaled row residual exceeds tolerance")
        if not prediction["target_membership_passed"]:
            failures.append("direct prediction is outside target range")
    return {
        "schema_version": "phase2-production-condition-validation/1",
        "condition_id": condition["condition_id"],
        "attempt_id": attempt.name,
        "validated_at": now(),
        "normalized_status": normalized,
        "incumbent_available": incumbent,
        "metrics": metrics,
        "passed": not failures,
        "failures": failures,
    }


def run(campaign_root: Path) -> None:
    campaign_root = campaign_root.resolve(strict=True)
    manifest_path = campaign_root / "campaign-execution-manifest.json"
    manifest = read_json(manifest_path)
    if manifest["campaign_id"] != CAMPAIGN_ID:
        raise RuntimeError("campaign identifier mismatch")
    expected = (
        campaign_root / "campaign-execution-manifest.sha256"
    ).read_text(encoding="ascii").split()[0]
    if sha256_file(manifest_path) != expected:
        raise RuntimeError("campaign execution manifest hash mismatch")
    if sha256_file(CONDITION_MANIFEST) != CONDITION_MANIFEST_SHA256:
        raise RuntimeError("canonical condition manifest drift")
    if sha256_file(TARGET_MANIFEST) != TARGET_MANIFEST_SHA256:
        raise RuntimeError("canonical target-range manifest drift")

    lock_stream = (campaign_root / "dispatcher.lock").open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another campaign dispatcher is active") from exc
    launched = campaign_root / "LAUNCHED.json"
    if not launched.exists():
        atomic_write_json(
            launched,
            {
                "schema_version": "phase2-production-launch/1",
                "campaign_id": CAMPAIGN_ID,
                "campaign_manifest_sha256": expected,
                "launched_at": now(),
                "pid": os.getpid(),
                "worker_count": 1,
                "cplex_threads": 0,
            },
        )
    run_root = campaign_root / "runs"
    events_path = campaign_root / "campaign-events.jsonl"
    status_path = campaign_root / "campaign-status.json"
    counts: dict[str, int] = {}
    completed = 0
    canary_passed = False
    values = conditions()

    def publish(
        *,
        active_condition_id: str | None,
        paused: bool,
        pause_reason: str | None,
    ) -> None:
        atomic_write_json(
            status_path,
            {
                "schema_version": "phase2-production-campaign-status/1",
                "campaign_id": CAMPAIGN_ID,
                "updated_at": now(),
                "condition_count": 960,
                "completed_count": completed,
                "pending_count": 960 - completed,
                "active_condition_id": active_condition_id,
                "normalized_status_counts": counts,
                "canary_condition_count": 4,
                "canary_passed": canary_passed,
                "dispatch_paused": paused,
                "pause_reason": pause_reason,
                "maximum_concurrent_conditions": 1,
                "cplex_threads": 0,
            },
            overwrite=status_path.exists(),
        )

    for index, condition in enumerate(values, start=1):
        publish(
            active_condition_id=condition["condition_id"],
            paused=False,
            pause_reason=None,
        )
        try:
            attempt, status = run_condition(
                repository_root=REPOSITORY,
                run_root=run_root,
                campaign_id=CAMPAIGN_ID,
                condition=condition,
                executable=CPLEX,
                time_limit_seconds=3600,
                resume=True,
            )
        except BaseException as exc:
            failure = {
                "schema_version": "phase2-production-campaign-event/1",
                "condition_index": index,
                "condition_id": condition["condition_id"],
                "recorded_at": now(),
                "exception": {
                    "class": type(exc).__name__,
                    "message": str(exc),
                },
                "validation": {
                    "passed": False,
                    "failures": [
                        "infrastructure exception during condition execution"
                    ],
                },
            }
            with events_path.open("ab") as stream:
                stream.write(canonical_json_bytes(failure))
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(failure, sort_keys=True), flush=True)
            publish(
                active_condition_id=None,
                paused=True,
                pause_reason=(
                    "infrastructure exception: " f"{type(exc).__name__}: {exc}"
                ),
            )
            return
        validation = validate_attempt(attempt, status, condition)
        if index <= 4:
            resumed_attempt, resumed_status = run_condition(
                repository_root=REPOSITORY,
                run_root=run_root,
                campaign_id=CAMPAIGN_ID,
                condition=condition,
                executable=CPLEX,
                time_limit_seconds=3600,
                resume=True,
            )
            restart_passed = (
                resumed_attempt == attempt
                and resumed_status["attempt_id"] == status["attempt_id"]
            )
            validation["restart_protection_passed"] = restart_passed
            if not restart_passed:
                validation["passed"] = False
                validation["failures"].append(
                    "restart protection did not return the completed attempt"
                )
            canary_path = (
                campaign_root
                / "canary"
                / f"{index:02d}-{condition['condition_id']}.json"
            )
            if canary_path.exists():
                prior = read_json(canary_path)
                if (
                    prior["condition_id"] != validation["condition_id"]
                    or not prior["passed"]
                ):
                    raise RuntimeError("existing canary evidence is inconsistent")
            else:
                atomic_write_json(canary_path, validation)
        event = {
            "schema_version": "phase2-production-campaign-event/1",
            "condition_index": index,
            "condition_id": condition["condition_id"],
            "attempt_path": attempt.relative_to(campaign_root).as_posix(),
            "status": status,
            "validation": validation,
            "recorded_at": now(),
        }
        with events_path.open("ab") as stream:
            stream.write(canonical_json_bytes(event))
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(event, sort_keys=True), flush=True)
        if not validation["passed"]:
            publish(
                active_condition_id=None,
                paused=True,
                pause_reason="; ".join(validation["failures"]),
            )
            return
        completed += 1
        normalized = status["normalized_status"]
        counts[normalized] = counts.get(normalized, 0) + 1
        if index == 4:
            canary_passed = True
            canary_marker = campaign_root / "ROLLING_CANARY_PASSED.json"
            if not canary_marker.exists():
                atomic_write_json(
                    canary_marker,
                    {
                        "schema_version": "phase2-production-canary/1",
                        "campaign_id": CAMPAIGN_ID,
                        "condition_count": 4,
                        "passed": True,
                        "completed_at": now(),
                    },
                )
        publish(
            active_condition_id=None,
            paused=False,
            pause_reason=None,
        )
    atomic_write_json(
        campaign_root / "CAMPAIGN_COMPLETED.json",
        {
            "schema_version": "phase2-production-campaign-completion/1",
            "campaign_id": CAMPAIGN_ID,
            "condition_count": 960,
            "normalized_status_counts": counts,
            "completed_at": now(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--campaign-root", type=Path, required=True)
    prepare_parser.add_argument("--acceptance-attempt", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--campaign-root", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.campaign_root, arguments.acceptance_attempt.resolve())
    else:
        run(arguments.campaign_root)


if __name__ == "__main__":
    main()
