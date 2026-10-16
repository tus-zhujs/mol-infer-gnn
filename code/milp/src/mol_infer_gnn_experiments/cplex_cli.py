"""Declared external CPLEX CLI audit and non-scientific PuLP smoke test."""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, atomic_write_json, sha256_file
from .errors import AttemptExistsError, ExternalToolError, PathClosureError
from .path_closure import RuntimePathPolicy
from .schema import load_json, validate_named

CPLEX_ENV_VAR = "MOL_INFER_GNN_CPLEX_BIN"
EXPECTED_CPLEX_VERSION = "22.1.1"
_VERSION = re.compile(
    r"Interactive Optimizer\s+([0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)?)"
)
_LICENSE_FAILURE = re.compile(
    r"(no\s+license|license\s+(?:error|failed|failure|not found|unavailable)|"
    r"cannot\s+find.*license|licensed number of users already reached)",
    re.IGNORECASE,
)
_RELEVANT_ENV = re.compile(
    r"(CPLEX|ILOG|IBM|LICENSE|^LM_|MOL_INFER_GNN_CPLEX_BIN)", re.IGNORECASE
)
_SECRET_ENV = re.compile(
    r"(LICENSE|TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CplexResolution:
    source: str
    requested_path: str | None
    resolved_real_path: str | None
    executable_declared: bool
    executable_found: bool
    executable_permission: bool
    failure_reason: str | None


@dataclass(frozen=True)
class CplexCliPreflight:
    executable_declared: bool
    resolution_source: str
    requested_path: str | None
    executable_found: bool
    resolved_real_path: str | None
    executable_permission: bool
    executable_size_bytes: int | None
    executable_sha256: str | None
    dynamic_libraries_resolved: bool | None
    dynamic_library_output: str
    banner: str
    banner_exit_code: int | None
    actual_version: str | None
    exact_version: str | None
    expected_version: str
    expected_version_matched: bool
    version_matches: bool
    pulp_version: str | None
    pulp_adapter_available: bool
    license_usable: bool
    license_available: bool
    toy_solve_passed: bool
    scientific_solve_permitted: bool
    execution_permitted: bool
    failure_reason: str | None
    stage_failures: dict[str, str | None]
    relevant_environment: dict[str, str]
    probe_performed: bool
    executable_visible: bool
    executable_path: str | None
    python_api_visible: bool
    python_api_location: str | None
    license_visibility: str


def load_external_tool_declaration(path: Path) -> dict[str, Any]:
    value = load_json(path, schema_name="external-tool-declaration")
    if not isinstance(value, dict):
        raise ExternalToolError("external tool declaration must be an object")
    return value


def redact_relevant_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Retain discovery context while never serializing license-like values."""
    return {
        name: "[REDACTED]" if _SECRET_ENV.search(name) else value
        for name, value in sorted(environment.items())
        if _RELEVANT_ENV.search(name)
    }


def resolve_cplex_executable(
    *,
    explicit_path: str | None = None,
    environment: Mapping[str, str] | None = None,
    declaration: Mapping[str, Any] | None = None,
) -> CplexResolution:
    """Resolve CLI argument, environment declaration, then PATH."""
    environment = os.environ if environment is None else environment
    source = "unavailable"
    requested: str | None = None
    if explicit_path:
        source, requested = "cli-argument", explicit_path
    elif environment.get(CPLEX_ENV_VAR):
        source, requested = "environment-variable", environment[CPLEX_ENV_VAR]
    else:
        discovered = shutil.which("cplex", path=environment.get("PATH"))
        if discovered:
            source, requested = "PATH", discovered
    if requested is None:
        return CplexResolution(
            source=source,
            requested_path=None,
            resolved_real_path=None,
            executable_declared=declaration is not None,
            executable_found=False,
            executable_permission=False,
            failure_reason="CPLEX executable is unavailable",
        )

    candidate = Path(requested)
    if not candidate.is_absolute():
        discovered = shutil.which(requested, path=environment.get("PATH"))
        if discovered is None:
            return CplexResolution(
                source=source,
                requested_path=requested,
                resolved_real_path=None,
                executable_declared=declaration is not None,
                executable_found=False,
                executable_permission=False,
                failure_reason=f"CPLEX executable was not found: {requested}",
            )
        candidate = Path(discovered)

    if source != "PATH" and declaration is None:
        return CplexResolution(
            source=source,
            requested_path=requested,
            resolved_real_path=None,
            executable_declared=False,
            executable_found=False,
            executable_permission=False,
            failure_reason="absolute external executable path is undeclared",
        )
    try:
        policy = RuntimePathPolicy(Path.cwd())
        resolved = policy.resolve_external_executable(
            str(candidate),
            declared=declaration is not None or source == "PATH",
            must_exist=True,
        )
    except (OSError, PathClosureError) as exc:
        return CplexResolution(
            source=source,
            requested_path=requested,
            resolved_real_path=None,
            executable_declared=declaration is not None,
            executable_found=False,
            executable_permission=False,
            failure_reason=str(exc),
        )
    executable = os.access(resolved, os.X_OK)
    return CplexResolution(
        source=source,
        requested_path=requested,
        resolved_real_path=str(resolved),
        executable_declared=declaration is not None,
        executable_found=resolved.is_file(),
        executable_permission=executable,
        failure_reason=None if executable else "CPLEX path is not executable",
    )


def _canonical_version(actual: str | None) -> str | None:
    if actual is None:
        return None
    return ".".join(actual.split(".")[:3])


def _load_pulp() -> Any | None:
    try:
        return importlib.import_module("pulp")
    except ImportError:
        return None


def audit_cplex_cli(
    *,
    explicit_path: str | None = None,
    environment: Mapping[str, str] | None = None,
    declaration: Mapping[str, Any] | None = None,
    probe_license: bool = False,
    toy_solve_passed: bool = False,
) -> CplexCliPreflight:
    """Audit the external CLI without creating a mathematical model."""
    environment = os.environ if environment is None else environment
    resolution = resolve_cplex_executable(
        explicit_path=explicit_path,
        environment=environment,
        declaration=declaration,
    )
    stage_failures: dict[str, str | None] = {
        "declaration": None
        if resolution.executable_declared
        else "external CPLEX interface was not declared",
        "discovery": resolution.failure_reason,
        "dynamic_libraries": (
            None
            if resolution.executable_found
            else "not attempted because the executable was unavailable"
        ),
        "version": None,
        "pulp_adapter": None,
        "license": None,
        "toy_solve": None if toy_solve_passed else "toy solve has not passed",
    }
    size: int | None = None
    digest: str | None = None
    ldd_ok: bool | None = None
    ldd_output = ""
    banner = ""
    banner_exit: int | None = None
    actual_version: str | None = None
    license_usable = False
    resolved = (
        Path(resolution.resolved_real_path)
        if resolution.resolved_real_path is not None
        else None
    )
    if resolved is not None and resolution.executable_permission:
        size = resolved.stat().st_size
        digest = sha256_file(resolved)
        ldd_executable = shutil.which("ldd")
        if ldd_executable is None:
            stage_failures["dynamic_libraries"] = "ldd is unavailable"
        else:
            try:
                ldd = subprocess.run(
                    [ldd_executable, str(resolved)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=dict(environment),
                )
                ldd_output = (ldd.stdout + ldd.stderr).strip()
                ldd_ok = ldd.returncode == 0 and "not found" not in ldd_output
                if not ldd_ok:
                    stage_failures["dynamic_libraries"] = (
                        f"ldd exited {ldd.returncode} or reported an "
                        "unresolved library"
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                stage_failures["dynamic_libraries"] = f"ldd failed: {exc}"
        if probe_license:
            with tempfile.TemporaryDirectory(prefix="mol-infer-cplex-audit-") as work:
                try:
                    completed = subprocess.run(
                        [str(resolved), "-c", "quit"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=work,
                        env=dict(environment),
                    )
                    banner_exit = completed.returncode
                    banner = (completed.stdout + completed.stderr).strip()
                    match = _VERSION.search(banner)
                    actual_version = match.group(1) if match else None
                    license_usable = (
                        completed.returncode == 0
                        and actual_version is not None
                        and _LICENSE_FAILURE.search(banner) is None
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    banner = str(exc)
                    stage_failures["license"] = f"CPLEX initialization failed: {exc}"
        else:
            stage_failures["license"] = "license initialization was not requested"
    elif resolution.executable_found:
        stage_failures["discovery"] = "CPLEX path is not executable"

    canonical_version = _canonical_version(actual_version)
    version_matches = canonical_version == EXPECTED_CPLEX_VERSION
    if actual_version is None:
        stage_failures["version"] = "CPLEX banner version was not established"
    elif not version_matches:
        stage_failures["version"] = (
            f"expected {EXPECTED_CPLEX_VERSION}, found {actual_version}"
        )
    if probe_license and not license_usable and stage_failures["license"] is None:
        stage_failures["license"] = "CPLEX license initialization did not pass"

    pulp = _load_pulp()
    pulp_version = getattr(pulp, "__version__", None) if pulp else None
    adapter_available = False
    if pulp is not None and resolved is not None:
        try:
            adapter_available = bool(pulp.CPLEX_CMD(path=str(resolved)).available())
        except Exception as exc:  # pragma: no cover - third-party boundary
            stage_failures["pulp_adapter"] = f"PuLP adapter failed: {exc}"
    if not adapter_available and stage_failures["pulp_adapter"] is None:
        stage_failures["pulp_adapter"] = (
            "PuLP is unavailable"
            if pulp is None
            else "pulp.CPLEX_CMD(path=...) reports unavailable"
        )

    permitted = all(
        (
            resolution.executable_declared,
            resolution.executable_found,
            resolution.executable_permission,
            ldd_ok is True,
            version_matches,
            adapter_available,
            license_usable,
            toy_solve_passed,
        )
    )
    failure_reason = next(
        (
            stage_failures[name]
            for name in (
                "discovery",
                "declaration",
                "dynamic_libraries",
                "version",
                "pulp_adapter",
                "license",
                "toy_solve",
            )
            if stage_failures[name] is not None
        ),
        None,
    )
    return CplexCliPreflight(
        executable_declared=resolution.executable_declared,
        resolution_source=resolution.source,
        requested_path=resolution.requested_path,
        executable_found=resolution.executable_found,
        resolved_real_path=resolution.resolved_real_path,
        executable_permission=resolution.executable_permission,
        executable_size_bytes=size,
        executable_sha256=digest,
        dynamic_libraries_resolved=ldd_ok,
        dynamic_library_output=ldd_output,
        banner=banner,
        banner_exit_code=banner_exit,
        actual_version=actual_version,
        exact_version=canonical_version,
        expected_version=EXPECTED_CPLEX_VERSION,
        expected_version_matched=version_matches,
        version_matches=version_matches,
        pulp_version=pulp_version,
        pulp_adapter_available=adapter_available,
        license_usable=license_usable,
        license_available=license_usable,
        toy_solve_passed=toy_solve_passed,
        scientific_solve_permitted=permitted,
        execution_permitted=permitted,
        failure_reason=failure_reason,
        stage_failures=stage_failures,
        relevant_environment=redact_relevant_environment(environment),
        probe_performed=probe_license,
        executable_visible=resolution.executable_found,
        executable_path=resolution.resolved_real_path,
        python_api_visible=False,
        python_api_location=None,
        license_visibility=(
            "validated by no-model CPLEX CLI initialization"
            if license_usable
            else "not validated"
        ),
    )


def _claim_toy_attempt(output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    attempt_id = (
        f"toy-cplex-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:12]}"
    )
    attempt = output_root / attempt_id
    try:
        os.mkdir(attempt)
    except FileExistsError as exc:  # practically impossible; fail closed
        raise AttemptExistsError(f"toy attempt already exists: {attempt}") from exc
    return attempt


def _artifact(path: Path, root: Path, media_type: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "media_type": media_type,
        "required": True,
    }


def _termination_text(log_text: str) -> str:
    markers = ("Solution status", "MIP - ", "Integer optimal", "time limit")
    lines = [line for line in log_text.splitlines() if any(x in line for x in markers)]
    return "\n".join(lines if lines else log_text.splitlines()[-40:])


def canonical_cplex_command_lines(
    *,
    model_path: str,
    solution_path: str,
    time_limit_seconds: float,
    log_path: str | None = None,
) -> list[str]:
    """Return the complete frozen command stream in execution order."""
    lines = [f'read "{model_path}"']
    if log_path is not None:
        lines.append(f'set logfile "{log_path}"')
    lines.extend(
        (
            f"set timelimit {time_limit_seconds}",
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
            "set randomseed 0",
            "optimize",
            f'write "{solution_path}"',
            "quit",
        )
    )
    return lines


def _retained_solver(pulp: Any, executable: Path, attempt: Path, timeout: int) -> Any:
    """Create a CPLEX_CMD subclass with explicit retained names and process evidence."""
    log_path = attempt / "cplex.log"

    class RetainedCplexCmd(pulp.CPLEX_CMD):
        process_exit_code: int | None = None
        captured_stdout = b""
        captured_stderr = b""
        command_bytes = b""
        parse_failure: str | None = None

        def create_tmp_files(self, _name: str, *suffixes: str):
            names = {"lp": "model.lp", "sol": "solution.sol", "mst": "warm-start.mst"}
            paths = [attempt / names[suffix] for suffix in suffixes]
            if any(path.exists() for path in paths):
                raise AttemptExistsError("refusing to overwrite a toy-solve artifact")
            return (str(path) for path in paths)

        def delete_tmp_files(self, *_paths: str) -> None:
            return

        def actualSolve(self, lp):  # noqa: N802 - PuLP public API
            if not self.executable(self.path):
                raise ExternalToolError(f"PuLP cannot execute {self.path}")
            tmp_lp, tmp_sol, tmp_mst = self.create_tmp_files(
                lp.name, "lp", "sol", "mst"
            )
            variables = lp.writeLP(tmp_lp, writeSOS=1)
            if self.optionsDict.get("warmStart", False):
                raise ExternalToolError("canonical profile forbids warm starts")
            commands = "\n".join(
                canonical_cplex_command_lines(
                    model_path=tmp_lp,
                    solution_path=tmp_sol,
                    time_limit_seconds=self.timeLimit,
                    log_path=str(log_path),
                )
            ) + "\n"
            self.command_bytes = commands.encode("utf-8")
            process = subprocess.Popen(
                [self.path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=attempt,
            )
            self.captured_stdout, self.captured_stderr = process.communicate(
                self.command_bytes
            )
            self.process_exit_code = process.returncode
            status = pulp.constants.LpStatusUndefined
            solution_status = None
            if process.returncode == 0 and Path(tmp_sol).is_file():
                try:
                    (
                        status,
                        values,
                        reduced_costs,
                        shadow_prices,
                        slacks,
                        solution_status,
                    ) = self.readsol(tmp_sol)
                    lp.assignVarsVals(values)
                    lp.assignVarsDj(reduced_costs)
                    lp.assignConsPi(shadow_prices)
                    lp.assignConsSlack(slacks)
                except Exception as exc:  # preserve raw evidence; never reinterpret
                    self.parse_failure = str(exc)
            lp.assignStatus(status, solution_status)
            return status

    return RetainedCplexCmd(
        path=str(executable),
        mip=True,
        msg=False,
        timeLimit=timeout,
        gapRel=0,
        gapAbs=0,
        threads=0,
        options=[],
        warmStart=False,
        keepFiles=False,
        logPath=f'"{log_path}"',
    )


def run_toy_smoke(
    *,
    executable: Path,
    output_root: Path,
    timeout_seconds: int = 10,
    expected_version: str = EXPECTED_CPLEX_VERSION,
) -> tuple[Path, dict[str, Any]]:
    """Run one two-variable infrastructure MIP and publish completion last."""
    if not 1 <= timeout_seconds <= 30:
        raise ExternalToolError("toy timeout must be between 1 and 30 seconds")
    pulp = _load_pulp()
    if pulp is None:
        raise ExternalToolError("PuLP is required for the CPLEX CLI toy smoke test")
    executable = Path(executable).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ExternalToolError(f"CPLEX executable is unavailable: {executable}")
    attempt = _claim_toy_attempt(output_root)
    solver = _retained_solver(pulp, executable, attempt, timeout_seconds)

    problem = pulp.LpProblem("wp2a_cplex_cli_toy", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0, upBound=1, cat=pulp.LpBinary)
    y = pulp.LpVariable("y", lowBound=0, upBound=1, cat=pulp.LpBinary)
    problem += 3 * x + 2 * y, "toy_objective"
    problem += x + y <= 1, "choose_at_most_one"
    problem.solve(solver)

    stdout_path = attempt / "stdout.txt"
    stderr_path = attempt / "stderr.txt"
    commands_path = attempt / "cplex.commands"
    atomic_write_bytes(stdout_path, solver.captured_stdout)
    atomic_write_bytes(stderr_path, solver.captured_stderr)
    atomic_write_bytes(commands_path, solver.command_bytes)
    log_path = attempt / "cplex.log"
    model_path = attempt / "model.lp"
    solution_path = attempt / "solution.sol"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    version_match = _VERSION.search(
        solver.captured_stdout.decode("utf-8", errors="replace")
        + solver.captured_stderr.decode("utf-8", errors="replace")
        + log_text
    )
    actual_version = version_match.group(1) if version_match else None
    x_value = x.value()
    y_value = y.value()
    objective = pulp.value(problem.objective)
    feasible = (
        x_value is not None
        and y_value is not None
        and x_value >= -1e-9
        and y_value >= -1e-9
        and x_value <= 1 + 1e-9
        and y_value <= 1 + 1e-9
        and x_value + y_value <= 1 + 1e-9
    )
    integral = (
        x_value is not None
        and y_value is not None
        and abs(x_value - round(x_value)) <= 1e-9
        and abs(y_value - round(y_value)) <= 1e-9
    )
    objective_matches = objective is not None and abs(objective - 3.0) <= 1e-9
    independent = {
        "expected_optimum": 3,
        "feasible": feasible,
        "integral": integral,
        "objective_matches": objective_matches,
    }
    passed = all(
        (
            solver.process_exit_code == 0,
            solution_path.is_file(),
            _canonical_version(actual_version) == expected_version,
            feasible,
            integral,
            objective_matches,
            x_value is not None and abs(x_value - 1.0) <= 1e-9,
            y_value is not None and abs(y_value) <= 1e-9,
        )
    )
    result = {
        "schema_version": "cplex-toy-result/1",
        "scientific_execution": False,
        "attempt_id": attempt.name,
        "cplex_executable": str(executable),
        "cplex_version": actual_version or "0.0.0",
        "pulp_version": str(pulp.__version__),
        "parameters": {
            "threads": 0,
            "parallel_mode": "deterministic",
            "random_seed": 0,
            "time_limit_seconds": timeout_seconds,
            "relative_mip_gap": "0",
            "absolute_mip_gap": "0",
            "warm_start": False,
        },
        "process_exit_code": solver.process_exit_code,
        "stdout_path": stdout_path.name,
        "stderr_path": stderr_path.name,
        "cplex_log_path": log_path.name,
        "model_path": model_path.name,
        "solution_path": solution_path.name,
        "solution_file_exists": solution_path.is_file(),
        "raw_cplex_termination_text": _termination_text(log_text),
        "pulp_status": pulp.LpStatus.get(problem.status, str(problem.status)),
        "variable_values": {"x": x_value, "y": y_value},
        "objective_value": objective,
        "independent_verification": independent,
        "toy_solve_passed": passed,
    }
    validate_named(result, "cplex-toy-result")
    result_path = attempt / "toy-solve.json"
    atomic_write_json(result_path, result)
    artifacts = [
        _artifact(model_path, attempt, "application/lp"),
        _artifact(log_path, attempt, "text/plain"),
        _artifact(solution_path, attempt, "application/xml"),
        _artifact(stdout_path, attempt, "text/plain"),
        _artifact(stderr_path, attempt, "text/plain"),
        _artifact(commands_path, attempt, "text/plain"),
        _artifact(result_path, attempt, "application/json"),
    ]
    manifest = {
        "schema_version": "artifact-manifest/1",
        "manifest_id": f"{attempt.name}-artifacts",
        "root_kind": "attempt",
        "artifacts": artifacts,
    }
    validate_named(manifest, "artifact-manifest")
    manifest_hash = atomic_write_json(attempt / "artifact-manifest.json", manifest)
    atomic_write_json(
        attempt / "COMPLETED.json",
        {
            "schema_version": "cplex-toy-completion/1",
            "attempt_id": attempt.name,
            "artifact_manifest_sha256": manifest_hash,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return attempt, result


def preflight_dict(preflight: CplexCliPreflight) -> dict[str, Any]:
    return asdict(preflight)
