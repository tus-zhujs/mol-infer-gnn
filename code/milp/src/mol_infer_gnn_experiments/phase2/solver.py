"""Canonical CPLEX 22.1.1 CLI execution for the migrated PuLP model."""

from __future__ import annotations

import subprocess
import time
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

from ..cplex_cli import canonical_cplex_command_lines
from ..errors import AttemptExistsError, ExternalToolError
from ..status import RawTerminationReason


def effective_parameter_record(time_limit_seconds: float) -> dict[str, Any]:
    return {
        "profile_id": "cplex-22.1.1-deterministic-v1",
        "timeLimit": time_limit_seconds,
        "threads": 0,
        "threads_policy": "automatic",
        "gapRel": 0.0,
        "gapAbs": 0.0,
        "warmStart": False,
        "mip_integrality_tolerance": "1e-9",
        "simplex_feasibility_tolerance": "1e-8",
        "numerical_emphasis": True,
        "parallel_mode": 1,
        "presolve": True,
        "mip_emphasis": 1,
        "parameter_display": True,
        "random_seed": 0,
        "cuts": "CPLEX default",
        "heuristics": "CPLEX default",
        "search_strategy": "CPLEX default",
    }


def detect_termination(
    text: str, *, process_exit_code: int | None
) -> RawTerminationReason:
    lowered = text.lower()
    if (
        "integer infeasible" in lowered
        or "mip - integer infeasible" in lowered
        or "integer solution does not exist" in lowered
    ):
        return RawTerminationReason.INTEGER_INFEASIBLE
    if "integer optimal solution" in lowered or "mip - integer optimal" in lowered:
        return RawTerminationReason.OPTIMAL
    if "time limit exceeded" in lowered or "time limit, no integer solution" in lowered:
        return RawTerminationReason.TIME_LIMIT
    if "out of memory" in lowered or "memory limit exceeded" in lowered:
        return RawTerminationReason.MEMORY_LIMIT
    if process_exit_code not in {None, 0}:
        return RawTerminationReason.SOLVER_FAILURE
    return RawTerminationReason.UNKNOWN


def _solution_header(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    header = element_tree.parse(path).getroot().find("header")
    if header is None:
        return None
    return dict(header.attrib)


def canonical_cplex_solver(
    *,
    executable: Path,
    attempt: Path,
    time_limit_seconds: float = 3600,
) -> Any:
    """Construct the production adapter using standard PuLP."""
    import pulp

    executable = executable.resolve(strict=True)
    solver_root = attempt / "solver"
    formulation_root = attempt / "formulation"
    model_path = formulation_root / "model.lp"
    solution_path = solver_root / "solution.sol"
    log_path = solver_root / "cplex.log"

    class CanonicalCplexCmd(pulp.CPLEX_CMD):
        process_exit_code: int | None = None
        captured_stdout = b""
        captured_stderr = b""
        command_bytes = b""
        solver_runtime_seconds: float | None = None
        solution_header: dict[str, Any] | None = None
        parse_failure: str | None = None

        def actualSolve(self, lp):  # noqa: N802 - third-party public API
            if not self.executable(self.path):
                raise ExternalToolError(f"cannot execute CPLEX: {self.path}")
            if any(path.exists() for path in (model_path, solution_path, log_path)):
                raise AttemptExistsError("refusing to overwrite solver evidence")
            variables = lp.writeLP(str(model_path), writeSOS=1)
            lines = canonical_cplex_command_lines(
                model_path=str(model_path),
                solution_path=str(solution_path),
                time_limit_seconds=self.timeLimit,
                log_path=str(log_path),
            )
            self.command_bytes = ("\n".join(lines) + "\n").encode("utf-8")
            started = time.monotonic()
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
            self.solver_runtime_seconds = time.monotonic() - started
            self.process_exit_code = process.returncode
            status = pulp.constants.LpStatusUndefined
            solution_status = None
            self.solution_header = _solution_header(solution_path)
            if self.solution_header is not None:
                try:
                    root = element_tree.parse(solution_path).getroot()
                    values = {
                        node.get("name"): float(node.get("value"))
                        for node in root.find("variables") or ()
                    }
                    for variable in lp.variables():
                        variable.varValue = values.get(variable.name, 0.0)
                    code = self.solution_header.get("solutionStatusValue")
                    if code in {"1", "101", "102"}:
                        status = pulp.constants.LpStatusOptimal
                    else:
                        solution_status = (
                            pulp.constants.LpSolutionIntegerFeasible
                        )
                except Exception as exc:
                    self.parse_failure = str(exc)
            lp.assignStatus(status, solution_status)
            return status

    return CanonicalCplexCmd(
        path=str(executable),
        mip=True,
        msg=False,
        timeLimit=time_limit_seconds,
        gapRel=0.0,
        gapAbs=0.0,
        threads=0,
        options=[],
        warmStart=False,
        keepFiles=True,
    )
