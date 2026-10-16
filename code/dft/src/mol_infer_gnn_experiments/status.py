"""Frozen solver status vocabularies."""

from enum import StrEnum


class NormalizedStatus(StrEnum):
    VERIFIED_CANDIDATE = "VERIFIED_CANDIDATE"
    INFEASIBLE_CERTIFIED = "INFEASIBLE_CERTIFIED"
    TIMEOUT_NO_CANDIDATE = "TIMEOUT_NO_CANDIDATE"
    SOLVER_ERROR = "SOLVER_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    POSTSOLVE_REJECTED = "POSTSOLVE_REJECTED"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


class RawTerminationReason(StrEnum):
    OPTIMAL = "OPTIMAL"
    INTEGER_INFEASIBLE = "INTEGER_INFEASIBLE"
    TIME_LIMIT = "TIME_LIMIT"
    MEMORY_LIMIT = "MEMORY_LIMIT"
    INTERRUPTED = "INTERRUPTED"
    SOLVER_FAILURE = "SOLVER_FAILURE"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    UNKNOWN = "UNKNOWN"


def normalize_termination(
    termination: RawTerminationReason,
    *,
    has_incumbent: bool,
    verification_passed: bool | None,
) -> NormalizedStatus:
    """Map evidence to the frozen vocabulary without conflating timeout/infeasible."""
    if has_incumbent:
        if verification_passed is True:
            return NormalizedStatus.VERIFIED_CANDIDATE
        if verification_passed is False:
            return NormalizedStatus.POSTSOLVE_REJECTED
        raise ValueError("incumbent requires an explicit verification result")
    if verification_passed is not None:
        raise ValueError("verification result is invalid without an incumbent")
    if termination is RawTerminationReason.INTEGER_INFEASIBLE:
        return NormalizedStatus.INFEASIBLE_CERTIFIED
    if termination is RawTerminationReason.TIME_LIMIT:
        return NormalizedStatus.TIMEOUT_NO_CANDIDATE
    if termination is RawTerminationReason.MEMORY_LIMIT:
        return NormalizedStatus.MEMORY_ERROR
    if termination is RawTerminationReason.INFRASTRUCTURE_FAILURE:
        return NormalizedStatus.INFRASTRUCTURE_ERROR
    return NormalizedStatus.SOLVER_ERROR
