"""Immutable, restart-aware attempt-directory framework."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from .canonical import atomic_write_json, sha256_file
from .errors import AttemptExistsError, ContractError
from .schema import validate_named

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
ATTEMPT_SUBDIRECTORIES = (
    "input",
    "formulation",
    "solver",
    "candidate",
    "verification",
    "provenance",
)


class AttemptState(StrEnum):
    ABSENT = "ABSENT"
    CLAIMED = "CLAIMED"
    INCOMPLETE = "INCOMPLETE"
    COMPLETED = "COMPLETED"
    CORRUPT = "CORRUPT"


@dataclass(frozen=True)
class AttemptClaim:
    schema_version: str
    campaign_id: str
    condition_id: str
    attempt_id: str
    attempt_number: int
    state: str
    created_at: str
    retry_of: str | None
    retry_reason: str | None
    required_artifacts: list[str]


def _checked_id(value: str, label: str) -> str:
    if _ID.fullmatch(value) is None:
        raise ContractError(f"invalid {label}: {value!r}")
    return value


def attempt_path(
    run_root: Path, campaign_id: str, condition_id: str, attempt_number: int
) -> Path:
    if attempt_number < 1 or attempt_number > 9999:
        raise ContractError("attempt number must be in 1..9999")
    campaign_id = _checked_id(campaign_id, "campaign_id")
    condition_id = _checked_id(condition_id, "condition_id")
    return (
        Path(run_root)
        / campaign_id
        / condition_id
        / f"attempt-{attempt_number:04d}"
    )


def claim_attempt(
    run_root: Path,
    campaign_id: str,
    condition_id: str,
    attempt_number: int,
    *,
    required_artifacts: list[str],
    retry_of: str | None = None,
    retry_reason: str | None = None,
) -> Path:
    path = attempt_path(run_root, campaign_id, condition_id, attempt_number)
    if (retry_of is None) != (retry_reason is None):
        raise ContractError("retry_of and retry_reason must be supplied together")
    if retry_of is not None and re.fullmatch(r"attempt-[0-9]{4}", retry_of) is None:
        raise ContractError("invalid retry_of attempt identifier")
    claim = AttemptClaim(
        schema_version="run-directory/1",
        campaign_id=campaign_id,
        condition_id=condition_id,
        attempt_id=path.name,
        attempt_number=attempt_number,
        state="CLAIMED",
        created_at=datetime.now(UTC).isoformat(),
        retry_of=retry_of,
        retry_reason=retry_reason,
        required_artifacts=required_artifacts,
    )
    claim_value = asdict(claim)
    validate_named(claim_value, "run-directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(path)
    except FileExistsError as exc:
        raise AttemptExistsError(
            f"attempt already exists and is immutable: {path}"
        ) from exc
    for subdirectory in ATTEMPT_SUBDIRECTORIES:
        (path / subdirectory).mkdir()
    atomic_write_json(path / "ATTEMPT.json", claim_value)
    return path


def classify_attempt(path: Path) -> AttemptState:
    path = Path(path)
    if not path.exists():
        return AttemptState.ABSENT
    if not path.is_dir() or not (path / "ATTEMPT.json").is_file():
        return AttemptState.CORRUPT
    completion = path / "COMPLETED.json"
    if completion.exists():
        return AttemptState.COMPLETED if completion.is_file() else AttemptState.CORRUPT
    populated = any(
        child.is_file()
        for subdirectory in ATTEMPT_SUBDIRECTORIES
        for child in (path / subdirectory).rglob("*")
    )
    return AttemptState.INCOMPLETE if populated else AttemptState.CLAIMED


def complete_attempt(path: Path, artifact_manifest: dict[str, object]) -> str:
    """Validate artifact bytes and publish the completion marker last."""
    path = Path(path)
    if classify_attempt(path) not in {AttemptState.CLAIMED, AttemptState.INCOMPLETE}:
        raise ContractError(f"attempt cannot be completed in current state: {path}")
    validate_named(artifact_manifest, "artifact-manifest")
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ContractError("artifact manifest requires an artifacts list")
    artifact_paths = {artifact["path"] for artifact in artifacts}
    claim = json.loads((path / "ATTEMPT.json").read_text(encoding="utf-8"))
    missing = set(claim["required_artifacts"]) - artifact_paths
    if missing:
        raise ContractError(f"required artifacts missing: {sorted(missing)}")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ContractError("artifact record must be an object")
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(relative, str) or ".." in Path(relative).parts:
            raise ContractError(f"unsafe artifact path: {relative!r}")
        candidate = (path / relative).resolve(strict=True)
        if not candidate.is_relative_to(path.resolve()):
            raise ContractError(f"artifact escapes attempt: {relative}")
        if candidate.stat().st_size != artifact["size_bytes"]:
            raise ContractError(f"artifact size mismatch: {relative}")
        if sha256_file(candidate) != expected:
            raise ContractError(f"artifact hash mismatch: {relative}")
    manifest_hash = atomic_write_json(
        path / "artifact-manifest.json", artifact_manifest
    )
    completion = {
        "schema_version": "attempt-completion/1",
        "attempt_id": path.name,
        "artifact_manifest_sha256": manifest_hash,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(path / "COMPLETED.json", completion)
    return manifest_hash
