"""Small strict JSON-Schema subset used by repository-owned contracts."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from typing import Any

from .errors import ContractError

SCHEMA_NAMES = (
    "artifact-manifest",
    "candidate-freeze",
    "cplex-toy-result",
    "condition-manifest",
    "dft-manifest",
    "environment-lock",
    "external-tool-declaration",
    "phase1-asset-bundle",
    "phase1-asset-inventory",
    "phase1-inference-input",
    "phase1-parity-reference",
    "phase1-parity-record",
    "phase2-production-condition",
    "raw-solver-termination",
    "run-directory",
    "solver-parameters",
    "structural-specification",
    "target-range-manifest",
    "verification-record",
)


def load_schema(name: str) -> dict[str, Any]:
    if name not in SCHEMA_NAMES:
        raise ContractError(f"unknown schema {name!r}")
    resource = files("mol_infer_gnn_experiments.schema_documents").joinpath(
        f"{name}.schema.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def load_json(path: Path, *, schema_name: str | None = None) -> Any:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON from {path}: {exc}") from exc
    if schema_name is not None:
        validate_named(value, schema_name)
    return value


def validate(value: Any, schema: dict[str, Any]) -> None:
    errors: list[str] = []
    _validate(value, schema, schema, "$", errors)
    if errors:
        raise ContractError("; ".join(errors))


def validate_named(value: Any, schema_name: str) -> None:
    validate(value, load_schema(schema_name))
    if not isinstance(value, dict):
        return
    if schema_name == "condition-manifest":
        _ordered_bounds(value["target"], "$.target")
    elif schema_name == "structural-specification":
        feature_ids: set[str] = set()
        for index, bound in enumerate(value["bounds"]):
            _ordered_bounds(bound, f"$.bounds[{index}]")
            if bound["feature_id"] in feature_ids:
                raise ContractError(
                    f"$.bounds[{index}]: duplicate feature_id {bound['feature_id']!r}"
                )
            feature_ids.add(bound["feature_id"])
    elif schema_name == "raw-solver-termination":
        has_incumbent = value["has_incumbent"]
        if has_incumbent != (value["incumbent_count"] > 0):
            raise ContractError(
                "$: has_incumbent must agree with positive incumbent_count"
            )
    elif schema_name == "run-directory":
        expected = f"attempt-{value['attempt_number']:04d}"
        if value["attempt_id"] != expected:
            raise ContractError("$: attempt_id does not match attempt_number")
    elif schema_name == "artifact-manifest":
        _require_unique_key(value["artifacts"], "path", "$.artifacts")
    elif schema_name == "candidate-freeze":
        _require_unique_key(value["entries"], "candidate_id", "$.entries")
    elif schema_name == "dft-manifest":
        _require_unique_key(value["entries"], "dft_id", "$.entries")
    elif schema_name == "environment-lock":
        _require_unique_key(value["packages"], "name", "$.packages")
        _require_unique_key(
            value["external_components"], "name", "$.external_components"
        )
        unresolved = any(
            component["required_version"] is None
            for component in value["external_components"]
        )
        if value["scientific_execution_permitted"] and unresolved:
            raise ContractError(
                "$: unresolved external component forbids scientific execution"
            )
    elif schema_name == "external-tool-declaration":
        if value["tool_id"] == "cplex-cli":
            if value["environment_variable"] != "MOL_INFER_GNN_CPLEX_BIN":
                raise ContractError(
                    "$: cplex-cli requires MOL_INFER_GNN_CPLEX_BIN"
                )
            if value["executable_name"] != "cplex":
                raise ContractError("$: cplex-cli executable_name must be cplex")
    elif schema_name == "phase1-asset-bundle":
        _require_unique_key(value["checkpoints"], "logical_asset_id", "$.checkpoints")
        _require_unique_key(value["checkpoints"], "source_path", "$.checkpoints")
    elif schema_name == "phase1-asset-inventory":
        from .assets.catalog import (
            EXPECTED_CHECKPOINT_IDS,
            PHASE2_CHECKPOINT_IDS,
        )

        _require_unique_key(
            value["checkpoint_records"], "logical_asset_id", "$.checkpoint_records"
        )
        _require_unique_key(
            value["canonical_objects"], "destination_path", "$.canonical_objects"
        )
        if value["status"] == "ready" and len(value["checkpoint_records"]) != 36:
            raise ContractError("$: ready Phase 1 inventory requires 36 checkpoints")
        if value["status"] == "pending" and value["checkpoint_records"]:
            raise ContractError("$: pending Phase 1 inventory cannot claim checkpoints")
        if value["expected_checkpoint_ids"] != list(EXPECTED_CHECKPOINT_IDS):
            raise ContractError("$: expected Phase 1 checkpoint identities drifted")
        if value["phase2_checkpoint_ids"] != list(PHASE2_CHECKPOINT_IDS):
            raise ContractError("$: Phase 2 checkpoint subset identities drifted")
        if value["status"] == "ready":
            record_ids = [
                record["logical_asset_id"] for record in value["checkpoint_records"]
            ]
            if record_ids != list(EXPECTED_CHECKPOINT_IDS):
                raise ContractError(
                    "$: ready checkpoint record order or identity drifted"
                )
            if value["split_record"] is None:
                raise ContractError("$: ready Phase 1 inventory requires a split")
            objects = {
                item["destination_path"]: item for item in value["canonical_objects"]
            }
            if value["canonical_checkpoint_file_count"] != len(objects):
                raise ContractError(
                    "$: canonical checkpoint file count is inconsistent"
                )
            if value["deduplicated_copies_avoided"] != 36 - len(objects):
                raise ContractError("$: checkpoint deduplication count is inconsistent")
            for record in value["checkpoint_records"]:
                item = objects.get(record["destination_path"])
                if (
                    item is None
                    or item["sha256"] != record["destination_sha256"]
                    or record["source_sha256"] != record["destination_sha256"]
                    or record["logical_asset_id"] not in item["logical_asset_ids"]
                ):
                    raise ContractError(
                        "$: checkpoint record does not match its canonical object"
                    )
            if (
                value["split_record"]["source_sha256"]
                != value["split_record"]["destination_sha256"]
            ):
                raise ContractError("$: split source and destination hashes differ")
        elif (
            value["split_record"] is not None
            or value["canonical_objects"]
            or value["canonical_checkpoint_file_count"] != 0
            or value["deduplicated_copies_avoided"] != 0
        ):
            raise ContractError("$: pending Phase 1 inventory claims live assets")
    elif schema_name == "verification-record":
        status = value["normalized_status"]
        incumbent = value["has_incumbent"]
        passed = value["all_mandatory_checks_passed"]
        candidate = value["candidate_artifact_sha256"]
        _require_unique_key(value["checks"], "check_id", "$.checks")
        required_checks = {
            "required_variables",
            "integrality",
            "variable_bounds",
            "linear_constraints",
            "structural_reconstruction",
            "graph_identity",
            "predictor_equivalence",
            "target_membership",
            "provenance",
        }
        check_ids = {check["check_id"] for check in value["checks"]}
        if incumbent and check_ids != required_checks:
            raise ContractError("$: incumbent requires all mandatory check records")
        if not incumbent and (value["checks"] or passed):
            raise ContractError(
                "$: no-incumbent record cannot claim verification checks"
            )
        if status == "VERIFIED_CANDIDATE":
            if not incumbent or not passed or candidate is None:
                raise ContractError("$: verified candidate evidence is incomplete")
        elif status == "POSTSOLVE_REJECTED":
            if not incumbent or passed:
                raise ContractError("$: rejected incumbent evidence is inconsistent")
        elif incumbent:
            raise ContractError(
                "$: incumbent must normalize to verified or post-solve rejected"
            )


def _ordered_bounds(value: dict[str, Any], path: str) -> None:
    try:
        lower = Decimal(value["lower"])
        upper = Decimal(value["upper"])
    except InvalidOperation as exc:
        raise ContractError(f"{path}: invalid decimal bound") from exc
    if lower > upper:
        raise ContractError(f"{path}: lower bound exceeds upper bound")


def _require_unique_key(
    values: list[dict[str, Any]], key: str, path: str
) -> None:
    identifiers = [value[key] for value in values]
    if len(identifiers) != len(set(identifiers)):
        raise ContractError(f"{path}: duplicate {key}")


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ContractError(f"only local schema references are supported: {ref}")
    node: Any = root
    for component in ref[2:].split("/"):
        node = node[component.replace("~1", "/").replace("~0", "~")]
    return node


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[expected]


def _validate(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if "$ref" in schema:
        _validate(value, _resolve_ref(schema["$ref"], root), root, path, errors)
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not in allowed vocabulary")
    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else expected
        if not any(_matches_type(value, item) for item in expected_types):
            errors.append(f"{path}: expected {' or '.join(expected_types)}")
            return
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required field {required!r}")
        if schema.get("additionalProperties") is False:
            for key in value.keys() - properties.keys():
                errors.append(f"{path}: unknown field {key!r}")
        for key, child in value.items():
            if key in properties:
                _validate(child, properties[key], root, f"{path}.{key}", errors)
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: too many items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: items must be unique")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate(item, item_schema, root, f"{path}[{index}]", errors)
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: string is too short")
        pattern = schema.get("pattern")
        if pattern and re.fullmatch(pattern, value) is None:
            errors.append(f"{path}: does not match {pattern!r}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
