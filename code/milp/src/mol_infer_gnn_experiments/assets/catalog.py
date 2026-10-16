"""Frozen logical identity catalog for the approved Phase 1 population."""

from __future__ import annotations

from dataclasses import dataclass

TARGET_PROPERTIES = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "cv",
    "u0_atom",
)
PHASE2_PROPERTIES = frozenset(("homo", "lumo", "gap", "zpve"))
ARCHITECTURES = ((2, 16), (2, 32), (3, 16), (3, 32))
SPLIT_ID = "qm9-seed42-v1"
SPLIT_SHA256 = "c7f8e0246cf3aff6ced4c135fd8a8db6c6a9d4e85d231b804158fe0870aeb06d"
TEST_ORDER_SHA256 = (
    "07580bc4174252a7110f880fa4d1368a23044f7f78339a90cfc69f279cf7e13b"
)


@dataclass(frozen=True)
class CheckpointIdentity:
    logical_asset_id: str
    target_property: str
    model_depth: int
    hidden_width: int
    architecture_id: str
    phase2_selected: bool


def checkpoint_id(target_property: str, model_depth: int, hidden_width: int) -> str:
    return f"journal_{target_property}_{model_depth}lay_h{hidden_width}"


def expected_checkpoint_identities() -> tuple[CheckpointIdentity, ...]:
    return tuple(
        CheckpointIdentity(
            logical_asset_id=checkpoint_id(target, depth, width),
            target_property=target,
            model_depth=depth,
            hidden_width=width,
            architecture_id=f"gnn-d{depth}-h{width}",
            phase2_selected=target in PHASE2_PROPERTIES,
        )
        for target in TARGET_PROPERTIES
        for depth, width in ARCHITECTURES
    )


EXPECTED_CHECKPOINTS = expected_checkpoint_identities()
EXPECTED_BY_ID = {item.logical_asset_id: item for item in EXPECTED_CHECKPOINTS}
EXPECTED_CHECKPOINT_IDS = tuple(item.logical_asset_id for item in EXPECTED_CHECKPOINTS)
PHASE2_CHECKPOINT_IDS = tuple(
    item.logical_asset_id for item in EXPECTED_CHECKPOINTS if item.phase2_selected
)


def validate_catalog() -> None:
    if len(EXPECTED_CHECKPOINTS) != 36 or len(EXPECTED_BY_ID) != 36:
        raise RuntimeError("Phase 1 catalog must contain 36 unique identities")
    if len(PHASE2_CHECKPOINT_IDS) != 16:
        raise RuntimeError("Phase 2 subset must contain 16 unique identities")
