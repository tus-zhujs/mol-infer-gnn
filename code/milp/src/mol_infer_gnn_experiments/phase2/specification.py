"""Validated baseline and Design C bond-pattern specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import ContractError

ChemicalPattern = tuple[str, str, int]

DESIGN_C_ALLOWED_INTERIOR: frozenset[ChemicalPattern] = frozenset(
    {
        ("C4", "C4", 1),
        ("C4", "C4", 2),
        ("C4", "C4", 3),
        ("C4", "N3", 1),
        ("C4", "N3", 2),
        ("C4", "O2", 1),
        ("N3", "O2", 1),
    }
)
DESIGN_C_INTERIOR_BOUNDS: Mapping[ChemicalPattern, int] = {
    ("C4", "C4", 3): 1,
    ("C4", "N3", 2): 3,
    ("N3", "O2", 1): 1,
}
DESIGN_C_LEAF_BOUNDS: Mapping[ChemicalPattern, int] = {
    ("C4", "C4", 3): 2,
    ("C4", "N3", 2): 1,
    ("C4", "N3", 3): 2,
    ("C4", "O2", 2): 2,
}


@dataclass(frozen=True)
class StructuralSpecification:
    specification_id: str
    enabled: bool
    allowed_interior: frozenset[ChemicalPattern]
    interior_upper_bounds: Mapping[ChemicalPattern, int]
    leaf_upper_bounds: Mapping[ChemicalPattern, int]

    def validate(self) -> None:
        if not self.enabled:
            if self.allowed_interior or self.interior_upper_bounds or self.leaf_upper_bounds:
                raise ContractError("disabled bond-pattern block must contain no rules")
            return
        if self.allowed_interior != DESIGN_C_ALLOWED_INTERIOR:
            raise ContractError("guided allowed-interior set is not Design C")
        if dict(self.interior_upper_bounds) != dict(DESIGN_C_INTERIOR_BOUNDS):
            raise ContractError("guided interior bounds are not Design C")
        if dict(self.leaf_upper_bounds) != dict(DESIGN_C_LEAF_BOUNDS):
            raise ContractError("guided leaf bounds are not Design C")


BASELINE_I5 = StructuralSpecification(
    specification_id="baseline-i5-disabled-v1",
    enabled=False,
    allowed_interior=frozenset(),
    interior_upper_bounds={},
    leaf_upper_bounds={},
)
GUIDED_I5_PRIME = StructuralSpecification(
    specification_id="design-c-i5-prime-v1",
    enabled=True,
    allowed_interior=DESIGN_C_ALLOWED_INTERIOR,
    interior_upper_bounds=DESIGN_C_INTERIOR_BOUNDS,
    leaf_upper_bounds=DESIGN_C_LEAF_BOUNDS,
)


def oriented_chemical_total(
    coordinates: Mapping[ChemicalPattern, Any], pattern: ChemicalPattern
) -> Any:
    """Sum unlike orientations and include a same-label coordinate exactly once."""
    first, second, multiplicity = pattern
    values = []
    forward = (first, second, multiplicity)
    reverse = (second, first, multiplicity)
    if forward in coordinates:
        values.append(coordinates[forward])
    if first != second and reverse in coordinates:
        values.append(coordinates[reverse])
    if not values:
        raise ContractError(f"missing leaf coordinates for Design C: {pattern}")
    return sum(values[1:], values[0])


def apply_design_c_constraints(
    problem: Any,
    *,
    specification: StructuralSpecification,
    interior_totals: Mapping[ChemicalPattern, Any],
    leaf_oriented: Mapping[ChemicalPattern, Any],
) -> Any:
    specification.validate()
    if not specification.enabled:
        return problem

    def chemical_key(pattern: ChemicalPattern) -> ChemicalPattern:
        first, second, multiplicity = pattern
        return (
            (first, second, multiplicity)
            if first <= second
            else (second, first, multiplicity)
        )

    allowed = {chemical_key(pattern) for pattern in specification.allowed_interior}
    seen: set[ChemicalPattern] = set()
    for pattern, value in interior_totals.items():
        key = chemical_key(pattern)
        if key in seen:
            continue
        seen.add(key)
        if key not in allowed:
            name = f"design_c_exclude_interior_{key[0]}_{key[1]}_{key[2]}"
            problem += value == 0, name
    for pattern, upper in specification.interior_upper_bounds.items():
        key = chemical_key(pattern)
        value = interior_totals.get(key)
        if value is None:
            value = interior_totals.get((key[1], key[0], key[2]))
        if value is None:
            raise ContractError(f"missing interior coordinate for Design C: {pattern}")
        problem += (
            value <= upper,
            f"design_c_interior_{pattern[0]}_{pattern[1]}_{pattern[2]}",
        )
    for pattern, upper in specification.leaf_upper_bounds.items():
        total = oriented_chemical_total(leaf_oriented, pattern)
        problem += (
            total <= upper,
            f"design_c_leaf_{pattern[0]}_{pattern[1]}_{pattern[2]}",
        )
    return problem
