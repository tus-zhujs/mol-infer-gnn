"""Repository-contained runtime path policy."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .errors import PathClosureError

_PERSONAL_ABSOLUTE = re.compile(
    r"^(?:/" r"Users/[^/]+|/" r"home/[^/]+)(?:/|$)"
)
_PERSONAL_TOKENS = ("/" + "Users/", "/" + "home/")
_SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml"}


def _within(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _validate_relative_text(raw: str) -> PurePosixPath:
    if not raw or "\x00" in raw:
        raise PathClosureError("path must be nonempty and contain no NUL")
    if raw.startswith("~"):
        raise PathClosureError(f"home-relative path is forbidden: {raw}")
    reject_personal_absolute_path(raw)
    if "\\" in raw:
        raise PathClosureError(f"noncanonical path separator is forbidden: {raw}")
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise PathClosureError(f"absolute runtime asset path is forbidden: {raw}")
    if ".." in path.parts:
        raise PathClosureError(f"parent-directory reference is forbidden: {raw}")
    if "." in path.parts:
        raise PathClosureError(f"noncanonical dot component is forbidden: {raw}")
    return path


@dataclass(frozen=True)
class RuntimePathPolicy:
    repository_root: Path
    output_roots: tuple[Path, ...] = ()
    declared_assets: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_root", self.repository_root.resolve())
        object.__setattr__(
            self, "output_roots", tuple(path.resolve() for path in self.output_roots)
        )
        for asset in self.declared_assets:
            _validate_relative_text(asset)

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return (self.repository_root, *self.output_roots)

    def resolve_asset(
        self,
        raw: str,
        *,
        must_exist: bool = True,
        require_declared: bool = True,
    ) -> Path:
        relative = _validate_relative_text(raw)
        normalized = relative.as_posix()
        if require_declared and normalized not in self.declared_assets:
            raise PathClosureError(f"undeclared runtime asset: {normalized}")
        lexical = self.repository_root.joinpath(*relative.parts)
        resolved = lexical.resolve(strict=must_exist)
        if not _within(resolved, self.allowed_roots):
            raise PathClosureError(
                f"symlink or path escape outside declared roots: {raw}"
            )
        return resolved

    def resolve_output(self, root: Path, raw: str) -> Path:
        root = root.resolve()
        if root not in self.output_roots:
            raise PathClosureError(f"undeclared output root: {root}")
        relative = _validate_relative_text(raw)
        resolved = root.joinpath(*relative.parts).resolve(strict=False)
        if not _within(resolved, (root,)):
            raise PathClosureError(f"output path escapes root: {raw}")
        return resolved

    def resolve_external_executable(
        self,
        raw: str,
        *,
        declared: bool,
        must_exist: bool = True,
    ) -> Path:
        """Resolve an explicit external executable without widening asset closure."""
        if not declared:
            raise PathClosureError(
                f"undeclared absolute external executable path: {raw}"
            )
        reject_personal_absolute_path(raw)
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise PathClosureError(
                f"external executable declaration requires an absolute path: {raw}"
            )
        resolved = candidate.resolve(strict=must_exist)
        if must_exist and not resolved.is_file():
            raise PathClosureError(f"external executable is not a file: {raw}")
        return resolved


def reject_personal_absolute_path(value: str) -> None:
    if _PERSONAL_ABSOLUTE.match(value):
        raise PathClosureError(f"personal absolute path is forbidden: {value}")


def audit_source_tree(root: Path) -> list[str]:
    """Return runtime-source closure violations without following symlinked files."""
    root = root.resolve()
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            resolved = path.resolve(strict=False)
            if not _within(resolved, (root,)):
                violations.append(
                    f"{path.relative_to(root)}: symlink escape -> {resolved}"
                )
            continue
        if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(token in line for token in _PERSONAL_TOKENS):
                violations.append(f"{relative}:{line_number}: personal absolute path")
            if re.search(r"(?<![A-Za-z0-9.])\.\./", line):
                violations.append(f"{relative}:{line_number}: parent reference")
            if "PYTHONPATH" in line and path.suffix != ".py":
                violations.append(f"{relative}:{line_number}: PYTHONPATH dependency")
    return violations


def imports_from_pythonpath() -> bool:
    return bool(os.environ.get("PYTHONPATH"))
