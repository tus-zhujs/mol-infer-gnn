"""Typed failures used by the WP2 runtime foundation."""


class ContractError(ValueError):
    """A serialized object violates a canonical contract."""


class PathClosureError(ContractError):
    """A runtime path escapes the declared closure."""


class AttemptExistsError(FileExistsError):
    """An immutable attempt identifier already exists."""


class AtomicWriteError(OSError):
    """An atomic write could not preserve the requested policy."""


class ExternalToolError(RuntimeError):
    """A declared external tool failed discovery, audit, or execution."""
