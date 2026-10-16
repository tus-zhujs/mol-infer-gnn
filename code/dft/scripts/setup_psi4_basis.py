#!/usr/bin/env python3
"""Fetch the BSE basis and install only the exact bytes used in the paper.

Uses only Python's standard library. An installed Psi4 environment is needed
to discover the default destination; no quantum-chemical calculations are run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.error import URLError
from urllib.request import urlopen


# Text API linked from the official BSE page, with the same selection.
BSE_URL = (
    "https://www.basissetexchange.org/api/basis/6-31g%282df%2Cp%29/"
    "format/gaussian94/?elements=1,6,7,8,9&version=0"
)
BASIS_FILENAME = "qm9_6-31g_2df_p.gbs"
EXPECTED_SHA256 = "6d8ae9cf42487cba7ca69fd3b99fc12667b71131fdfe06692835a67536929be6"


def verify_basis(data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            "Basis SHA-256 mismatch; nothing was installed.\n"
            f"Expected: {EXPECTED_SHA256}\nReceived: {digest}\n"
            "The BSE response with the spherical prefix and initial separator "
            "must match exactly. Do not change the expected hash or bypass this check."
        )


def download_basis() -> bytes:
    with urlopen(BSE_URL, timeout=60) as response:
        body = response.read()
    # Restore only the initial separator used in the experimental file.
    # Keep all existing bytes, including the header and basis coefficients.
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.split() == [b"H", b"0"]:
            previous = next((item.strip() for item in reversed(lines[:index]) if item.strip()), b"")
            if previous != b"****":
                lines.insert(index, b"****\n")
            break
    body = b"".join(lines)
    data = b"spherical\n\n" + body
    verify_basis(data)
    return data


def psi4_data_dir() -> Path:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import psi4; print(psi4.core.get_datadir())"],
            check=True, capture_output=True, text=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Cannot locate Psi4 data. Activate your Psi4 Python environment "
            "or specify --psi4-data-dir with its get_datadir() directory."
        ) from exc
    path = result.stdout.strip()
    if not path or not Path(path).is_dir():
        raise RuntimeError(f"Invalid Psi4 data directory: {path!r}")
    return Path(path)


def install_basis(data: bytes, data_dir: Path) -> tuple[Path, bool]:
    """Install atomically without replacing any existing file."""
    verify_basis(data)
    directory = data_dir / "basis"
    if not directory.is_dir():
        raise RuntimeError(f"Psi4 basis directory does not exist: {directory}")
    destination = directory / BASIS_FILENAME

    def existing_matches() -> bool:
        if destination.read_bytes() != data:
            raise FileExistsError(
                f"Different basis file already exists; refusing to overwrite: {destination}"
            )
        return True

    if destination.exists():
        existing_matches()
        return destination, False
    # Link a fully written temporary file into place. Unlike replace(), link()
    # fails if another process creates the destination in the meantime.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".qm9-basis-", delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing_matches()
            return destination, False
    finally:
        if temporary is not None:
            temporary.unlink()
    return destination, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--psi4-data-dir", type=Path,
        help="Psi4 get_datadir() path (its basis/ subdirectory must exist)",
    )
    args = parser.parse_args(argv)
    try:
        directory = args.psi4_data_dir if args.psi4_data_dir is not None else psi4_data_dir()
        destination = directory / "basis" / BASIS_FILENAME
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != EXPECTED_SHA256:
                raise FileExistsError(
                    f"Existing basis has a different SHA-256; refusing to overwrite: {destination}"
                )
            print(f"Already identical; unchanged: {destination}")
            return 0
        data = download_basis()
        destination, installed = install_basis(data, directory)
    except (OSError, ValueError, RuntimeError, URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{'Installed' if installed else 'Already identical; unchanged'}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
