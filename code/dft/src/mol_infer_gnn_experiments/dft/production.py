"""Restart-safe Psi4 and PySCF+geomeTRIC workflows for final graph candidates."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from ..canonical import atomic_write_bytes, atomic_write_json, sha256_file

METHOD = "B3LYP"
PSI4_FINAL_BASIS = "qm9_6-31g_2df_p"
PYSCF_FINAL_BASIS = "6-31G(2df,p)"
RESULT_FIELDS = [
    "graph_id", "source_condition_ids", "specification_membership", "input_sdf",
    "starting_geometry", "final_geometry", "engine", "engine_version", "method",
    "basis", "homo_eV", "lumo_eV", "gap_eV", "zpve_eV",
    "optimization_status", "connectivity_preserved", "frequencies_cm-1",
    "imaginary_frequency_count_below_minus20", "soft_negative_frequency_count",
    "most_negative_frequency_cm-1", "final_dft_status", "runtime_seconds",
    "native_input_path", "native_output_path", "error",
]


def _elapsed_text(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"


class _ProgressReporter:
    """Small console reporter; it never scans campaign artifacts."""

    def __init__(
        self,
        engine: str,
        index: int,
        total: int,
        entry: dict[str, Any],
        *,
        heartbeat_seconds: float = 60.0,
        stream: TextIO | None = None,
    ) -> None:
        self.engine = engine
        self.engine_label = "Psi4" if engine == "psi4" else "PySCF+geomeTRIC"
        self.index = index
        self.total = total
        self.entry = entry
        self.heartbeat_seconds = heartbeat_seconds
        self.stream = stream or sys.stdout
        self.started = time.monotonic()
        self.stage_name = "initializing"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _print(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def start(self) -> None:
        specifications = " / ".join(self.entry["specification_membership"])
        self._print("-" * 60)
        self._print(f"[START] {self.engine_label} {self.index} / {self.total}")
        self._print(f"Graph: {self.entry['graph_id']}")
        self._print(f"Sources: {len(self.entry['source_condition_ids'])}")
        self._print(f"Specifications: {specifications}")
        self._print(f"Started: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        self._print("-" * 60)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"dft-heartbeat-{self.entry['graph_id']}",
            daemon=True,
        )
        self._thread.start()

    def stage(self, _stage_key: str, label: str) -> None:
        self.stage_name = label
        self._print(f"[{self.index}/{self.total}] {label}")

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            elapsed = _elapsed_text(time.monotonic() - self.started)
            self._print(
                f"[RUNNING] {self.engine_label} {self.index}/{self.total} | "
                f"{self.stage_name} | elapsed {elapsed}"
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"), overwrite=True)


def _load_sdf(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecule = supplier[0] if len(supplier) else None
    if molecule is None:
        raise RuntimeError(f"RDKit rejected SDF: {path}")
    if molecule.GetNumConformers() != 1:
        raise RuntimeError(f"SDF requires exactly one conformer: {path}")
    return molecule


def _charge_multiplicity(molecule: Chem.Mol) -> tuple[int, int]:
    charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    electrons = sum(atom.GetAtomicNum() for atom in molecule.GetAtoms()) - charge
    return charge, 1 if electrons % 2 == 0 else 2


def _symbols_coordinates(molecule: Chem.Mol) -> tuple[list[str], np.ndarray]:
    conformer = molecule.GetConformer()
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    coordinates = np.asarray(
        [list(conformer.GetAtomPosition(index)) for index in range(molecule.GetNumAtoms())],
        dtype=float,
    )
    return symbols, coordinates


def _write_geometry_sdf(template: Chem.Mol, coordinates: np.ndarray, path: Path) -> None:
    molecule = Chem.Mol(template)
    conformer = molecule.GetConformer()
    if coordinates.shape != (molecule.GetNumAtoms(), 3):
        raise RuntimeError("optimized coordinate shape does not match input graph")
    for index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(index, tuple(float(value) for value in xyz))
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(molecule)
    finally:
        writer.close()


def _heavy_edges(molecule: Chem.Mol) -> set[tuple[int, int]]:
    return {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in molecule.GetBonds()
        if molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() > 1
        and molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() > 1
    }


def _connectivity_preserved(template: Chem.Mol, coordinates: np.ndarray) -> bool:
    probe = Chem.RWMol()
    for atom in template.GetAtoms():
        copied = Chem.Atom(atom.GetAtomicNum())
        copied.SetFormalCharge(atom.GetFormalCharge())
        probe.AddAtom(copied)
    conformer = Chem.Conformer(template.GetNumAtoms())
    for index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(index, tuple(float(value) for value in xyz))
    probe.AddConformer(conformer)
    charge, _multiplicity = _charge_multiplicity(template)
    rdDetermineBonds.DetermineConnectivity(probe, charge=charge)
    return _heavy_edges(template) == _heavy_edges(probe)


def _frequency_diagnostics(frequencies: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in frequencies if math.isfinite(float(value))]
    severe = sum(value < -20.0 for value in finite)
    soft = sum(-20.0 <= value < 0.0 for value in finite)
    return {
        "frequencies_cm-1": finite,
        "imaginary_frequency_count_below_minus20": severe,
        "soft_negative_frequency_count": soft,
        "most_negative_frequency_cm-1": min(finite) if finite else None,
    }


def _psi4_workflow(
    molecule: Chem.Mol,
    output_path: Path,
    stage: Callable[[str, str], None],
) -> dict[str, Any]:
    import psi4

    symbols, coordinates = _symbols_coordinates(molecule)
    charge, multiplicity = _charge_multiplicity(molecule)
    geometry = [f"{charge} {multiplicity}", "units angstrom", "no_reorient", "no_com"]
    geometry.extend(
        f"{symbol} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"
        for symbol, xyz in zip(symbols, coordinates, strict=True)
    )
    psi4.core.set_output_file(str(output_path), False)
    psi4.set_options({"reference": "rks" if multiplicity == 1 else "uks"})
    psi_molecule = psi4.geometry("\n".join(geometry))
    stage("preoptimization", "B3LYP/3-21G preoptimization")
    psi4.optimize(f"{METHOD}/{PREOPT_BASIS}", molecule=psi_molecule)
    stage("final_optimization", "B3LYP/6-31G(2df,p) final optimization")
    psi4.optimize(f"{METHOD}/{FINAL_BASIS}", molecule=psi_molecule)
    stage("frequency", "harmonic frequency calculation")
    _energy, wavefunction = psi4.frequency(
        f"{METHOD}/{FINAL_BASIS}", molecule=psi_molecule, return_wfn=True
    )
    stage("property_extraction", "extracting HOMO/LUMO/Gap/ZPVE")
    orbital = np.asarray(wavefunction.epsilon_a().to_array(), dtype=float)
    homo_index = int(wavefunction.nalpha()) - 1
    homo = float(orbital[homo_index] * HARTREE_TO_EV)
    lumo = float(orbital[homo_index + 1] * HARTREE_TO_EV)
    frequencies = [
        _signed_frequency(value)
        for value in wavefunction.frequencies().to_array()
    ]
    final_coordinates = (
        np.asarray(psi_molecule.geometry().to_array(), dtype=float)
        * BOHR_TO_ANGSTROM
    )
    return {
        "engine_version": psi4.__version__,
        "homo_eV": homo,
        "lumo_eV": lumo,
        "gap_eV": lumo - homo,
        "final_coordinates": final_coordinates,
        **_frequency_record(frequencies),
    }


def _pyscf_workflow(
    molecule: Chem.Mol,
    output_path: Path,
    stage: Callable[[str, str], None],
) -> dict[str, Any]:
    import geometric
    import pyscf
    from pyscf import dft, gto
    from pyscf.geomopt.geometric_solver import optimize
    from pyscf.hessian import thermo

    symbols, coordinates = _symbols_coordinates(molecule)
    charge, multiplicity = _charge_multiplicity(molecule)

    def build(symbols_value: list[str], coords: np.ndarray, basis: str) -> Any:
        return gto.M(
            atom=list(zip(symbols_value, coords.tolist(), strict=True)),
            basis=basis,
            charge=charge,
            spin=multiplicity - 1,
            unit="Angstrom",
            verbose=4,
        )

    pre_mol = build(symbols, coordinates, PREOPT_BASIS)
    pre_mf = (dft.RKS(pre_mol) if multiplicity == 1 else dft.UKS(pre_mol))
    pre_mf.xc = "b3lyp"
    stage("preoptimization", "B3LYP/3-21G preoptimization")
    pre_optimized = optimize(pre_mf)
    final_start = build(
        symbols,
        np.asarray(pre_optimized.atom_coords(unit="Angstrom"), dtype=float),
        FINAL_BASIS,
    )
    final_mf = dft.RKS(final_start) if multiplicity == 1 else dft.UKS(final_start)
    final_mf.xc = "b3lyp"
    stage("final_optimization", "B3LYP/6-31G(2df,p) final optimization")
    final_optimized = optimize(final_mf)
    final_mf = dft.RKS(final_optimized) if multiplicity == 1 else dft.UKS(final_optimized)
    final_mf.xc = "b3lyp"
    final_mf.kernel()
    if not final_mf.converged:
        raise RuntimeError("final PySCF SCF did not converge")
    stage("frequency", "harmonic frequency calculation")
    hessian = final_mf.Hessian().kernel()
    analysis = thermo.harmonic_analysis(final_optimized, hessian)
    frequencies = [
        _signed_frequency(value) for value in analysis["freq_wavenumber"]
    ]
    stage("property_extraction", "extracting HOMO/LUMO/Gap/ZPVE")
    orbital = final_mf.mo_energy
    if isinstance(orbital, tuple):
        occupied = sorted(
            float(energy)
            for energies, occupations in zip(final_mf.mo_energy, final_mf.mo_occ, strict=True)
            for energy, occupation in zip(energies, occupations, strict=True)
            if occupation > 0
        )
        unoccupied = sorted(
            float(energy)
            for energies, occupations in zip(final_mf.mo_energy, final_mf.mo_occ, strict=True)
            for energy, occupation in zip(energies, occupations, strict=True)
            if occupation == 0
        )
        homo_h, lumo_h = occupied[-1], unoccupied[0]
    else:
        occupied = np.asarray(final_mf.mo_occ) > 0
        homo_h = float(np.max(np.asarray(orbital)[occupied]))
        lumo_h = float(np.min(np.asarray(orbital)[~occupied]))
    homo = homo_h * HARTREE_TO_EV
    lumo = lumo_h * HARTREE_TO_EV
    return {
        "engine_version": f"PySCF {pyscf.__version__}; geomeTRIC {geometric.__version__}",
        "homo_eV": homo,
        "lumo_eV": lumo,
        "gap_eV": lumo - homo,
        "final_coordinates": np.asarray(
            final_optimized.atom_coords(unit="Angstrom"), dtype=float
        ),
        **_frequency_record(frequencies),
    }


def _completed_result(
    entry: dict[str, Any], output_root: Path
) -> dict[str, Any] | None:
    run_root = output_root / "dft_runs" / entry["engine"] / entry["graph_id"]
    completion = run_root / "COMPLETED.json"
    result_path = run_root / "result.json"
    if not completion.is_file() or not result_path.is_file():
        return None
    try:
        record = _read_json(completion)
        result = _read_json(result_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if (
        record.get("graph_id") != entry["graph_id"]
        or record.get("engine") != entry["engine"]
        or record.get("result_sha256") != sha256_file(result_path)
        or result.get("graph_id") != entry["graph_id"]
        or result.get("engine") != entry["engine"]
        or not result.get("final_dft_status")
    ):
        return None
    return result


def _is_shared_infrastructure_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(
        marker in text
        for marker in (
            "BasisSetNotFound",
            "required old Psi4 basis file is missing",
            "old Psi4 basis definition hash mismatch",
            "No module named 'psi4'",
            'No module named "psi4"',
        )
    )


def _run_entry(
    entry: dict[str, Any],
    output_root: Path,
    stage_callback: Callable[[str, str], None] | None = None,
    engine_version: str | None = None,
) -> dict[str, Any]:
    engine = entry["engine"]
    graph_id = entry["graph_id"]
    run_root = output_root / "dft_runs" / engine / graph_id
    completion = run_root / "COMPLETED.json"
    result_path = run_root / "result.json"
    prior_result = _completed_result(entry, output_root)
    if prior_result is not None:
        return prior_result
    started = time.monotonic()
    current_stage = "input_preparation"

    def stage(stage_key: str, label: str) -> None:
        nonlocal current_stage
        current_stage = stage_key
        if stage_callback is not None:
            stage_callback(stage_key, label)

    run_root.mkdir(parents=True, exist_ok=True)
    native_input = run_root / "native-input.json"
    native_output = run_root / ("psi4.out" if engine == "psi4" else "pyscf.out")
    starting = run_root / "starting-geometry.sdf"
    final_geometry = run_root / "final-geometry.sdf"
    result: dict[str, Any] = {
        "graph_id": graph_id,
        "source_condition_ids": ";".join(entry["source_condition_ids"]),
        "specification_membership": ";".join(entry["specification_membership"]),
        "input_sdf": entry["input_graph_sdf"],
        "starting_geometry": str(starting.resolve()),
        "final_geometry": str(final_geometry.resolve()),
        "engine": engine,
        "engine_version": engine_version,
        "method": METHOD,
        "basis": PSI4_FINAL_BASIS if engine == "psi4" else PYSCF_FINAL_BASIS,
        "homo_eV": None,
        "lumo_eV": None,
        "gap_eV": None,
        "zpve_eV": None,
        "optimization_status": "FAILED",
        "connectivity_preserved": False,
        "frequencies_cm-1": [],
        "imaginary_frequency_count_below_minus20": None,
        "soft_negative_frequency_count": None,
        "most_negative_frequency_cm-1": None,
        "final_dft_status": "DFT_FAILED",
        "runtime_seconds": None,
        "native_input_path": str(native_input.resolve()),
        "native_output_path": str(native_output.resolve()),
        "error": None,
    }
    try:
        if not native_input.exists():
            atomic_write_json(native_input, entry)
        stage("input_preparation", "loading manifest input graph")
        molecule = _load_sdf(Path(entry["input_graph_sdf"]))
        previous_directory = Path.cwd()
        try:
            os.chdir(run_root)
            if engine == "psi4":
                from .robust_psi4_backend import run_psi4_graph

                computed = run_psi4_graph(molecule, native_output, stage)
            else:
                from .robust_pyscf_backend import run_pyscf_graph

                computed = run_pyscf_graph(molecule, native_output, stage)
        finally:
            os.chdir(previous_directory)
        starting_molecule = computed.pop("starting_molecule")
        starting_coordinates = np.asarray(
            [
                list(starting_molecule.GetConformer().GetAtomPosition(index))
                for index in range(starting_molecule.GetNumAtoms())
            ],
            dtype=float,
        )
        _write_geometry_sdf(starting_molecule, starting_coordinates, starting)
        coordinates = computed.pop("final_coordinates")
        computed.pop("multiplicity", None)
        computed.update(_frequency_diagnostics(computed["frequencies_cm-1"]))
        _write_geometry_sdf(starting_molecule, coordinates, final_geometry)
        stage("connectivity", "checking optimized heavy-atom connectivity")
        preserved = _connectivity_preserved(starting_molecule, coordinates)
        severe = computed["imaginary_frequency_count_below_minus20"]
        result.update(computed)
        result["optimization_status"] = "COMPLETED"
        result["connectivity_preserved"] = preserved
        if not preserved:
            result["final_dft_status"] = "DFT_REJECTED_CONNECTIVITY"
        elif severe:
            result["final_dft_status"] = "DFT_REJECTED_IMAGINARY_FREQUENCY"
        else:
            result["final_dft_status"] = "DFT_ACCEPTED"
    except Exception as exc:
        if _is_shared_infrastructure_error(exc):
            raise RuntimeError(
                f"shared DFT infrastructure failure; campaign stopped: {exc}"
            ) from exc
        failure_status = {
            "etkdg": "ETKDG_FAILED",
            "mmff": "MMFF_FAILED",
            "input_preparation": "INPUT_PREPARATION_FAILED",
            "preoptimization": "PREOPTIMIZATION_FAILED",
            "final_optimization": "FINAL_OPTIMIZATION_FAILED",
            "frequency": "FREQUENCY_FAILED",
            "property_extraction": "PROPERTY_EXTRACTION_FAILED",
            "connectivity": "CONNECTIVITY_CHECK_FAILED",
        }.get(current_stage, "ENGINE_ERROR")
        result["final_dft_status"] = failure_status
        if current_stage in {"frequency", "property_extraction", "connectivity"}:
            result["optimization_status"] = "COMPLETED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        with native_output.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
    result["runtime_seconds"] = time.monotonic() - started
    atomic_write_json(result_path, result, overwrite=result_path.exists())
    atomic_write_json(
        completion,
        {"graph_id": graph_id, "engine": engine, "result_sha256": sha256_file(result_path)},
        overwrite=completion.exists(),
    )
    return result


def _latex_table(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    lines = ["\\begin{tabular}{" + "l" * len(headers) + "}", "\\hline", " & ".join(headers) + " \\\\", "\\hline"]
    lines.extend(" & ".join(str(value) for value in row) + " \\\\" for row in rows)
    lines.extend(["\\hline", "\\end{tabular}", ""])
    atomic_write_bytes(path, "\n".join(lines).encode("utf-8"), overwrite=True)


def _write_combined_and_analysis(output_root: Path) -> None:
    psi_path = output_root / "dft_psi4_results.csv"
    pyscf_path = output_root / "dft_pyscf_results.csv"
    if not psi_path.is_file() or not pyscf_path.is_file():
        return
    psi_rows = list(csv.DictReader(psi_path.open(encoding="utf-8")))
    pyscf_rows = list(csv.DictReader(pyscf_path.open(encoding="utf-8")))
    combined = psi_rows + pyscf_rows
    _write_csv(output_root / "dft_combined_results.csv", combined, RESULT_FIELDS)
    by_key = {(row["graph_id"], row["engine"]): row for row in combined}
    candidates = list(
        csv.DictReader((output_root / "i5_i5prime_candidate_recheck.csv").open(encoding="utf-8"))
    )
    comparison = []
    property_column = {"homo": "homo_eV", "lumo": "lumo_eV", "gap": "gap_eV", "zpve": "zpve_eV"}
    for candidate in candidates:
        if candidate["final_scientific_status"] != "ACCEPTED_CANDIDATE":
            continue
        for engine in ("psi4", "pyscf-geometric"):
            result = by_key.get((candidate["graph_id"], engine))
            if not result or result["final_dft_status"] != "DFT_ACCEPTED":
                continue
            prop = candidate["property"]
            dft_value = float(result[property_column[prop]])
            prediction = float(candidate["direct_physical_prediction"])
            lower, upper = float(candidate["target_lower"]), float(candidate["target_upper"])
            distance = 0.0 if lower <= dft_value <= upper else min(abs(dft_value - lower), abs(dft_value - upper))
            comparison.append(
                {
                    "condition_id": candidate["condition_id"], "graph_id": candidate["graph_id"],
                    "specification": candidate["specification"], "property": prop,
                    "architecture": candidate["architecture"], "target_range_index": candidate["target_range_index"],
                    "engine": engine, "prediction": prediction, "dft_value": dft_value,
                    "absolute_prediction_dft_discrepancy": abs(prediction - dft_value),
                    "target_lower": lower, "target_upper": upper, "target_range_distance": distance,
                    "dft_target_hit": lower <= dft_value <= upper,
                }
            )
    comparison_fields = list(comparison[0]) if comparison else ["condition_id"]
    _write_csv(output_root / "dft_prediction_comparison.csv", comparison, comparison_fields)
    summary = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison:
        grouped[(row["engine"], row["property"])].append(row)
    for (engine, prop), values in sorted(grouped.items()):
        summary.append(
            {
                "engine": engine, "property": prop, "count": len(values),
                "mean_absolute_prediction_dft_discrepancy": sum(float(row["absolute_prediction_dft_discrepancy"]) for row in values) / len(values),
                "target_hit_count": sum(str(row["dft_target_hit"]).lower() == "true" for row in values),
                "mean_target_range_distance": sum(float(row["target_range_distance"]) for row in values) / len(values),
            }
        )
    _write_csv(output_root / "dft_summary_by_property.csv", summary, list(summary[0]) if summary else ["engine"])
    agreement = []
    for graph_id in sorted({row["graph_id"] for row in combined}):
        left, right = by_key.get((graph_id, "psi4")), by_key.get((graph_id, "pyscf-geometric"))
        if not left or not right or left["final_dft_status"] != "DFT_ACCEPTED" or right["final_dft_status"] != "DFT_ACCEPTED":
            continue
        agreement.append({"graph_id": graph_id, **{f"absolute_{key}_agreement_eV": abs(float(left[key]) - float(right[key])) for key in ("homo_eV", "lumo_eV", "gap_eV", "zpve_eV")}})
    _write_csv(output_root / "dft_engine_agreement.csv", agreement, list(agreement[0]) if agreement else ["graph_id"])
    spec_summary = []
    spec_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison:
        spec_groups[(row["engine"], row["property"], row["specification"])].append(row)
    for key, values in sorted(spec_groups.items()):
        spec_summary.append({"engine": key[0], "property": key[1], "specification": key[2], "count": len(values), "mean_absolute_discrepancy": sum(float(row["absolute_prediction_dft_discrepancy"]) for row in values) / len(values), "target_hit_count": sum(str(row["dft_target_hit"]).lower() == "true" for row in values)})
    _write_csv(output_root / "dft_i5_vs_i5prime.csv", spec_summary, list(spec_summary[0]) if spec_summary else ["engine"])
    matched = []
    match_groups: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in comparison:
        match_groups[(row["engine"], row["property"], row["architecture"], row["target_range_index"])][row["specification"]] = row
    for key, values in sorted(match_groups.items()):
        if set(values) == {"I5", "I5-prime"}:
            matched.append({"engine": key[0], "property": key[1], "architecture": key[2], "target_range_index": key[3], "i5_condition_id": values["I5"]["condition_id"], "i5prime_condition_id": values["I5-prime"]["condition_id"], "i5_absolute_discrepancy": values["I5"]["absolute_prediction_dft_discrepancy"], "i5prime_absolute_discrepancy": values["I5-prime"]["absolute_prediction_dft_discrepancy"]})
    _write_csv(output_root / "dft_matched_i5_i5prime.csv", matched, list(matched[0]) if matched else ["engine"])
    acceptance = []
    for engine in ("psi4", "pyscf-geometric"):
        values = [row for row in combined if row["engine"] == engine]
        counts = Counter(row["final_dft_status"] for row in values)
        acceptance.append({"engine": engine, "total": len(values), "connectivity_preserved": sum(str(row["connectivity_preserved"]).lower() == "true" for row in values), "no_frequency_below_minus20": sum(row["imaginary_frequency_count_below_minus20"] == "0" for row in values), **counts})
    acceptance_fields = sorted({key for row in acceptance for key in row})
    _write_csv(output_root / "dft_connectivity_frequency_summary.csv", acceptance, acceptance_fields)
    _latex_table(output_root / "dft_summary_by_property.tex", list(summary[0]) if summary else ["engine"], [list(row.values()) for row in summary])
    _latex_table(output_root / "dft_i5_vs_i5prime.tex", list(spec_summary[0]) if spec_summary else ["engine"], [list(row.values()) for row in spec_summary])
    _latex_table(output_root / "dft_connectivity_frequency_summary.tex", acceptance_fields, [[row.get(key, 0) for key in acceptance_fields] for row in acceptance])


def run_engine(manifest_path: Path, output_root: Path, engine: str) -> None:
    if engine not in {"psi4", "pyscf-geometric"}:
        raise ValueError(f"unsupported DFT engine: {engine}")
    manifest = _read_json(manifest_path.resolve(strict=True))
    entries = [entry for entry in manifest["entries"] if entry["engine"] == engine]
    if engine == "psi4":
        from .robust_psi4_backend import preflight_psi4_backend

        engine_version = preflight_psi4_backend()
    else:
        from .robust_pyscf_backend import preflight_pyscf_backend

        engine_version = preflight_pyscf_backend()
    rows = []
    for index, entry in enumerate(entries, start=1):
        existing = _completed_result(entry, output_root)
        if existing is not None:
            result = existing
            print(
                f"[SKIP] {engine} {index}/{len(entries)} | "
                f"{entry['graph_id']} | {result['final_dft_status']}",
                flush=True,
            )
        else:
            reporter = _ProgressReporter(engine, index, len(entries), entry)
            reporter.start()
            try:
                result = _run_entry(
                    entry,
                    output_root,
                    reporter.stage,
                    engine_version,
                )
            finally:
                reporter.stop()
        rows.append(result)
        runtime = float(result.get("runtime_seconds") or 0.0)
        print(
            f"[FINISH] {engine} {index}/{len(entries)} | {entry['graph_id']} | "
            f"{result['final_dft_status']} | runtime {runtime:.1f} s",
            flush=True,
        )
        counts = Counter(row["final_dft_status"] for row in rows)
        accepted_count = counts["DFT_ACCEPTED"]
        print(
            f"Progress: {len(rows)} completed | {len(entries) - len(rows)} remaining",
            flush=True,
        )
        print(
            f"Accepted: {accepted_count} | Failed/rejected: "
            f"{len(rows) - accepted_count}",
            flush=True,
        )
        if result["final_dft_status"] != "DFT_ACCEPTED" and index < len(entries):
            print(f"Continuing to graph {index + 1} / {len(entries)}", flush=True)
        filename = "dft_psi4_results.csv" if engine == "psi4" else "dft_pyscf_results.csv"
        _write_csv(output_root / filename, rows, RESULT_FIELDS)
    _write_combined_and_analysis(output_root)
