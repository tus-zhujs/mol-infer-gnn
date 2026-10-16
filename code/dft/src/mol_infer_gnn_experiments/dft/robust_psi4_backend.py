"""Psi4 scientific core retained from DFT_psi4_folder_zpve_robust_v3.py."""

from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
TARGET_LEVEL = "b3lyp/qm9_6-31g_2df_p"
PREOPT_LEVEL = "b3lyp/3-21g"
BASIS_FILENAME = "qm9_6-31g_2df_p.gbs"
BASIS_SHA256 = "6d8ae9cf42487cba7ca69fd3b99fc12667b71131fdfe06692835a67536929be6"


@dataclass(frozen=True)
class JobOptions:
    level: str = TARGET_LEVEL
    preopt_level: str = PREOPT_LEVEL
    threads: int = 6
    memory: str = "16GB"
    geom_maxiter: int = 200
    preopt_geom_maxiter: int = 80
    scf_maxiter: int = 800
    e_convergence: float = 1e-7
    d_convergence: float = 1e-7
    robust_opt: bool = True


@dataclass(frozen=True)
class Psi4Profile:
    name: str
    scf_maxiter: int
    e_convergence: float
    d_convergence: float
    level_shift: float | None = None
    damping_percentage: float | None = None
    guess: str | None = None
    soscf: bool | None = None


@contextlib.contextmanager
def quiet_external_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def ensure_3d_and_mmff(
    mol: Chem.Mol,
    seed: int = 1,
    before_mmff: Callable[[], None] | None = None,
) -> Chem.Mol:
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol, addCoords=True)
    params = ETKDGv3()
    params.randomSeed = int(seed)
    result = EmbedMolecule(mol, params)
    if result != 0:
        params.useRandomCoords = True
        result = EmbedMolecule(mol, params)
    if result != 0:
        raise ValueError("RDKit ETKDG failed to generate a 3D conformer.")
    if before_mmff is not None:
        before_mmff()
    try:
        MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return mol


def choose_multiplicity(mol: Chem.Mol, charge: int) -> tuple[int, int]:
    nelec = int(sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge)
    radicals = int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()))
    spin_2s = radicals if radicals >= 0 and radicals % 2 == nelec % 2 else nelec % 2
    return spin_2s, spin_2s + 1


def rdkit_to_psi4_geometry(mol: Chem.Mol) -> tuple[str, int, int, int]:
    charge = int(Chem.GetFormalCharge(mol))
    spin_2s, multiplicity = choose_multiplicity(mol, charge)
    conf = mol.GetConformer(0)
    lines = [f"{charge} {multiplicity}"]
    for atom in mol.GetAtoms():
        point = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol()} {point.x:.10f} {point.y:.10f} {point.z:.10f}"
        )
    return "\n".join(lines), charge, spin_2s, multiplicity


def safe_set_options(psi4: Any, options: dict[str, Any], log: TextIO) -> None:
    for key, value in options.items():
        try:
            psi4.set_options({key: value})
        except Exception as exc:
            log.write(
                f"Warning: Psi4 option {key}={value!r} ignored: {type(exc).__name__}\n"
            )


def common_options(options: JobOptions, geom_maxiter: int) -> dict[str, Any]:
    return {
        "geom_maxiter": int(geom_maxiter),
        "optking__geom_maxiter": int(geom_maxiter),
        "g_convergence": "GAU",
        "opt_coordinates": "both",
        "optking__opt_coordinates": "both",
        "maxiter": int(options.scf_maxiter),
        "scf__maxiter": int(options.scf_maxiter),
        "e_convergence": float(options.e_convergence),
        "scf__e_convergence": float(options.e_convergence),
        "d_convergence": float(options.d_convergence),
        "scf__d_convergence": float(options.d_convergence),
        "dynamic_level": 1,
        "scf__fail_on_maxiter": False,
    }


def make_profiles(options: JobOptions) -> list[Psi4Profile]:
    base_max = max(int(options.scf_maxiter), 300)
    profiles = [
        Psi4Profile(
            "normal",
            base_max,
            options.e_convergence,
            options.d_convergence,
            guess="sad",
        )
    ]
    if options.robust_opt:
        profiles.extend(
            [
                Psi4Profile(
                    "shift_0.25_damp_20",
                    max(base_max, 500),
                    max(options.e_convergence, 1e-7),
                    max(options.d_convergence, 1e-7),
                    level_shift=0.25,
                    damping_percentage=20.0,
                    guess="sad",
                ),
                Psi4Profile(
                    "shift_0.50_damp_30_soscf",
                    max(base_max, 800),
                    max(options.e_convergence, 1e-7),
                    max(options.d_convergence, 1e-7),
                    level_shift=0.50,
                    damping_percentage=30.0,
                    guess="sad",
                    soscf=True,
                ),
            ]
        )
    return profiles


def apply_profile(
    psi4: Any,
    options: JobOptions,
    profile: Psi4Profile,
    geom_maxiter: int,
    log: TextIO,
) -> None:
    values = common_options(options, geom_maxiter)
    values.update(
        {
            "maxiter": profile.scf_maxiter,
            "scf__maxiter": profile.scf_maxiter,
            "e_convergence": profile.e_convergence,
            "scf__e_convergence": profile.e_convergence,
            "d_convergence": profile.d_convergence,
            "scf__d_convergence": profile.d_convergence,
        }
    )
    if profile.guess is not None:
        values.update({"guess": profile.guess, "scf__guess": profile.guess})
    if profile.level_shift is not None:
        values.update(
            {"level_shift": profile.level_shift, "scf__level_shift": profile.level_shift}
        )
    if profile.damping_percentage is not None:
        values.update(
            {
                "damping_percentage": profile.damping_percentage,
                "scf__damping_percentage": profile.damping_percentage,
            }
        )
    if profile.soscf is not None:
        values.update({"soscf": profile.soscf, "scf__soscf": profile.soscf})
    safe_set_options(psi4, values, log)


def optimize_with_retries(
    psi4: Any,
    level: str,
    molecule: Any,
    options: JobOptions,
    log: TextIO,
    label: str,
    geom_maxiter: int,
) -> tuple[float, Any, str]:
    last_error: Exception | None = None
    for profile in make_profiles(options):
        log.write(f"\n{label} optimization attempt: {profile.name}\n")
        log.flush()
        apply_profile(psi4, options, profile, geom_maxiter, log)
        try:
            with quiet_external_output():
                energy, wfn = psi4.optimize(
                    level, molecule=molecule, return_wfn=True, engine="geometric"
                )
            log.write(f"Optimization status: OK with {profile.name}\n")
            log.flush()
            return float(energy), wfn, profile.name
        except Exception as exc:
            last_error = exc
            log.write(
                f"Optimization status: FAILED with {profile.name}: "
                f"{type(exc).__name__}: {str(exc).splitlines()[0]}\n"
            )
            log.flush()
            try:
                psi4.core.clean()
                psi4.core.clean_variables()
            except Exception:
                pass
    assert last_error is not None
    raise last_error


def final_energy_with_retries(
    psi4: Any, level: str, molecule: Any, options: JobOptions, log: TextIO
) -> tuple[float, Any, str]:
    last_error: Exception | None = None
    for profile in make_profiles(options):
        log.write(f"\nFinal single-point attempt: {profile.name}\n")
        log.flush()
        apply_profile(psi4, options, profile, options.geom_maxiter, log)
        try:
            with quiet_external_output():
                energy, wfn = psi4.energy(level, molecule=molecule, return_wfn=True)
            return float(energy), wfn, profile.name
        except Exception as exc:
            last_error = exc
            try:
                psi4.core.clean_variables()
            except Exception:
                pass
    assert last_error is not None
    raise last_error


def homo_lumo_gap_from_wfn(wfn: Any) -> tuple[float, float, float]:
    eps_a = np.asarray(wfn.epsilon_a_subset("AO", "ALL").np, dtype=float)
    na = int(wfn.nalpha())
    homos = [float(eps_a[na - 1])]
    lumos = [float(eps_a[na])]
    try:
        eps_b = np.asarray(wfn.epsilon_b_subset("AO", "ALL").np, dtype=float)
        nb = int(wfn.nbeta())
        homos.append(float(eps_b[nb - 1]))
        lumos.append(float(eps_b[nb]))
    except Exception:
        pass
    homo, lumo = max(homos), min(lumos)
    return homo, lumo, lumo - homo


def run_frequency(
    psi4: Any, level: str, molecule: Any, options: JobOptions, log: TextIO
) -> tuple[Any, str]:
    last_error: Exception | None = None
    for profile in make_profiles(options):
        log.write(f"\nFrequency attempt: {profile.name}\n")
        log.flush()
        apply_profile(psi4, options, profile, options.geom_maxiter, log)
        try:
            with quiet_external_output():
                _, wfn = psi4.frequency(level, molecule=molecule, return_wfn=True)
            return wfn, profile.name
        except Exception as exc:
            last_error = exc
            try:
                psi4.core.clean_variables()
            except Exception:
                pass
    assert last_error is not None
    raise last_error


def extract_zpve_hartree(psi4: Any, wfn: Any) -> float:
    variables = dict(wfn.variables())
    for key in ("ZPVE", "ZERO-POINT ENERGY", "ZERO POINT ENERGY"):
        if key in variables:
            return float(variables[key])
        try:
            return float(psi4.variable(key))
        except Exception:
            pass
    for key, value in variables.items():
        upper = str(key).upper()
        if "ZPVE" in upper or ("ZERO" in upper and "POINT" in upper):
            return float(value)
    raise RuntimeError("Psi4 frequency completed without a ZPVE variable")


def signed_frequency(value: Any) -> float:
    frequency = complex(value)
    return -abs(float(frequency.imag)) if abs(frequency.imag) > 1e-10 else float(frequency.real)


def basis_path(psi4: Any) -> Path:
    return Path(psi4.core.get_datadir()) / "basis" / BASIS_FILENAME


def preflight_psi4_backend() -> str:
    import psi4

    path = basis_path(psi4)
    if not path.is_file():
        raise RuntimeError(f"required old Psi4 basis file is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BASIS_SHA256:
        raise RuntimeError(f"old Psi4 basis definition hash mismatch: {path}")
    molecule = psi4.geometry(
        "0 2\n"
        "H 0 0 0\nC 3 0 0\nN 6 0 0\nO 9 0 0\nF 12 0 0\n"
        "units angstrom\nno_reorient\nno_com"
    )
    psi4.core.BasisSet.build(molecule, "BASIS", "qm9_6-31g_2df_p")
    return str(psi4.__version__)


def run_psi4_graph(
    molecule: Chem.Mol,
    output_path: Path,
    stage: Callable[[str, str], None],
) -> dict[str, Any]:
    import psi4

    options = JobOptions()
    try:
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
    except Exception:
        pass
    psi4.set_num_threads(options.threads)
    psi4.set_memory(options.memory)
    psi4.core.set_output_file(str(output_path), False)
    RDLogger.DisableLog("rdApp.*")

    stage("etkdg", "ETKDGv3 fresh embedding (seed 1)")
    prepared = ensure_3d_and_mmff(
        molecule, seed=1, before_mmff=lambda: stage("mmff", "MMFF preparation")
    )
    geometry, _charge, _spin, multiplicity = rdkit_to_psi4_geometry(prepared)
    psi_molecule = psi4.geometry(geometry)
    with output_path.open("a", encoding="utf-8") as log:
        stage("preoptimization", f"{PREOPT_LEVEL} geomeTRIC preoptimization")
        try:
            _, pre_wfn, _ = optimize_with_retries(
                psi4,
                options.preopt_level,
                psi_molecule,
                options,
                log,
                f"Preopt {options.preopt_level}",
                options.preopt_geom_maxiter,
            )
            psi_molecule = pre_wfn.molecule()
            psi_molecule.update_geometry()
        except Exception as exc:
            log.write(
                f"Preopt failed and will be ignored: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0]}\n"
            )
            log.flush()

        stage("final_optimization", f"{TARGET_LEVEL} geomeTRIC target optimization")
        _opt_energy, opt_wfn, _ = optimize_with_retries(
            psi4,
            options.level,
            psi_molecule,
            options,
            log,
            f"Target {options.level}",
            options.geom_maxiter,
        )
        optimized_molecule = opt_wfn.molecule()
        optimized_molecule.update_geometry()
        stage("property_extraction", f"{TARGET_LEVEL} final single point")
        _energy, final_wfn, _ = final_energy_with_retries(
            psi4, options.level, optimized_molecule, options, log
        )
        homo, lumo, gap = homo_lumo_gap_from_wfn(final_wfn)
        stage("frequency", f"{TARGET_LEVEL} harmonic frequency/ZPVE")
        frequency_wfn, _ = run_frequency(
            psi4, options.level, optimized_molecule, options, log
        )
        frequencies = [
            signed_frequency(value)
            for value in frequency_wfn.frequencies().to_array()
        ]
        zpve = extract_zpve_hartree(psi4, frequency_wfn)

    final_coordinates = (
        np.asarray(optimized_molecule.geometry().to_array(), dtype=float)
        * BOHR_TO_ANGSTROM
    )
    return {
        "engine_version": str(psi4.__version__),
        "homo_eV": homo * HARTREE_TO_EV,
        "lumo_eV": lumo * HARTREE_TO_EV,
        "gap_eV": gap * HARTREE_TO_EV,
        "zpve_eV": zpve * HARTREE_TO_EV,
        "frequencies_cm-1": frequencies,
        "final_coordinates": final_coordinates,
        "starting_molecule": prepared,
        "multiplicity": multiplicity,
    }
