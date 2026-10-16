"""PySCF scientific core retained from DFT_pyscf_folder_zpve_robust_v3.py."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

HARTREE_TO_EV = 27.211386245988
TARGET_BASIS = "6-31G(2df,p)"
PREOPT_BASIS = "3-21g"


@dataclass(frozen=True)
class JobOptions:
    xc: str = "B3LYP"
    basis: str = TARGET_BASIS
    preopt_basis: str = PREOPT_BASIS
    threads: int = 8
    max_memory_mb: int = 16000
    max_cycle: int = 800
    maxsteps: int = 200
    preopt_maxsteps: int = 80
    grid_level: int = 3
    conv_tol: float = 1e-8
    robust_opt: bool = True
    allow_unconverged_gradients: bool = False
    verbose_engine: bool = False


@dataclass(frozen=True)
class SCFProfile:
    name: str
    max_cycle: int
    conv_tol: float
    level_shift: float
    damp: float
    diis_space: int
    init_guess: str
    use_newton: bool = False


@contextlib.contextmanager
def quiet_external_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def embed_3d_with_rdkit(mol: Chem.Mol, seed: int = 1) -> Chem.Mol:
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
    return mol


def ensure_3d_and_mmff(
    mol: Chem.Mol,
    seed: int = 1,
    before_mmff: Callable[[], None] | None = None,
) -> Chem.Mol:
    mol = embed_3d_with_rdkit(mol, seed=seed)
    if before_mmff is not None:
        before_mmff()
    try:
        MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return mol


def rdkit_mol_to_xyz_block(mol: Chem.Mol) -> str:
    conformer = mol.GetConformer(0)
    lines = []
    for atom in mol.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol()} {point.x:.10f} {point.y:.10f} {point.z:.10f}"
        )
    return "\n".join(lines)


def pyscf_mol_to_xyz_block(mol: Any) -> str:
    coordinates = mol.atom_coords(unit="Angstrom")
    return "\n".join(
        f"{mol.atom_symbol(index)} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
        for index, xyz in enumerate(coordinates)
    )


def choose_spin_2s(mol: Chem.Mol, charge: int) -> int:
    nelec = int(sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge)
    radicals = int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()))
    return radicals if radicals >= 0 and radicals % 2 == nelec % 2 else nelec % 2


def make_pyscf_mol(
    gto: Any,
    xyz_block: str,
    charge: int,
    spin: int,
    basis: str,
    options: JobOptions,
) -> Any:
    return gto.M(
        atom=xyz_block,
        basis=basis,
        charge=charge,
        spin=spin,
        max_memory=options.max_memory_mb,
        verbose=0 if not options.verbose_engine else 4,
    )


def make_scf_profiles(options: JobOptions) -> list[SCFProfile]:
    base_max = max(int(options.max_cycle), 300)
    profiles = [
        SCFProfile("normal", base_max, options.conv_tol, 0.0, 0.0, 12, "minao")
    ]
    if options.robust_opt:
        profiles.extend(
            [
                SCFProfile(
                    "robust_shift_0.25_damp_0.20",
                    max(base_max, 500),
                    max(options.conv_tol, 1e-8),
                    0.25,
                    0.20,
                    16,
                    "atom",
                ),
                SCFProfile(
                    "robust_shift_0.50_damp_0.30",
                    max(base_max, 800),
                    max(options.conv_tol, 1e-8),
                    0.50,
                    0.30,
                    20,
                    "atom",
                ),
            ]
        )
    return profiles


def make_mf(dft: Any, mol: Any, options: JobOptions, profile: SCFProfile) -> Any:
    mf = dft.RKS(mol) if int(getattr(mol, "spin", 0)) == 0 else dft.UKS(mol)
    mf.xc = options.xc
    mf.max_cycle = int(profile.max_cycle)
    mf.conv_tol = float(profile.conv_tol)
    mf.verbose = 0 if not options.verbose_engine else 4
    try:
        mf.grids.level = int(options.grid_level)
        mf.level_shift = float(profile.level_shift)
        mf.damp = float(profile.damp)
        mf.diis_space = int(profile.diis_space)
        mf.init_guess = str(profile.init_guess)
    except Exception:
        pass
    if profile.use_newton:
        try:
            mf = mf.newton()
        except Exception:
            pass
    return mf


def optimize_molecule_with_retries(
    dft: Any,
    geometric_solver: Any,
    mol: Any,
    options: JobOptions,
    log: TextIO,
    label: str,
    maxsteps: int,
) -> tuple[Any, str]:
    last_error: Exception | None = None
    profiles = make_scf_profiles(options)
    for profile in profiles:
        log.write(f"\n{label} geometry optimization attempt: {profile.name}\n")
        log.flush()
        mf = make_mf(dft, mol, options, profile)
        try:
            with quiet_external_output(not options.verbose_engine):
                optimized = geometric_solver.optimize(
                    mf, maxsteps=int(maxsteps), assert_convergence=True
                )
            log.write(f"Optimization status: OK with {profile.name}\n")
            log.flush()
            return optimized, profile.name
        except Exception as exc:
            last_error = exc
            log.write(
                f"Optimization status: FAILED with {profile.name}: "
                f"{type(exc).__name__}: {str(exc).splitlines()[0]}\n"
            )
            log.flush()
    if options.allow_unconverged_gradients:
        profile = profiles[-1]
        mf = make_mf(dft, mol, options, profile)
        with quiet_external_output(not options.verbose_engine):
            optimized = geometric_solver.optimize(
                mf, maxsteps=int(maxsteps), assert_convergence=False
            )
        return optimized, profile.name + "_allow_unconverged_gradients"
    assert last_error is not None
    raise last_error


def run_final_scf(
    dft: Any, mol: Any, options: JobOptions, log: TextIO
) -> tuple[Any, float, str]:
    last_error: Exception | None = None
    for profile in make_scf_profiles(options):
        log.write(f"\nFinal SCF attempt: {profile.name}\n")
        log.flush()
        mf = make_mf(dft, mol, options, profile)
        try:
            with quiet_external_output(not options.verbose_engine):
                energy = mf.kernel()
            if not bool(getattr(mf, "converged", False)):
                raise RuntimeError("Final SCF did not converge")
            return mf, float(energy), profile.name
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def compute_homo_lumo_gap(mf: Any) -> tuple[float, float, float]:
    mo_occ, mo_energy = mf.mo_occ, mf.mo_energy
    if isinstance(mo_occ, (tuple, list)) and len(mo_occ) == 2:
        occupied, virtual = [], []
        for occupations, energies in zip(mo_occ, mo_energy, strict=True):
            occupations = np.asarray(occupations)
            energies = np.asarray(energies)
            if np.any(occupations > 0):
                occupied.append(float(np.max(energies[occupations > 0])))
            if np.any(occupations == 0):
                virtual.append(float(np.min(energies[occupations == 0])))
        if not occupied or not virtual:
            raise ValueError("Could not determine UKS HOMO/LUMO from occupations.")
        homo, lumo = max(occupied), min(virtual)
        return homo, lumo, abs(lumo - homo)
    occupations, energies = np.asarray(mo_occ), np.asarray(mo_energy)
    if not np.any(occupations > 0) or not np.any(occupations == 0):
        raise ValueError("Could not determine RKS HOMO/LUMO from occupations.")
    homo = float(np.max(energies[occupations > 0]))
    lumo = float(np.min(energies[occupations == 0]))
    return homo, lumo, abs(lumo - homo)


def signed_frequency(value: Any) -> float:
    frequency = complex(value)
    return -abs(float(frequency.imag)) if abs(frequency.imag) > 1e-10 else float(frequency.real)


def preflight_pyscf_backend() -> str:
    import geometric
    import pyscf
    from pyscf import gto

    for element in ("H", "C", "N", "O", "F"):
        gto.basis.load(TARGET_BASIS, element)
    return f"PySCF {pyscf.__version__}; geomeTRIC {geometric.__version__}"


def run_pyscf_graph(
    molecule: Chem.Mol,
    output_path: Path,
    stage: Callable[[str, str], None],
) -> dict[str, Any]:
    import geometric
    import pyscf
    from pyscf import dft, gto, lib
    from pyscf.geomopt import geometric_solver
    from pyscf.hessian import thermo

    options = JobOptions()
    lib.num_threads(options.threads)
    RDLogger.DisableLog("rdApp.*")
    stage("etkdg", "ETKDGv3 fresh embedding (seed 1)")
    prepared = ensure_3d_and_mmff(
        molecule, seed=1, before_mmff=lambda: stage("mmff", "MMFF preparation")
    )
    charge = int(Chem.GetFormalCharge(prepared))
    spin = choose_spin_2s(prepared, charge)
    xyz_block = rdkit_mol_to_xyz_block(prepared)

    with output_path.open("a", encoding="utf-8") as log:
        stage("preoptimization", f"B3LYP/{PREOPT_BASIS} geomeTRIC preoptimization")
        pre_molecule = make_pyscf_mol(
            gto, xyz_block, charge, spin, options.preopt_basis, options
        )
        try:
            pre_optimized, _ = optimize_molecule_with_retries(
                dft,
                geometric_solver,
                pre_molecule,
                options,
                log,
                f"Preopt {options.xc}/{options.preopt_basis}",
                options.preopt_maxsteps,
            )
            xyz_block = pyscf_mol_to_xyz_block(pre_optimized)
        except Exception as exc:
            log.write(
                f"Preopt failed and will be ignored: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0]}\n"
            )
            log.flush()

        target_molecule = make_pyscf_mol(
            gto, xyz_block, charge, spin, options.basis, options
        )
        stage("final_optimization", f"B3LYP/{TARGET_BASIS} geomeTRIC target optimization")
        optimized, _ = optimize_molecule_with_retries(
            dft,
            geometric_solver,
            target_molecule,
            options,
            log,
            f"Target {options.xc}/{options.basis}",
            options.maxsteps,
        )
        stage("property_extraction", f"B3LYP/{TARGET_BASIS} final SCF")
        mf, _energy, _ = run_final_scf(dft, optimized, options, log)
        homo, lumo, gap = compute_homo_lumo_gap(mf)
        stage("frequency", f"B3LYP/{TARGET_BASIS} Hessian/frequency/ZPVE")
        with quiet_external_output(not options.verbose_engine):
            hessian = mf.Hessian().kernel()
            frequency_info = thermo.harmonic_analysis(optimized, hessian)
            thermo_info = thermo.thermo(mf, frequency_info["freq_au"])
        frequencies = [
            signed_frequency(value) for value in frequency_info["freq_wavenumber"]
        ]
        zpve = float(thermo_info["ZPE"][0])

    return {
        "engine_version": f"PySCF {pyscf.__version__}; geomeTRIC {geometric.__version__}",
        "homo_eV": homo * HARTREE_TO_EV,
        "lumo_eV": lumo * HARTREE_TO_EV,
        "gap_eV": gap * HARTREE_TO_EV,
        "zpve_eV": zpve * HARTREE_TO_EV,
        "frequencies_cm-1": frequencies,
        "final_coordinates": np.asarray(
            optimized.atom_coords(unit="Angstrom"), dtype=float
        ),
        "starting_molecule": prepared,
        "multiplicity": spin + 1,
    }
