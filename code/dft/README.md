# Quantum-chemical evaluation

## Overview and entry points

`scripts/run_production_dft.py` evaluates molecular SDF inputs with either Psi4
or PySCF/geomeTRIC. The shared workflow is in `src/mol_infer_gnn_experiments/dft/production.py`; `robust_psi4_backend.py` and
`robust_pyscf_backend.py` implement the respective calculation backends.
Both prepare coordinates with RDKit ETKDGv3 and MMFF, then perform B3LYP
preoptimization, final optimization, and frequency/property calculations.
The workflow checks heavy-atom connectivity and frequencies and extracts HOMO,
LUMO, gap, and ZPVE values. Backend selection does not change the input graph population.

## Dependencies

Use Python 3.11 or later, NumPy, and RDKit, plus the selected backend:
- `--engine psi4`: Psi4 (recorded version 1.10.1).
- `--engine pyscf-geometric`: PySCF 2.13.1 and geomeTRIC 1.1.1.
Separate Python environments may be used for the two engines. The script adds this directory's `src/` to its import path; do not mix it with
`code/milp/src/` in the same Python process.

## Input preparation

The 175 input structures are in `../../results/molecules/` and graph/condition relationships are in `../../results/candidate_mapping.csv`.
A DFT execution manifest is not included. Prepare a JSON object with an `entries`
array, using one entry per graph and engine. Each entry needs `engine`, `graph_id`,
`input_graph_sdf`, `source_condition_ids`, and `specification_membership`.
Use lists for the final two fields and absolute SDF paths valid on your machine;
retain the graph IDs and all condition associations from the supplied mapping.
Engine names must be exactly `psi4` or `pyscf-geometric`.

Psi4 calculations used Basis Set Exchange (BSE) `6-31G(2df,p)`, data version 0,
for H, C, N, O and F. Basis data is not included in this repository.
The setup script obtains Gaussian94 text from the API linked on the
[official BSE page](https://www.basissetexchange.org/basis/6-31g%282df%2Cp%29/format/gaussian94/?elements=1,6,7,8,9&version=0).
From `code/dft/`, activate your Psi4 environment and run:

```sh
python scripts/setup_psi4_basis.py
```

Internet access is required. The script uses only the Python standard library,
adds the original `spherical` prefix and, if absent, the `****` separator immediately
before the first H block to restore the experimental file format. It requires SHA-256
`6d8ae9cf42487cba7ca69fd3b99fc12667b71131fdfe06692835a67536929be6`.
Only matching data is installed as `qm9_6-31g_2df_p.gbs` under
`psi4.core.get_datadir()/basis/`, which must be writable. An identical existing
file is left unchanged; a different file is never overwritten.
After successful installation, use the existing Psi4 backend as described below.

No basis coefficients or headers are rewritten. If the restored file does not
match the required hash, installation stops.

PySCF uses its installed `6-31G(2df,p)` basis; both backends use `3-21g`
for preoptimization.

## Basic use

From `code/dft/`, the checked CLI syntax is shown below. These commands have not
been run here; replace the example paths after preparing inputs and dependencies.
Use a new output directory, separate from the supplied paper results.

```sh
python scripts/run_production_dft.py --manifest /path/to/manifest.json \
  --output-root /path/to/new-results --engine psi4
python scripts/run_production_dft.py --manifest /path/to/manifest.json \
  --output-root /path/to/new-results --engine pyscf-geometric
```

When both engine CSVs share an output directory, the script also creates comparison
tables. Before that stage, place a working copy of `../../results/candidate_verification.csv`
there named `i5_i5prime_candidate_recheck.csv`, as required by the comparison reader.

## Outputs

`dft_runs/<engine>/<graph_id>/` contains input metadata, starting/final SDFs,
native logs, `result.json`, and `COMPLETED.json`. Completed matching jobs are reused.
Engine tables are `dft_psi4_results.csv` and `dft_pyscf_results.csv`; failures and
structural/frequency rejections are recorded as well as accepted calculations.
The supplied `../../results/dft_psi4.csv` and `dft_pyscf.csv` can be read without running DFT.
