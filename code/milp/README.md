# Molecular design with MILP

## Overview and entry points

This code combines structural constraints with a trained GNN to search for
molecules inside a requested property range.

- `src/mol_infer_gnn_experiments/phase2/formulation.py`: `build_common_formulation()` builds the PuLP model.
- `src/src_milp/`: structural variables, constraints, and graph assignment.
- `src/src_inverter/`: trained-network loading and GNN-to-MILP encoding.
- `src/mol_infer_gnn_experiments/phase2/solver.py`: `canonical_cplex_solver()` runs CPLEX and parses its solution.
- `scripts/run_phase2_production.py`: prepares and runs the 960-condition campaign.
- `scripts/finalize_i5_candidates.py`: reconstructs and verifies saved candidates.

## Dependencies

Use Python 3.11 or later, `PuLP==2.6.0` (see `requirements.txt`), PyTorch,
PyTorch Geometric, RDKit, NumPy, and pandas; finalization also uses NetworkX.
CPLEX 22.1.1 requires a separate installation and valid license.
Set `CPLEX_EXECUTABLE` to its absolute executable path before starting the runner.
For Python API use, add this directory's `src/` to `PYTHONPATH`.
The scripts set this import path themselves. Use a separate process from the DFT code,
which contains another package with the same `mol_infer_gnn_experiments` name.

## Inputs and working layout

Included reference inputs are `../../inputs/specifications/`,
`../../inputs/target_ranges/`, `../../inputs/phase2_conditions.jsonl`,
and the checkpoints/configurations in `../../models/`.
The I5/I5′ formulation applies the additional restrictions in `phase2/specification.py`.

The runner treats this `code/milp/` directory as its repository root. It expects:
- `data/phase1/asset-inventory.json` and `config-inventory.json`, including normalization metadata and paths to the model files;
- `data/phase2/instances/canonical/` containing instance and fringe-library files;
- `data/phase2/target_ranges/target-range-manifest.json`;
- `data/phase2/conditions/production-conditions.jsonl` and `production-manifest.json`.

Obtain the missing inventories and campaign metadata from the authors and restore
the paths they describe in a separate working copy. Renaming the public checkpoint
files alone does not supply the missing metadata. Campaign preparation also expects
`data/phase1/live-parity-result.json`, a Git checkout, and a completed CPLEX acceptance
attempt with its result and artifact manifest. Those records are not bundled.

## Basic use and outputs

For model generation only, call `build_common_formulation(repository_root=..., condition=...)`
with a condition record and the prepared layout; export its returned `problem` using
PuLP's `writeLP(...)`. This API builds the model without solving it.
For CPLEX execution, the runner's CLI has `prepare --campaign-root DIR --acceptance-attempt DIR`
and `run --campaign-root DIR`. These are interface descriptions, not a ready-to-run
recipe for this package: the external records and layout above must first be supplied.
No end-to-end command has been tested for this publication layout.

Each condition writes an attempt directory containing `formulation/model.lp`,
solver commands/logs/solution, status and verification records, and candidate artifacts.
Finalization accepts `--campaign-root DIR --output-root DIR` and consumes those records.
Saved results can instead be read from `../../results/milp_conditions.csv` and the candidate CSVs.
The single-optimization CPLEX adapter is retained; S10 interval/big-M limitations
remain as described in the top-level README.
