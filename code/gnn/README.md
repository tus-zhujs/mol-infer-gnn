# GNN training and evaluation

## Overview and entry points

The two-level GNN predicts QM9 properties from molecular graphs and rho-fringe trees.
`src/network.py` defines the networks; `src/chemicalgraph.py` and `src/dataset.py`
build features; `src/GNN.py` implements tuning, training, and evaluation.

- `GNN_main_qm9.py`: tune hyperparameters, train, and evaluate one configuration.
- `evaluate_journal_remaining_qm9_checkpoints.py`: evaluate saved models on a common split.
- `evaluate_old_new_qm9_checkpoints.py`: compare two model families; requires extra models.

## Dependencies and inputs

Training imports PyTorch, torchvision, PyTorch Geometric, torch-sparse, RDKit,
NumPy, pandas, SciPy, scikit-learn, and Optuna. Use compatible PyTorch extension builds.
Keep `src/` and `common_eval/` beside the scripts so local imports resolve.

Obtain the prepared QM9 files from the authors, or prepare equivalent files from QM9:
`qm9_all.sdf` with matching molecule IDs, `qm9_complete_values.csv` with molecule IDs
in its first column and named property columns, and `qm9_uncharacterized.csv`
with an `id` column of excluded QM9 indices. Preserve molecule order and property units;
the reader applies the conversions defined in the training script.
Place these files in this directory. They are not included in the package.
The supplied `../../inputs/qm9_split_seed42.csv` records the reference split.
The scripts recreate the split from ordered, filtered molecules; they do not load that CSV.

## Training

Run from `code/gnn/` in a writable working copy; create `model/` and `log/` first.
The following illustrates the checked CLI syntax, but has not been executed here
and requires the prepared data and dependencies above:

```sh
python GNN_main_qm9.py qm9_all.sdf qm9_complete_values.csv \
  --val homo --layer_num 2 --hidden_node 16 --output_prefix homo_example
```

The script uses seed 42, 110,000 training molecules, and 10,000 validation molecules;
the remainder forms the test set. Training includes an Optuna search.
`--resume CHECKPOINT` passes an existing checkpoint to the final training stage.
Outputs include `model/<prefix>.pth`, `model/<prefix>_config.txt`, training logs
in `log/`, temporary result files, and graph-dataset files under `data/`.

## Evaluating checkpoints

The evaluator expects all 36 model/config pairs in `model/`, using names such as
`3_homo_2lay_h16_nodropout.pth` and its matching `_config.txt` file.
The public `../../models/` uses model-ID names and contains only 16 checkpoints.
Obtain the other 20 weights and the original filename mapping from the authors,
and arrange working copies under the names required by `checkpoint_inventory()`.
For all 36 models, the evaluator accepts `--include-evaluated-properties`,
`--device cpu` (or `cuda`), and `--output-dir NEW_DIRECTORY`.
Without that flag it evaluates 24 models and also requires `--existing-hlg-dir`
containing previous HOMO/LUMO/GAP evaluation records, which are not bundled.
The two-family evaluator additionally needs models in `../../proceeding/model/`.
These evaluators cannot run on the included 16-checkpoint folder without adaptation.
Outputs include metrics CSVs, split metadata, and `per_molecule_predictions/*.csv.gz`.
Saved paper metrics are available directly in `../../results/gnn_models.csv`.
