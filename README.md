# Molecular Inverse Design with GNNs and MILP

This repository provides the source code, trained models, and experimental data accompanying the paper **“Molecular Inverse Design under the Two-Layered Model with Graph Neural Networks and Mixed Integer Linear Programming.”**

## Overview

We combine graph neural networks (GNNs) and mixed-integer linear programming (MILP) for molecular inverse design under the two-layered model of chemical graphs.

GNNs are trained on the QM9 dataset to predict molecular properties. The trained predictors are then incorporated into MILP formulations to generate molecular graphs satisfying prescribed structural constraints and target property ranges. The resulting molecular candidates are further evaluated using density functional theory (DFT) calculations with Psi4 and PySCF.

Earlier implementations of molecular inference methods are available in the [mol-infer project](https://github.com/ku-dml/mol-infer).

## Repository Structure

| Directory | Description |
|---|---|
| [`code/gnn/`](code/gnn/) | GNN architectures, training, and evaluation |
| [`code/milp/`](code/milp/) | MILP formulations, GNN encoding, and molecular graph generation |
| [`code/dft/`](code/dft/) | Molecular geometry optimization and quantum-chemical property calculations |
| [`inputs/`](inputs/) | Structural specifications, fringe-tree libraries, target property ranges, and experimental settings |
| [`models/`](models/) | Trained GNN models and configuration files |
| [`results/`](results/) | GNN performance, MILP outcomes, molecular candidates, and DFT results |
| [`results/canonical/`](results/canonical/) | Summary tables for the experimental comparisons |
| [`results/molecules/`](results/molecules/) | Generated molecular structures in SDF format |

Code guides: [GNN](code/gnn/README.md) · [MILP](code/milp/README.md) · [DFT](code/dft/README.md).

For Psi4, obtain the basis data from BSE using the [DFT setup instructions](code/dft/README.md#input-preparation).

## Experimental Results

The repository includes the results of the following experiments:

- **GNN prediction:** 36 models trained for nine molecular properties.
- **MILP inverse design:** 960 experimental conditions covering different structural specifications, molecular properties, and GNN architectures.
- **Molecular candidates:** 178 candidate records corresponding to 175 distinct molecular graphs.
- **DFT evaluation:** Psi4 and PySCF calculations for the 175 molecular graphs.

The main results are available in the CSV files under [`results/`](results/), while the corresponding molecular structures are provided under [`results/molecules/`](results/molecules/).

## Trained Models

The 16 GNN checkpoints used for MILP inverse design are included in [`models/checkpoints/`](models/checkpoints/). These models cover HOMO, LUMO, HOMO–LUMO gap, and zero-point vibrational energy (ZPVE), with four architectures for each property.

Model configurations and additional information are provided in [`models/`](models/).

## Dependencies

The implementation uses Python and several scientific computing libraries, including PyTorch, PyTorch Geometric, RDKit, and PuLP.

MILP optimization uses IBM ILOG CPLEX 22.1.1, while DFT calculations use Psi4 1.10.1 and PySCF 2.13.1.

The MILP code uses PuLP 2.6.0 with a custom CPLEX execution interface. External software and the original QM9 dataset must be obtained separately.



