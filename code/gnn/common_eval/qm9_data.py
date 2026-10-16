from __future__ import annotations

import csv
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model_compat import GraphBatch


HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414
PROPERTY_CONVERSIONS = {
    "mu": 1.0,
    "alpha": 1.0,
    "homo": HAR2EV,
    "lumo": HAR2EV,
    "gap": HAR2EV,
    "r2": 1.0,
    "zpve": HAR2EV,
    "cv": 1.0,
    "u0_atom": KCALMOL2EV,
}
PROPERTY_NATIVE_UNITS = {
    "mu": "D",
    "alpha": "bohr^3",
    "homo": "eV",
    "lumo": "eV",
    "gap": "eV",
    "r2": "bohr^2",
    "zpve": "eV",
    "cv": "cal/(mol K)",
    "u0_atom": "eV",
}
PROPERTY_SOURCE_UNITS = {
    "mu": "D",
    "alpha": "bohr^3",
    "homo": "Hartree",
    "lumo": "Hartree",
    "gap": "Hartree",
    "r2": "bohr^2",
    "zpve": "Hartree",
    "cv": "cal/(mol K)",
    "u0_atom": "kcal/mol",
}


@dataclass
class MoleculeGraph:
    cid: str
    x: torch.Tensor
    edge_index: torch.Tensor
    pos: torch.Tensor


def load_excluded_cids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {f"gdb_{row['id']}" for row in csv.DictReader(handle)}


def read_sdf_cids(path: Path) -> list[str]:
    cids: list[str] = []
    first_line: str | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if first_line is None:
                first_line = line.rstrip("\r\n")
            if line.strip() == "$$$$":
                cids.append(first_line)
                first_line = None
    return cids


def read_property_values(path: Path, properties: tuple[str, ...]) -> dict[str, dict[str, float]]:
    unknown = set(properties) - PROPERTY_CONVERSIONS.keys()
    if unknown:
        raise KeyError(f"Missing property conversion metadata: {sorted(unknown)}")
    values: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            values[row["mol_id"]] = {
                prop: float(row[prop]) * PROPERTY_CONVERSIONS[prop]
                for prop in properties
            }
    return values


def make_split(filtered_cids: list[str], seed: int = 42) -> tuple[list[str], list[str], list[str]]:
    if len(filtered_cids) != 130_831:
        raise ValueError(f"Expected 130831 filtered molecules, found {len(filtered_cids)}")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(filtered_cids), generator=generator).tolist()
    train = [filtered_cids[i] for i in permutation[:110_000]]
    validation = [filtered_cids[i] for i in permutation[110_000:120_000]]
    test = [filtered_cids[i] for i in permutation[120_000:]]
    return train, validation, test


def _import_chemicalgraph(learning_dir: Path):
    learning_dir = learning_dir.resolve()
    learning_dir_text = str(learning_dir)
    if learning_dir_text in sys.path:
        sys.path.remove(learning_dir_text)
    sys.path.insert(0, learning_dir_text)
    try:
        importlib.import_module("pandas")
    except ModuleNotFoundError:
        sys.modules["pandas"] = types.ModuleType("pandas")
    try:
        importlib.import_module("rdkit")
    except ModuleNotFoundError:
        rdkit = types.ModuleType("rdkit")
        chem = types.ModuleType("rdkit.Chem")
        rdkit.Chem = chem
        sys.modules["rdkit"] = rdkit
        sys.modules["rdkit.Chem"] = chem
    module = importlib.import_module("src.chemicalgraph")
    module_path = Path(module.__file__).resolve()
    expected_path = learning_dir / "src" / "chemicalgraph.py"
    if module_path != expected_path:
        raise ImportError(f"Imported unexpected src.chemicalgraph: {module_path}; expected {expected_path}")
    return module


def construct_selected_graphs(
    sdf_path: Path,
    selected_cids: list[str],
    learning_dir: Path,
) -> tuple[dict[str, MoleculeGraph], GraphBatch]:
    chemicalgraph = _import_chemicalgraph(learning_dir)
    selected = set(selected_cids)
    cg_list = chemicalgraph.ChemicalGraph_list(False)
    with sdf_path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            graph = chemicalgraph.ChemicalGraph()
            answer = graph.construct_from_SDF_file(handle)
            if answer == 0:
                break
            if answer == 1 and graph.CID in selected:
                graph.change_atom_name()
                cg_list.append(graph)
    if len(cg_list) != len(selected_cids):
        raise ValueError(f"Constructed {len(cg_list)} selected graphs, expected {len(selected_cids)}")
    cg_list.get_fringe_tree()

    molecule_graphs: dict[str, MoleculeGraph] = {}
    for cg in cg_list:
        interior = list(cg.interior_indices())
        index = {u: i for i, u in enumerate(interior)}
        x = [cg[u].get_feature() for u in interior]
        pos = [cg[u].fringe_tree_TS for u in interior]
        src: list[int] = []
        dst: list[int] = []
        for u in interior:
            for v in cg[u].adj:
                if v in index:
                    src.append(index[u])
                    dst.append(index[v])
        molecule_graphs[cg.CID] = MoleculeGraph(
            cid=cg.CID,
            x=torch.tensor(np.asarray(x), dtype=torch.float32),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.long),
        )

    ft_x: list[np.ndarray] = []
    ft_src: list[int] = []
    ft_dst: list[int] = []
    ft_batch: list[int] = []
    offset = 0
    for graph_index, tree in enumerate(cg_list.fringe_tree_list):
        vertices = list(tree.vertex_list.keys())
        index = {u: i for i, u in enumerate(vertices)}
        for u in vertices:
            ft_x.append(tree[u].get_feature_for_FT(tree.root))
            ft_batch.append(graph_index)
            for v in tree[u].adj:
                ft_src.append(offset + index[u])
                ft_dst.append(offset + index[v])
        offset += len(vertices)
    ft_graph = GraphBatch(
        x=torch.tensor(np.asarray(ft_x), dtype=torch.float32),
        edge_index=torch.tensor([ft_src, ft_dst], dtype=torch.long),
        batch=torch.tensor(ft_batch, dtype=torch.long),
    )
    return molecule_graphs, ft_graph


def batch_molecules(graphs: list[MoleculeGraph]) -> GraphBatch:
    xs: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    batches: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    offset = 0
    for graph_index, graph in enumerate(graphs):
        xs.append(graph.x)
        edges.append(graph.edge_index + offset)
        batches.append(torch.full((graph.x.shape[0],), graph_index, dtype=torch.long))
        positions.append(graph.pos)
        offset += graph.x.shape[0]
    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edges, dim=1),
        batch=torch.cat(batches, dim=0),
        pos=torch.cat(positions, dim=0),
    )
