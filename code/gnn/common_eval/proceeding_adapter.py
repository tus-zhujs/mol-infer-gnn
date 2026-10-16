from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .model_compat import GNN2LNoEdgeType


@dataclass(frozen=True)
class ProceedingAdapter:
    checkpoint_path: Path
    property_name: str
    hidden_dim: int

    era: str = "proceeding"
    num_layers: int = 3
    target_space: str = "native_eV"
    dropout_training: float = 0.2

    def build_model(self) -> GNN2LNoEdgeType:
        return GNN2LNoEdgeType(
            hidden_channels_IN=self.hidden_dim,
            num_layers_IN=self.num_layers,
            dropout=self.dropout_training,
        )

    def to_native(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction

    @property
    def transform_metadata(self) -> dict[str, object]:
        return {
            "target_space": self.target_space,
            "inverse_applied": False,
            "mean": None,
            "std": None,
            "formula": "identity; checkpoint output is already eV",
        }
