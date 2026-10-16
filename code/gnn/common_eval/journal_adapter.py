from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .model_compat import GNN2LNoEdgeType


def read_config(path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        key, values = parts[0], parts[1:]
        if key.startswith("ANN_channels"):
            config[key] = tuple(int(value) for value in values)
        elif key in {"y_mean", "y_std"}:
            config[key] = float(values[0])
        else:
            config[key] = int(values[0])
    return config


@dataclass(frozen=True)
class JournalAdapter:
    checkpoint_path: Path
    config_path: Path
    property_name: str

    era: str = "journal"
    target_space: str = "train_standardized"
    dropout_training: None = None

    @property
    def config(self) -> dict[str, object]:
        return read_config(self.config_path)

    @property
    def hidden_dim(self) -> int:
        return int(self.config["hidden_channels_IN"])

    @property
    def num_layers(self) -> int:
        return int(self.config["num_layers_IN"])

    @property
    def y_mean(self) -> float:
        return float(self.config["y_mean"])

    @property
    def y_std(self) -> float:
        return float(self.config["y_std"])

    def build_model(self) -> GNN2LNoEdgeType:
        config = self.config
        return GNN2LNoEdgeType(
            in_channels_FT=int(config["in_channels_FT"]),
            hidden_channels_FT=int(config["hidden_channels_FT"]),
            num_layers_FT=int(config["num_layers_FT"]),
            p_channels_FT=int(config["p_channels_FT"]),
            ANN_channels_FT=tuple(config["ANN_channels_FT"]),
            in_channels_IN=int(config["in_channels_IN"]),
            hidden_channels_IN=int(config["hidden_channels_IN"]),
            num_layers_IN=int(config["num_layers_IN"]),
            p_channels_IN=int(config["p_channels_IN"]),
            ANN_channels_IN=tuple(config["ANN_channels_IN"]),
            dropout=None,
        )

    def to_native(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction * self.y_std + self.y_mean

    @property
    def transform_metadata(self) -> dict[str, object]:
        return {
            "target_space": self.target_space,
            "inverse_applied": True,
            "mean": self.y_mean,
            "std": self.y_std,
            "formula": "y_native = y_standardized * train_std + train_mean",
        }
