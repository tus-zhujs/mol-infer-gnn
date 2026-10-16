from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    pos: torch.Tensor | None = None


def _pool_sum(x: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    out = x.new_zeros((size, x.shape[1]))
    out.index_add_(0, batch, x)
    return out


def _pool_mean(x: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    out = _pool_sum(x, batch, size)
    count = x.new_zeros((size, 1))
    count.index_add_(0, batch, x.new_ones((x.shape[0], 1)))
    return out / count.clamp_min_(1)


class SelfConvNoEdgeType(torch.nn.Module):
    """PyTorch-only equivalent of the project's PyG MessagePassing layer."""

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin_l_self = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_l.reset_parameters()
        self.lin_l_self.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(0, dst, x[src])
        out = self.lin_l(aggregated) + self.lin_l_self(x) + self.bias
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out


class GNNFTNoEdgeType(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        p_channels: int,
        ANN_channels: tuple[int, ...],
        dropout: float | None,
    ):
        super().__init__()
        self.embedding_layer = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList(
            SelfConvNoEdgeType(hidden_channels, hidden_channels, normalize=True)
            for _ in range(num_layers)
        )
        self.convs_to_p = torch.nn.Linear(hidden_channels, p_channels)
        self.Linears = torch.nn.ModuleList()
        self.Linears.append(torch.nn.Linear(p_channels, ANN_channels[0]))
        for i in range(len(ANN_channels) - 1):
            self.Linears.append(torch.nn.Linear(ANN_channels[i], ANN_channels[i + 1]))
        self.dropout = dropout
        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.LeakyReLU(0.1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data: GraphBatch) -> torch.Tensor:
        x = self.embedding_layer(data.x)
        for conv in self.convs:
            x = self.activation2(conv(x, data.edge_index))
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation2(self.convs_to_p(x))
        size = int(data.batch.max().item()) + 1
        x = _pool_mean(x, data.batch, size)
        for layer in self.Linears[:-1]:
            x = self.activation1(layer(x))
        return self.softmax(self.Linears[-1](x))


class GNN2LIntNoEdgeType(torch.nn.Module):
    def __init__(
        self,
        in_channels_atom: int,
        out_channels_FT: int,
        hidden_channels: int,
        num_layers: int,
        p_channels: int,
        ANN_channels: tuple[int, ...],
        dropout: float | None,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels_atom + out_channels_FT if i == 0 else hidden_channels
            self.convs.append(SelfConvNoEdgeType(in_channels, hidden_channels))
        self.convs_to_p = torch.nn.Linear(hidden_channels, p_channels, bias=False)
        self.Linears = torch.nn.ModuleList()
        self.Linears.append(torch.nn.Linear(p_channels, ANN_channels[0]))
        for i in range(len(ANN_channels) - 1):
            self.Linears.append(torch.nn.Linear(ANN_channels[i], ANN_channels[i + 1]))
        self.extra_x_length = None
        self.dropout = dropout
        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.LeakyReLU(0.1)

    def forward(self, data: GraphBatch, FT_output: torch.Tensor) -> torch.Tensor:
        if data.pos is None:
            raise ValueError("Interior graph batch is missing fringe-tree positions")
        x = torch.cat((data.x, FT_output[data.pos, :]), dim=1)
        for conv in self.convs:
            x = self.activation2(conv(x, data.edge_index))
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs_to_p(x)
        size = int(data.batch.max().item()) + 1
        x = self.activation2(_pool_sum(x, data.batch, size))
        for layer in self.Linears[:-1]:
            x = self.activation1(layer(x))
        return self.Linears[-1](x)


class GNN2LNoEdgeType(torch.nn.Module):
    """State-dict-compatible evaluator for network.GNN_2L_no_edge_type."""

    def __init__(
        self,
        *,
        in_channels_FT: int = 14,
        hidden_channels_FT: int = 256,
        num_layers_FT: int = 16,
        p_channels_FT: int = 256,
        ANN_channels_FT: tuple[int, ...] = (256, 128, 64, 32, 16, 8),
        in_channels_IN: int = 7,
        hidden_channels_IN: int = 16,
        num_layers_IN: int = 3,
        p_channels_IN: int = 32,
        ANN_channels_IN: tuple[int, ...] = (16, 1),
        dropout: float | None = None,
    ):
        super().__init__()
        self.GNN_2L_int = GNN2LIntNoEdgeType(
            in_channels_IN,
            ANN_channels_FT[-1],
            hidden_channels_IN,
            num_layers_IN,
            p_channels_IN,
            ANN_channels_IN,
            dropout,
        )
        self.GNN_FT = GNNFTNoEdgeType(
            in_channels_FT,
            hidden_channels_FT,
            num_layers_FT,
            p_channels_FT,
            ANN_channels_FT,
            dropout,
        )

    def encode_fringe_trees(self, ft_data: GraphBatch) -> torch.Tensor:
        return self.GNN_FT(ft_data)

    def forward_interior(self, data: GraphBatch, ft_output: torch.Tensor) -> torch.Tensor:
        return self.GNN_2L_int(data, ft_output)
