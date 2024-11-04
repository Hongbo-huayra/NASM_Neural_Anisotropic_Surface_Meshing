from typing import List, Optional, Tuple, Union, Callable

import torch.nn.functional as F
from torch import Tensor
import torch
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver


class SAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        nbr_channels: int,
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "max",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.nbr_channels = nbr_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0] + nbr_channels+1

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_feature: [F_j | F_j - F_i | |F_j - F_i|]
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        dist = torch.linalg.vector_norm(x_j - x_i, dim=-1).unsqueeze(-1)
        msg = torch.cat([x_j, x_j - x_i, dist], dim=-1)
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


class SAGEConv_residual(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 nbr_channels: int = 0,
                 act: Union[str, Callable] = 'leakyrelu',
                 norm: Union[str, Callable] = 'batch',
                 block_num=2) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.block_num = block_num
        self.act = activation_resolver(act)
        self.mlp = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(block_num):
            self.mlp.append(
                SAGEConv(hidden_channels, nbr_channels, hidden_channels))
            self.norms.append(normalization_resolver(norm, hidden_channels))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.mlp:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        input_x = x
        for i in range(self.block_num):
            x = self.mlp[i](x, edge_index)
            x = self.norms[i](x)
            x = self.act(x)
        x = x + input_x
        return x
