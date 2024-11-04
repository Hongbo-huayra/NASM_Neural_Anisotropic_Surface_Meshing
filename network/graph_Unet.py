from typing import Callable, List, Union
import torch
from torch import Tensor
from torch.nn.functional import interpolate
from torch_geometric.nn import TopKPooling
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import OptTensor
from torch_geometric.utils.repeat import repeat
from .sage_conv import SAGEConv, SAGEConv_residual


class GraphUNet(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        ori_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'leakyrelu',
        norm: Union[str, Callable] = 'batch',
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.ori_channels = ori_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        channels = hidden_channels

        self.in_convs = torch.nn.ModuleList()
        self.in_convs.append(SAGEConv(in_channels, in_channels, channels))
        self.in_convs.append(normalization_resolver(norm, channels))

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(SAGEConv_residual(channels, channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth):
            self.up_convs.append(
                SAGEConv_residual(in_channels, channels))

        out_channels -= ori_channels
        self.out_convs = Linear(channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: OptTensor = None) -> Tensor:

        x = self.in_convs[0](x, edge_index)
        x = self.in_convs[1](x)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        perms = []

        for i in range(self.depth):
            x = self.down_convs[i](x, edge_index)
            x, edge_index, _, _, perm, _ = self.pools[i](
                x, edge_index)

            if i < self.depth-1:
                xs += [x]
                edge_indices += [edge_index]

            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]

            perm = perms[j]

            # interpolate
            tmp_x = x.unsqueeze(0).permute(0, 2, 1)
            up = interpolate(
                tmp_x,
                size=res.size(0),
                mode='nearest-exact')
            up = up.squeeze(0).permute(1, 0)

            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index)

        x = self.out_convs(x)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
