import torch
import torch_geometric
from torch import Tensor, nn
from functools import partial
from typing import Any, List, Mapping, Tuple, Union
from torch_scatter import scatter_add
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Optional, Size


class DataNormalizer(torch.nn.Module):
    """
    Module for normalizing input data.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Initialize the DataNormalizer module.

        Args:
            mean (torch.Tensor): Mean values for normalization.
            std (torch.Tensor): Standard deviation values for normalization.
        """
        super(DataNormalizer, self).__init__()
        self.register_buffer('mean', torch.as_tensor(mean))
        self.register_buffer('std', torch.as_tensor(std))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform data normalization.

        Args:
            input (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Normalized data.
        """
        self.std[self.std==0] = 1
        return (input - self.mean) / self.std

    
class ConvLayer(torch_geometric.nn.conv.MessagePassing):
    """
    Custom convolutional layer.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        """
        Initialize the ConvLayer.

        Args:
            in_channels (Union[int, Tuple[int, int]]): Number of input channels.
            out_channels (int): Number of output channels.
            aggr (str, optional): Aggregation method for message passing. Defaults to 'mean'.
            bias (bool, optional): Whether to include bias terms. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = torch_geometric.nn.dense.linear.Linear(
            in_channels[0], out_channels, bias=bias, weight_initializer='glorot'
        )
        self.lin_r = torch_geometric.nn.dense.linear.Linear(
            in_channels[1], out_channels, bias=False, weight_initializer='glorot'
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the ConvLayer."""
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[torch.Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> torch.Tensor:
        """
        Perform forward propagation.

        Args:
            x (Union[torch.Tensor, OptPairTensor]): Input features.
            edge_index (Adj): Graph connectivity.
            edge_weight (OptTensor, optional): Edge weights. Defaults to None.
            size (Size, optional): Size of the graph. Defaults to None.

        Returns:
            torch.Tensor: Output features.
        """
        num_nodes = x.shape[0]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # If edge_weight == None -> unweighted Ad
        if (edge_weight == None):
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        
        # Normalize Ad
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out.float())

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: torch.Tensor, edge_weight: Optional[Tensor]) -> torch.Tensor:
        """
        Message function for message passing.

        Args:
            x_j (torch.Tensor): Input features.
            edge_weight (Optional[Tensor]): Edge weights. Defaults to None.

        Returns:
            torch.Tensor: Output features.
        """
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GraphNN(torch.nn.Module):
    _layers = {
        'sage': ConvLayer,
        'gcn': partial(torch_geometric.nn.GCNConv, add_self_loops=False),
    }

    def __init__(self, network_type: str, neurons_per_layer: List[int],
                 mean: Optional[Tensor] = None, std: Optional[Tensor] = None, activation: Optional[str] = 'relu'):
        """
        Graph neural network model.

        Args:
            network_type (str): Type of network architecture. Allowed values: 'sage', 'gcn'.
            neurons_per_layer (List[int]): Number of neurons per layer.
            mean (Optional[Tensor]): Mean for data normalization. Defaults to None.
            std (Optional[Tensor]): Standard deviation for data normalization. Defaults to None.
            activation: Optional[nn.Module]: Activation funtion for intermediate layers. Defaults to ReLU.
        """
        super(GraphNN, self).__init__()
        assert network_type.lower() in self._layers.keys()

        self.network_type = network_type
        if mean is not None and std is not None:
            self.normalize = DataNormalizer(mean=mean, std=std)
        else:
            self.normalize = nn.Identity()

        layer = self._layers[network_type.lower()]
        self.gcn_stack = torch.nn.ModuleList()
        for i in range(len(neurons_per_layer) - 1):
            self.gcn_stack.append(layer(in_channels=neurons_per_layer[i], out_channels=neurons_per_layer[i + 1]))
        
        if (activation == 'sigmoid'):
            self.activation = nn.Sigmoid()
        elif (activation == 'tanh'):
            self.activation = nn.Tanh()
        elif (activation == 'relu'):
            self.activation = nn.ReLU(inplace=True)
        elif (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        elif (activation == 'silu'):
            self.activation = nn.SiLU(inplace=True)


    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """
        Loads the model state dictionary.

        Args:
            state_dict (Mapping[str, Any]): State dictionary to load.
            strict (bool): Whether to strictly enforce that the keys in `state_dict` match the keys returned by
                           `self.state_dict()`. Defaults to True.
        """
        sd_keys = state_dict.keys()
        if 'normalize.mean' in sd_keys and 'normalize.std' in sd_keys:
            mean, std = state_dict['normalize.mean'], state_dict['normalize.std']
            self.normalize = DataNormalizer(mean=mean, std=std)
        super(GraphNN, self).load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        """
        Forward pass of the graph neural network.

        Args:
            x (Tensor): Input features.
            edge_index (Tensor): Edge indices.
            edge_weight (Tensor): Edge weights.

        Returns:
            Tensor: Output tensor.
        """
        x = self.normalize(x)

        for i, gcn_block in enumerate(self.gcn_stack):
            x = gcn_block(x.to(torch.float), edge_index, edge_weight)

            if i != (len(self.gcn_stack) - 1):
                x = self.activation(x)

        return x
