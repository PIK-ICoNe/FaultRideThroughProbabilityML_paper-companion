import numpy as np

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch_geometric.nn import (
    GCNConv,
    ARMAConv,
    SAGEConv,
    TAGConv,
    TransformerConv,
    GATv2Conv,
)
from torch_geometric.nn import Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import to_hetero


from torchmetrics import (
    F1Score,
    FBetaScore,
    Recall,
    Precision,
    R2Score,
    Accuracy,
    AUROC,
    MeanAbsoluteError,
    MeanSquaredError
    # AveragePrecision,
)
from torchmetrics.classification import MultilabelAveragePrecision

from ogb.graphproppred import  Evaluator

from DBGNN_src.DBGNN import *


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.fc(x)
        return F.relu(x)

class activation_function(torch.nn.Module):
    def __init__(self, activation_name):
        super(activation_function, self).__init__()
        if activation_name == "None":
            self.activation = nn.Identity()
        elif activation_name == "ReLU":
            self.activation = nn.ReLU()
        elif activation_name == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        else:
            print("unsupported activation_name: " + str(activation_name))

    def forward(self, x):
        return self.activation(x)


class ArmaNet_bench(torch.nn.Module):
    def __init__(
        self,
        num_classes=1,
        num_node_features=1,
        num_layers=4,
        num_stacks=3,
        final_sigmoid_layer=True,
    ):
        super(ArmaNet_bench, self).__init__()
        self.conv1 = ARMAConv(
            num_node_features,
            16,
            num_stacks=num_stacks,
            num_layers=num_layers,
            shared_weights=True,
            dropout=0.25,
        )
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2 = ARMAConv(
            16,
            num_classes,
            num_stacks=num_stacks,
            num_layers=num_layers,
            shared_weights=True,
            dropout=0.25,
            act=None,
        )
        self.conv2_bn = nn.BatchNorm1d(num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.final_sigmoid_layer = final_sigmoid_layer
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, batch):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight.float())
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index=edge_index, edge_weight=edge_weight.float())
        # x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.endLinear.reset_parameters()


class GCNConvModule(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, activation_name, dropout):
        super(GCNConvModule, self).__init__()
        self.activation = activation_function(activation_name)
        self.conv = GCNConv(input_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv(x.float(), edge_index=edge_index)  # ,
        # edge_weight=edge_weight.float())
        x = self.dropout(x)
        return self.activation(x)


class GCNModel(nn.Module):
    def __init__(
        self,
        num_layers,
        input_features_node_dim,
        hidden_channels,
        activation,
        dropout,
        linear_layer_after_conv,
        ll_after_conv_dim,
        final_linear_layer,
        final_sigmoid_layer,
        pool=False,
    ):
        super(GCNModel, self).__init__()
        self.dropout = dropout
        self.linear_layer_after_conv = linear_layer_after_conv
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer
        self.convlist = nn.ModuleList()
        conv = GCNConvModule(
            input_features_node_dim, hidden_channels, activation, dropout
        )
        self.convlist.append(conv)
        for i in range(1, num_layers):
            conv = GCNConvModule(hidden_channels, hidden_channels, activation, dropout)
            self.convlist.append(conv)
        if linear_layer_after_conv == True:
            self.ll_after_conv = nn.Linear(hidden_channels, ll_after_conv_dim)
        self.pool = pool
        if final_linear_layer == True:
            self.linear_layer = nn.Linear(hidden_channels, 1)
        if final_sigmoid_layer:
            self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, batch):
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](x, edge_index, edge_attr, batch)
        if self.linear_layer_after_conv:
            x = self.ll_after_conv(x)
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        if self.final_linear_layer == True:
            x = self.linear_layer(x)
        if self.final_sigmoid_layer:
            x = self.sigmoid_layer(x)
        return x


class DBGNNModel(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_channels_n,
        out_channels_n,
        hidden_channels_n,
        in_channels_e,
        hidden_channels_e,
        num_classes,
        num_steps,
        activation_name_n,
        activation_name_e,
        dropout_n_in_lay,
        dropout_e_in_lay,
        dropout_n_after_lay,
        dropout_e_after_lay,
        dropout_final,
        skip_connection_n,
        skip_connection_e,
        Δ,
        dense_after_linDB,
        pool,
        final_linear_layer,
        final_sigmoid_layer=True,
        bias_zero=False,
        scale_features=False,
    ):
        super(DBGNNModel, self).__init__()

        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer

        self.dbgnn_conv = DBGNN(
            in_channels_n,
            hidden_channels_n,
            out_channels_n,
            in_channels_e,
            hidden_channels_e,
            num_layers,
            num_steps,
            activation_name_n,
            activation_name_e,
            dropout_n_in_lay=dropout_n_in_lay,
            dropout_e_in_lay=dropout_e_in_lay,
            dropout_n_after_lay=dropout_n_after_lay,
            dropout_e_after_lay=dropout_e_after_lay,
            dropout_final=dropout_final,
            Δ=Δ,
            skip_connection_n=skip_connection_n,
            skip_connection_e=skip_connection_e,
            dense_after_linDB=dense_after_linDB,
            bias_zero=bias_zero,
            scale_features=scale_features,
        )

        self.pool = pool
        if final_linear_layer:
            self.endLinear = nn.Linear(out_channels_n, num_classes)
            if bias_zero == True:
                self.endLinear.bias.data = torch.zeros_like(self.endLinear.bias)
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.dbgnn_conv(x, edge_index, edge_attr)
        x = torch.t(x)
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        if self.final_linear_layer:
            x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class GATModel(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_channels,
        heads,
        dropout,
        add_self_loops,
        edge_dim,
        v2,
        linear_layer_after_conv,
        ll_after_conv_dim,
        final_linear_layer,
        final_sigmoid_layer,
        hetero=False,
        pool=False,
    ):
        super(GATModel, self).__init__()
        self.linear_layer_after_conv = linear_layer_after_conv
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer
        self.convlist = nn.ModuleList()
        if hetero == "hetero":
            self.hetero = True
        else:
            self.hetero = False

        for i in range(0, num_layers):
            if v2 == True:
                conv = GATv2Conv(
                    (-1, -1),
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    edge_dim=edge_dim,
                )
            else:
                conv = GATConv(
                    (-1, -1),
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    edge_dim=edge_dim,
                )
            self.convlist.append(conv)
        if v2 == True:
            conv = GATv2Conv(
                (-1, -1),
                out_channels=hidden_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=add_self_loops,
                edge_dim=edge_dim,
            )
        else:
            conv = GATConv(
                (-1, -1),
                out_channels=hidden_channels,
                heads=1,
                dropout=dropout,
                add_self_loops=add_self_loops,
                edge_dim=edge_dim,
            )
        self.convlist.append(conv)
        if linear_layer_after_conv == True:
            self.ll_after_conv = nn.Linear(hidden_channels, ll_after_conv_dim)
        self.pool = pool
        if final_linear_layer == True:
            if linear_layer_after_conv == True:
                self.final_ll = nn.Linear(ll_after_conv_dim, 1)
            else:
                self.final_ll = nn.Linear(hidden_channels, 1)
        if final_sigmoid_layer:
            self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, batch):
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](x.float(), edge_index, edge_attr.float())
        if self.linear_layer_after_conv:
            x = self.ll_after_conv(x)
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        if self.final_linear_layer == True:
            x = self.final_ll(x)
        if self.final_sigmoid_layer:
            x = self.sigmoid_layer(x)

        return x


class TAGConvModule(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, activation_name, K, dropout):
        super(TAGConvModule, self).__init__()
        self.activation = activation_function(activation_name)
        self.conv = TAGConv(input_channels, hidden_channels, K=K)
        # self.conv = TAGConv(-1, hidden_channels, K = K)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv(x, edge_index=edge_index)  # ,
        # edge_weight=edge_weight.float())
        x = self.dropout(x)
        return self.activation(x)


class TAGModel(nn.Module):
    def __init__(
        self,
        num_layers,
        num_input_node_features,
        hidden_channels,
        K,
        activation,
        dropout,
        linear_layer_after_conv,
        ll_after_conv_dim,
        final_linear_layer,
        final_sigmoid_layer,
    ):
        super(TAGModel, self).__init__()
        self.dropout = dropout
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer
        self.convlist = nn.ModuleList()
        conv = TAGConvModule(num_input_node_features, hidden_channels, activation, K, dropout)
        self.convlist.append(conv)
        for i in range(1, num_layers):
            conv = TAGConvModule(
                hidden_channels, hidden_channels, activation, K, dropout
            )
            self.convlist.append(conv)

        if final_linear_layer == True:
            self.linear_layer = nn.Linear(hidden_channels, 1)
        if final_sigmoid_layer:
            self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, batch):
        # x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](x, edge_index, edge_attr, batch)
            x = F.relu(x)
            # x = nn.Dropout(p=self.dropout)

        if self.final_linear_layer == True:
            x = self.linear_layer(x)
        if self.final_sigmoid_layer:
            x = self.sigmoid_layer(x)
        return x


class MetricObjectNan(nn.Module):
    def __init__(self):
        super(MetricObjectNan, self).__init__()

    def forward(self, outputs, labels):
        return torch.nan

    def update(self, outputs, labels):
        pass

    def reset(self):
        pass

    def compute(self):
        pass

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TAGConv, GATConv

class gnn_model_generic(torch.nn.Module):
    """
    A Graph Neural Network (GNN) module with skip connections and multiple layers, where the number
    of layers, the type of GNN layer, and other parameters can be defined.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        hidden_channels (int): Number of hidden features.
        num_layers (int): Number of layers in the GNN.
        layer_type (str): Type of GNN layer to use ('GCNConv', 'TAGConv', 'GATConv').
        K (int): Polynomial order for TAGConv. Default is 3.
        heads (int): Number of heads for GATConv. Default is 1.
        use_batch_norm (bool): Whether to use batch normalization. Default is True.
        use_skip_connections (bool): Whether to use skip connections between layers. Default is True.
        dropout_rate (float): Dropout probability. Default is 0.0 (no dropout).
        regression_head_layers (list): List of hidden layer sizes for the regression head. Example: [100, 500, 1].
    """
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, 
                 layer_type='GCNConv', K=3, heads=1, input_edge_dim=None, use_batch_norm=True, 
                 use_skip_connections=True, dropout_rate=0.0, regression_head_layers=None):
        super(gnn_model_generic, self).__init__()
        
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.K = K
        self.heads = heads
        self.use_batch_norm = use_batch_norm
        self.use_skip_connections = use_skip_connections
        self.dropout_rate = dropout_rate

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() if use_batch_norm else None

        # Linear projection for skip connections if dimensions do not match
        if in_channels != hidden_channels:
            self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        else:
            self.input_proj = None

        # Define a function to get the layer
        def get_layer(in_channels, out_channels, layer_type, K, heads):
            """
            Returns a GNN layer based on the specified type.
            """
            if layer_type == 'GCNConv':
                return GCNConv(in_channels, out_channels)
            elif layer_type == 'TAGConv':
                return TAGConv(in_channels, out_channels, K=K)
            elif layer_type == 'GATConv':
                return GATConv(in_channels, out_channels // heads, heads=heads, edge_dim=input_edge_dim, dropout=dropout_rate)
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        # Input layer
        self.convs.append(get_layer(in_channels, hidden_channels, layer_type, K, heads))
        if use_batch_norm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads if layer_type == 'GATConv' else hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            in_channels = hidden_channels * 2 if use_skip_connections else hidden_channels
            # self.convs.append(get_layer(in_channels * heads if layer_type == 'GATConv' else in_channels, hidden_channels, layer_type, K, heads))
            self.convs.append(get_layer(in_channels, hidden_channels, layer_type, K, heads))
            if use_batch_norm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads if layer_type == 'GATConv' else hidden_channels))

        # Regression head (customizable)
        if use_skip_connections:
            regression_head_input_size = hidden_channels * 2
            if regression_head_layers != None:
                regression_head_layers.insert(0, regression_head_input_size)
                regression_head_layers.append(out_channels)
        else:
            regression_head_input_size = hidden_channels
            if regression_head_layers != None:
                regression_head_layers.insert(0, regression_head_input_size)
                regression_head_layers.append(out_channels)
        self.regression_head_layers = regression_head_layers or [regression_head_input_size, out_channels]
        self.regression_head = self._build_regression_head(self.regression_head_layers)
        if layer_type in ["GCNConv", "TAGConv"]:
            self.edge_mlp = MLP(input_edge_dim, 1)

    def _build_regression_head(self, layer_sizes):
        """
        Builds the regression head dynamically from a list of layer sizes.

        Args:
            layer_sizes (list): List of integers where each entry represents the size of that layer.

        Returns:
            torch.nn.Sequential: The regression head as a sequential model.
        """
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Add activation and dropout between all except the last layer
                layers.append(torch.nn.ReLU())
                if self.dropout_rate > 0.0:
                    layers.append(torch.nn.Dropout(p=self.dropout_rate))
        return torch.nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass of the GNN.

        Args:
            data: PytorchGeometric data object
        
        Returns:
            torch.Tensor: Output node features.
        """

        if self.input_proj is not None:
            skip_connection = self.input_proj(x)
        else:
            skip_connection = x

        # Helper function to handle layers with/without edge_attr
        def apply_conv_layer(conv, x, edge_index, edge_attr):
            # Only pass edge_attr if the layer supports it
            if isinstance(conv, (GATConv)):  # List layers that support edge_attr
                return conv(x, edge_index, edge_attr)
            elif isinstance(conv, (TAGConv)):
                edge_attr_1d = self.edge_mlp(edge_attr)
                x = conv(x, edge_index, edge_attr_1d)
            else:
                x = conv(x, edge_index)
            if self.dropout_rate > 0.0 and isinstance(conv, (GCNConv, TAGConv)):
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            return x


        # GNN layers
        for i in range(self.num_layers):
            # x = self.convs[i](x, edge_index)
            x = apply_conv_layer(self.convs[i], x, edge_index, edge_attr)
            if self.use_batch_norm:
                x = self.bns[i](x)
            x = F.relu(x)
            if self.dropout_rate > 0.0:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if self.use_skip_connections:
                x = torch.cat([x, skip_connection], dim=1)  # Concatenate skip connection with the original input

        # Pass through the regression head
        x = self.regression_head(x)
        
        return x

class MetricObjectConvertLabelsLong(nn.Module):
    def __init__(self, name, task_argument=False, num_classes=False, beta=False):
        super(MetricObjectConvertLabelsLong, self).__init__()
        if name == "AveragePrecision":
            self.metric = AveragePrecision(task=task_argument, num_classes=num_classes)
        elif name == "auroc":
            self.metric = AUROC(task=task_argument, num_classes=num_classes)
        elif name == "MultilabelAveragePrecision":
            self.metric = MultilabelAveragePrecision(num_labels=10, average="macro", thresholds=None)
        else:
            print("unnkown metric")

    def forward(self, outputs, labels):
        return self.metric(outputs, labels.long())

    def reset(self):
        self.metric.reset()

    def update(self, output, label):
        self.metric.update(output, label.long())

    def compute(self):
        res = self.metric.compute()
        self.metric.reset()
        return res.item()


class MetricObject(nn.Module):
    def __init__(self, name, task_argument=False, num_classes=False, beta=False, average=False):
        super(MetricObject, self).__init__()
        if name == "r2":
            self.metric = R2Score()
        elif name == "accuracy":
            self.metric = Accuracy(task=task_argument, num_classes=num_classes)
        elif name == "f1":
            if average !=False:
                self.metric = F1Score(task=task_argument, num_classes=num_classes, average = average)
            else:    
                self.metric = F1Score(task=task_argument, num_classes=num_classes)
        elif name == "fbeta":
            self.metric = FBetaScore(
                task=task_argument, num_classes=num_classes, beta=beta
            )
        elif name == "recall":
            self.metric = Recall(task=task_argument, num_classes=num_classes)
        elif name == "precision":
            self.precision = Precision(task=task_argument, num_classes=num_classes)
        # elif name == "auroc":
        #     self.metric = AUROC(task = task_argument, num_classes=num_classes)
        elif name == "MAE":
            self.metric = MeanAbsoluteError()
        elif name == "MSE":
            self.metric = MeanSquaredError()
        elif name == "AveragePrecision":
            self.metric = AveragePrecision(task=task_argument, num_classes=num_classes)
        else:
            print("unnkown metric")

    def forward(self, outputs, labels):
        return self.metric(outputs, labels)

    def reset(self):
        self.metric.reset()

    def update(self, output, label):
        self.metric.update(output, label)

    def compute(self):
        res = self.metric.compute()
        self.metric.reset()
        return res.item()


class noGradientClippingObject(nn.Module):
    def __init__(self):
        super(noGradientClippingObject, self).__init__()

    def forward(self, parameters):
        pass


class GradientClippingNorm(nn.Module):
    def __init__(self, max_norm):
        super(GradientClippingNorm, self).__init__()
        self.max_norm = max_norm

    def forward(self, parameters):
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)


class GradientClippingValue(nn.Module):
    def __init__(self, clip_value):
        super(GradientClippingValue, self).__init__()
        self.clip_value = clip_value

    def forward(self, parameters):
        torch.nn.utils.clip_grad_value_(parameters, self.clip_value)


class GradientClippingObject(nn.Module):
    def __init__(
        self,
        gradient_clipping_norm,
        gradient_clipping_max_norm,
        gradient_clipping_value,
        gradient_clipping_value_clip,
    ):
        super(GradientClippingObject, self).__init__()
        if gradient_clipping_norm:
            self.gradient_clipping_norm = GradientClippingNorm(
                gradient_clipping_max_norm
            )
        else:
            self.gradient_clipping_norm = noGradientClippingObject()
        if gradient_clipping_value:
            self.gradient_clipping_value = GradientClippingValue(
                gradient_clipping_value_clip
            )
        else:
            self.gradient_clipping_value = noGradientClippingObject()

    def forward(self, model):
        self.gradient_clipping_norm(model.parameters())
        self.gradient_clipping_value(model.parameters())


# class F1Score(nn.Module):
#     def __init__(self, task, num_classes):
#         super(F1Score, self).__init__()
#         self.task = task
#         self.num_classes = num_classes
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GNNmodule(nn.Module):
    def __init__(self, config, criterion_positive_weight=False, config_ray=False):
        super(GNNmodule, self).__init__()
        cuda = config["cuda"]
        if "Fbeta::beta" in config:
            self.beta = config["Fbeta::beta"]
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.cuda = True
            print("cuda availabe:: send model to GPU")
        else:
            self.cuda = False
            self.device = torch.device("cpu")
            print("cuda unavailable:: train model on cpu")
        self.critierion_positive_weight = criterion_positive_weight
        if type(self.critierion_positive_weight) != bool:
            self.critierion_positive_weight = torch.tensor(
                self.critierion_positive_weight
            ).to(self.device)

        # seeds
        torch.manual_seed(config["manual_seed"])
        torch.cuda.manual_seed(config["manual_seed"])
        np.random.seed(config["manual_seed"])
        if self.cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        model_name = config["model_name"]
        final_sigmoid_layer = config["final_sigmoid_layer"]
        self.dtype = config["dtype"]
        self.grid_type = config["grid_type"]
        
        if "dropout_final" not in config:
            dropout_final = False
        else:
            dropout_final = config["dropout_final"]

        if model_name == "ArmaNet_bench":
            model = ArmaNet_bench(final_sigmoid_layer=final_sigmoid_layer)
        elif model_name == "GCN":
            model = GCNModel(
                config["num_layers"],
                config["input_features_node_dim"],
                config["hidden_channels"],
                config["activation_name_n"],
                config["dropout_n"],
                config["linear_layer_after_conv"],
                config["ll_after_conv_dim"],
                config["final_linear_layer"],
                config["final_sigmoid_layer"],
                config["pool"]
            )
        elif model_name == "GAT":
            model = GATModel(
                config["num_layers"],
                config["hidden_channels"],
                config["heads"],
                config["dropout_n"],
                config["add_self_loops"],
                config["input_features_edge_dim"],
                config["GAT::v2"],
                config["linear_layer_after_conv"],
                config["ll_after_conv_dim"],
                config["final_linear_layer"],
                config["final_sigmoid_layer"],
                config["grid_type"],
                config["pool"]
            )
        elif model_name == "TAG":
            model = TAGModel(
                config["num_layers"],
                config["input_features_node_dim"],
                config["hidden_channels"],
                config["TAG::K"],
                config["activation_name_n"],
                config["dropout_n"],
                config["linear_layer_after_conv"],
                config["ll_after_conv_dim"],
                config["final_linear_layer"],
                config["final_sigmoid_layer"],
            )
        elif config["model_name"] == "DBGNN":
            if "dropout_n_in_lay" not in config:
                dropout_n_in_lay = config["dropout_n"]
            else:
                dropout_n_in_lay = config["dropout_n_in_lay"]
            if "dropout_e_in_lay" not in config:
                dropout_e_in_lay = config["dropout_e"]
            else:
                dropout_e_in_lay = config["dropout_e_in_lay"]

            if "dropout_n_after_lay" not in config:
                 dropout_n_after_lay = 0.
            else:
                dropout_n_after_lay = config["dropout_n_after_lay"]
            
            if "dropout_e_after_lay" not in config:
                dropout_e_after_lay = 0.
            else:
                dropout_e_after_lay = config["dropout_e_after_lay"]
        
            model = DBGNNModel(
                num_layers=config["num_layers"],
                in_channels_n=config["DBGNN::in_channels_n"],
                out_channels_n=config["DBGNN::out_channels_n"],
                hidden_channels_n=config["DBGNN::hidden_channels_n"],
                in_channels_e=config["DBGNN::in_channels_e"],
                hidden_channels_e=config["DBGNN::hidden_channels_e"],
                num_classes=config["num_classes"],
                num_steps=config["DBGNN::num_steps"],
                activation_name_n=config["activation_name_n"],
                activation_name_e=config["activation_name_e"],
                dropout_n_in_lay=dropout_n_in_lay,
                dropout_e_in_lay=dropout_e_in_lay,
                dropout_n_after_lay=dropout_n_after_lay,
                dropout_e_after_lay=dropout_e_after_lay,
                dropout_final=dropout_final,
                skip_connection_n=config["skip_connection_n"],
                skip_connection_e=config["skip_connection_e"],
                Δ=config["DBGNN::Δ"],
                dense_after_linDB=config["DBGNN::dense_after_linDB"],
                pool=config["pool"],
                final_linear_layer=config["final_linear_layer"],
                final_sigmoid_layer=final_sigmoid_layer,
                bias_zero=config["bias_zero"],
                scale_features=config["DBGNN::scale_features"],
            )
        elif config["model_name"] == "generic_gnn":
            model = gnn_model_generic(
                in_channels = config["model::in_channels"], out_channels = config["model::out_channels"],
                hidden_channels = config["model::hidden_channels"], num_layers = config["num_layers"], layer_type=config["model::layer_type"], K=config["TAG::K"], heads=config["heads"], input_edge_dim=config["input_features_edge_dim"], use_batch_norm=config["model::use_batch_norm"], use_skip_connections=config["model::use_skip_connections"], dropout_rate=config["model::dropout_rate"], regression_head_layers=config["model::regression_head_layers"]
            )
        else:
            print("error: model type unkown")
        if self.grid_type == "hetero":
            data = config["hetero::datasample"]
            # model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            model = to_hetero(model, data.metadata(), aggr="sum")
        # model.double()
        if config["model_name"] != "GAT":
            # model = torch.compile(model)
            print("did not precompile model")
        if self.dtype == "float64":
            model.double()
        if self.dtype == "float16":
            model.half()

        model.to(self.device)

        self.model = model

        # criterion
        if config["criterion"] == "MSELoss":
            if criterion_positive_weight == True:
                self.criterion = nn.MSELoss(reduction="none")
            else:
                self.criterion = nn.MSELoss()
        if config["criterion"] == "L1Loss":
            if criterion_positive_weight == True:
                self.criterion = nn.L1Loss(reduction="none")
            else:
                self.criterion = nn.L1Loss()
        if config["criterion"] == "BCEWithLogitsLoss":
            if criterion_positive_weight == False:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(criterion_positive_weight)
                )
                print("positive_weigt used for criterion: ", criterion_positive_weight)
        if config["criterion"] == "BCELoss":
            self.criterion = nn.BCELoss()
        if config["criterion"] == "CELoss":
            self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        # set opimizer
        if config["optim::optimizer"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=config["optim::LR"],
                momentum=config["optim::momentum"],
                weight_decay=config["optim::weight_decay"],
            )
        if config["optim::optimizer"] == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["optim::LR"],
                weight_decay=config["optim::weight_decay"],
            )
        if config["optim::optimizer"] == "adamW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config["optim::LR"],
                weight_decay=config["optim::weight_decay"],
            )

        self.optimizer = optimizer

        # scheduler
        scheduler_name = config["optim::scheduler"]
        self.scheduler_name = scheduler_name
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=config["optim::ReducePlat_patience"],
                factor=config["optim::LR_reduce_factor"],
            )
        elif scheduler_name == "stepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["optim::stepLR_step_size"],
                gamma=config["optim::LR_reduce_factor"],
            )
        elif scheduler_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=config["optim::ExponentialLR::gamma"], last_epoch=-1
            )
        elif scheduler_name == "OneCycleLR":
            steps_per_epoch = config_ray["len_trainloader"]
            if "optim::pct_start" not in config:
                pct_start = 0.3
            else:
                pct_start = config["optim::pct_start"]
            if "optim::three_phase" not in config:
                three_phase = False
            else:
                three_phase = config["optim::three_phase"]
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config["optim::max_LR"],
                steps_per_epoch=steps_per_epoch,
                epochs=config["epochs"],
                div_factor=config["optim::div_factor"],
                anneal_strategy=config["optim::anneal_strategy"],
                final_div_factor=config["optim::final_div_factor"],
                pct_start=pct_start,
                three_phase=three_phase,
            )
        elif scheduler_name == "None":
            scheduler = None
        elif scheduler_name == None:
            scheduler = None
        self.scheduler = scheduler

        self.gradient_clipping = GradientClippingObject(
            config["gradient_clipping::grad_norm"],
            config["gradient_clipping::grad_norm::max_norm"],
            config["gradient_clipping::grad_value"],
            config["gradient_clipping::grad_value::clip_value"],
        )

        # metrics
        self.init_metrics(config)

        if config["dataset::name"] == "ogbg-molpcba":
            use_ogbg_molpcba = True
            self.evaluator = Evaluator('ogbg-molpcba')
        else:
            use_ogbg_molpcba = False
        # select proper training function
        if use_ogbg_molpcba:
            self.train_epoch_classification = self.train_epoch_molpcba
            self.eval_model_classification = self.eval_model_molpcba
        else:
            if "task_type::mask_nans" not in config:
                config["task_type::mask_nans"] = False
            if config["task_type"] == "classification":
                if config["task_type::mask_nans"] == False:
                    self.train_epoch_classification = self.train_epoch_classification_basic
                    self.eval_model_classification = self.eval_model_classification_basic
                elif config["task_type::mask_nans"]:
                    self.train_epoch_classification = self.train_epoch_classification_mask_nans
                    self.eval_model_classification = self.eval_model_classification_mask_nans
            
            if "dataset::single_grid" not in config:
                config["dataset::single_grid"] = False
            if config["dataset::single_grid"]:
                if "task_type::mask_nans" not in config:
                    self.train_eval_epoch_1grid = self.train_eval_epoch_1grid_basic
                elif config["task_type::mask_nans"]:
                    self.train_eval_epoch_1grid = self.train_eval_epoch_1grid_mask_nans
                
                

    def init_metrics(self, config):
        num_classes = config["num_classes"]
        if "task_argument" in config:
            task_argument = config["task_argument"]
        else:
            if num_classes > 2:
                task_argument = "multiclass"
            else:
                task_argument = "binary"
        if "metric::average" in config:
            average = config["metric::average"]
        else:
            average = False
        print("task_argument: ", task_argument)
        list_metrics = config["list_metrics"]
        if "r2" in list_metrics:
            self.r2_score = MetricObject("r2").to(self.device)
        else:
            self.r2_score = MetricObjectNan()
        if "accuracy" in list_metrics:
            self.accuracy = MetricObject("accuracy", task_argument, num_classes).to(
                self.device
            )
        else:
            self.accuracy = MetricObjectNan()
        if "f1" in list_metrics:
            self.f1_score = MetricObject("f1", task_argument, num_classes, average=average).to(
                self.device
            )
        else:
            self.f1_score = MetricObjectNan()
        if "fbeta" in list_metrics:
            self.fbeta = MetricObject(
                "fbeta", task_argument, num_classes, beta=self.beta
            ).to(self.device)
        else:
            self.fbeta = MetricObjectNan()
        if "recall" in list_metrics:
            self.recall = MetricObject("recall", task_argument, num_classes).to(
                self.device
            )
        else:
            self.recall = MetricObjectNan()
        if "precision" in list_metrics:
            self.precision = MetricObject("precision", task_argument, num_classes).to(
                self.device
            )
        else:
            self.precision = MetricObjectNan()
        if "auroc" in list_metrics:
            self.auroc = MetricObjectConvertLabelsLong(
                "auroc", task_argument, num_classes
            ).to(self.device)
        else:
            self.auroc = MetricObjectNan()
        if "MAE" in list_metrics:
            self.mae_score = MetricObject("MAE").to(self.device)
        else:
            self.mae_score = MetricObjectNan()
        if "MSE" in list_metrics:
            self.mse_score = MetricObject("MSE").to(self.device)
        else:
            self.mse_score = MetricObjectNan()
        # if "AveragePrecision" in list_metrics:
        #     self.average_precision = MetricObjectConvertLabelsLong(
        #         "AveragePrecision", task_argument, num_classes
        #     ).to(self.device)
        # else:
        #     self.average_precision = MetricObjectNan()
        if "MultilabelAveragePrecision" in list_metrics:
            self.average_precision = MetricObjectConvertLabelsLong(
                "MultilabelAveragePrecision", num_classes)
        else:
            self.average_precision = MetricObjectNan()
    def forward(self, x):
        # compute model prediction
        y = self.model(x)
        return y

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            # print(fname)
            perf_dict["state_dict"] = self.model.state_dict()
            torch.save(perf_dict, fname)
        return None

    def scheduler_step(self, criterion):
        scheduler_name = self.scheduler_name
        if scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(criterion)
        if scheduler_name == "stepLR":
            self.scheduler.step()
        if scheduler_name == "ExponentialLR":
            self.scheduler.step()

    def train_epoch_regression(self, data_loader):
        self.model.train()
        loss = 0.0
        for _, (data) in enumerate(data_loader):
            data.to(self.device)
            self.optimizer.zero_grad()
            output = torch.squeeze(
                self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            )
            labels = data.y
            # Check if mask exists and apply it
            if hasattr(data, 'mask'):
                mask = data.mask
                output = output[mask]
                labels = labels[mask]
            temp_loss = self.criterion(output, labels.float())
            temp_loss.backward()
            self.gradient_clipping(self.model)
            self.optimizer.step()
            loss += temp_loss.item()
            self.r2_score.update(output, labels)
            self.mae_score.update(output, labels)
            self.mse_score.update(output, labels)
        self.scheduler_step(loss)
        return loss/data.y.shape[0], self.r2_score.compute(), self.mae_score.compute(), self.mse_score.compute()

    def train_epoch_classification_basic(self, data_loader):
        self.model.train()
        loss = 0.0
        for _, (data) in enumerate(data_loader):
            data.to(self.device)
            self.optimizer.zero_grad()
            output = torch.squeeze(
                self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            )
            labels = torch.squeeze(data.y)
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.gradient_clipping(self.model)
            self.optimizer.step()
            loss += temp_loss.item()
            self.f1_score.update(output, labels)
            self.fbeta.update(output, labels)
            self.accuracy.update(output, labels)
            self.recall.update(output, labels)
            self.precision.update(output, labels)
            self.auroc.update(output, labels)
            self.average_precision.update(output, labels)
        self.scheduler_step(loss)
        return (
            loss/data.y.shape[0],
            self.accuracy.compute(),
            self.f1_score.compute(),
            self.fbeta.compute(),
            self.recall.compute(),
            self.precision.compute(),
            self.auroc.compute(),
            self.average_precision.compute(),
        )
        
    def train_epoch_classification_mask_nans(self, data_loader):
        self.model.train()
        loss = 0.0
        for _, (data) in enumerate(data_loader):
            data.to(self.device)
            self.optimizer.zero_grad()
            output = torch.squeeze(
                self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            )
            labels = torch.squeeze(data.y)
            mask = torch.isnan(labels)
            labels = labels[~mask]
            output = output[~mask]
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.gradient_clipping(self.model)
            self.optimizer.step()
            loss += temp_loss.item()
            self.f1_score.update(output, labels)
            self.fbeta.update(output, labels)
            self.accuracy.update(output, labels)
            self.recall.update(output, labels)
            self.precision.update(output, labels)
            self.auroc.update(output, labels)
            self.average_precision.update(output, labels)
        self.scheduler_step(loss)
        return (
            loss/data.y.shape[0],
            self.accuracy.compute(),
            self.f1_score.compute(),
            self.fbeta.compute(),
            self.recall.compute(),
            self.precision.compute(),
            self.auroc.compute(),
            self.average_precision.compute(),
        )
    
    def train_epoch_molpcba(self, data_loader):
        self.model.train()
        loss = 0.0
        num_target_classes = data_loader.dataset[0].y.shape[1]
        all_labels = torch.Tensor(0,num_target_classes).to(self.device)
        all_outputs = torch.Tensor(0,num_target_classes).to(self.device)
        for _, (data) in enumerate(data_loader):
            data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            labels = data.y
            mask = torch.isnan(labels)
            labels_masked = labels[~mask]
            output_masked = output[~mask]
            temp_loss = self.criterion(output_masked, labels_masked)
            temp_loss.backward()
            self.gradient_clipping(self.model)
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels], dim=0)
            all_outputs = torch.cat([all_outputs, output], dim=0)
        input_dict = {'y_true': all_labels, 'y_pred': all_outputs}
        own_average_precision = self.evaluator.eval(input_dict)["ap"]
        self.scheduler_step(loss)
        return (
            loss/data.y.shape[0],
            self.accuracy.compute(),
            self.f1_score.compute(),
            self.fbeta.compute(),
            self.recall.compute(),
            self.precision.compute(),
            self.auroc.compute(),
            own_average_precision,
        )

    def eval_model_regression(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.0
            for data in data_loader:
                data.to(self.device)
                labels = data.y
                output = torch.squeeze(
                    self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                )
                # Check if mask exists and apply it
                if hasattr(data, 'mask'):
                    mask = data.mask
                    output = output[mask]
                    labels = labels[mask]
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                self.r2_score.update(output, labels)
                self.mae_score.update(output, labels)
                self.mse_score.update(output, labels)
        return loss/data.y.shape[0], self.r2_score.compute(), self.mae_score.compute(), self.mse_score.compute()

    def eval_model_classification_basic(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.0
            for data in data_loader:
                data.to(self.device)
                output = torch.squeeze(
                    self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                )
                labels = torch.squeeze(data.y)
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                self.f1_score.update(output, labels)
                self.fbeta.update(output, labels)
                self.accuracy.update(output, labels)
                self.recall.update(output, labels)
                self.precision.update(output, labels)
                self.auroc.update(output, labels)
                self.average_precision.update(output, labels)
        return (
            loss/data.y.shape[0],
            self.accuracy.compute(),
            self.f1_score.compute(),
            self.fbeta.compute(),
            self.recall.compute(),
            self.precision.compute(),
            self.auroc.compute(),
            self.average_precision.compute(),
        )
    
    def eval_model_classification_mask_nans(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.0
            for data in data_loader:
                data.to(self.device)
                output = torch.squeeze(
                    self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                )
                labels = torch.squeeze(data.y)
                mask = torch.isnan(labels)
                labels = labels[~mask]
                output = output[~mask]
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                self.f1_score.update(output, labels)
                self.fbeta.update(output, labels)
                self.accuracy.update(output, labels)
                self.recall.update(output, labels)
                self.precision.update(output, labels)
                self.auroc.update(output, labels)
                self.average_precision.update(output, labels)
        return (
            loss/data.y.shape[0],
            self.accuracy.compute(),
            self.f1_score.compute(),
            self.fbeta.compute(),
            self.recall.compute(),
            self.precision.compute(),
            self.auroc.compute(),
            self.average_precision.compute(),
        )
    
    def eval_model_molpcba(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            num_target_classes = data_loader.dataset[0].y.shape[1]
            all_labels = torch.Tensor(0,num_target_classes).to(self.device)
            all_outputs = torch.Tensor(0,num_target_classes).to(self.device)
            loss = 0.0
            for data in data_loader:
                data.to(self.device)
                output = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                labels = data.y
                mask = torch.isnan(labels)
                labels_masked = labels[~mask]
                output_masked = output[~mask]
                temp_loss = self.criterion(output_masked, labels_masked)
                loss += temp_loss.item()
                self.f1_score.update(output, labels)
                self.fbeta.update(output, labels)
                self.accuracy.update(output, labels)
                self.recall.update(output, labels)
                self.precision.update(output, labels)
                self.auroc.update(output, labels)
                # self.average_precision.update(output, labels)
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_outputs = torch.cat([all_outputs, output], dim=0)
            input_dict = {'y_true': all_labels, 'y_pred': all_outputs}
            own_average_precision = self.evaluator.eval(input_dict)["ap"]
        return (
            loss/data.y.shape[0],
            self.accuracy.compute(),
            self.f1_score.compute(),
            self.fbeta.compute(),
            self.recall.compute(),
            self.precision.compute(),
            self.auroc.compute(),
            own_average_precision,
        )

    def train_epoch_regression_hetero(self, data_loader):
        self.model.train()
        loss = 0.0
        all_labels = torch.IntTensor(0).to(self.device)
        all_predictions = torch.Tensor(0).to(self.device)
        for _, (data) in enumerate(data_loader):
            data.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(
                data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict
            )
            predictions = torch.squeeze(
                torch.cat([predictions["load"], predictions["normalForm"]])
            )
            labels = torch.cat([data["load"].y, data["normalForm"].y])
            temp_loss = self.criterion(predictions, labels.float())
            temp_loss.backward()
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping_max_norm
                )
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_predictions = torch.cat([all_predictions, predictions])
        R2 = self.r2_score(all_predictions, all_labels)
        self.scheduler_step(loss)
        return loss, R2.item()

    def eval_model_regression_hetero(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.0
            all_labels = torch.IntTensor(0).to(self.device)
            all_predictions = torch.Tensor(0).to(self.device)
            for data in data_loader:
                data.to(self.device)
                labels = torch.cat([data["load"].y, data["normalForm"].y])
                predictions = self.model(
                    data.x_dict,
                    data.edge_index_dict,
                    data.edge_attr_dict,
                    data.batch_dict,
                )
                predictions = torch.squeeze(
                    torch.cat([predictions["load"], predictions["normalForm"]])
                )
                temp_loss = self.criterion(predictions, labels)
                loss += temp_loss.item()
                all_predictions = torch.cat([all_predictions, predictions])
                all_labels = torch.cat([all_labels, labels])
        R2 = self.r2_score(all_predictions, all_labels)
        return loss, R2.item()

    def train_eval_epoch_1grid_basic(self, dataset):
        self.model.train()
        # for _, (data) in enumerate(data_loader):
        data = dataset[0]
        data.to(self.device)
        self.optimizer.zero_grad()
        output = torch.squeeze(
            self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        )
        labels = data.y
        labels_train = labels[data.train_mask]
        output_train = output[data.train_mask]
        train_loss = self.criterion(output_train, labels_train)
        train_loss.backward()
        self.gradient_clipping(self.model)
        self.optimizer.step()

        self.f1_score.update(output_train, labels_train)
        train_f1 = self.f1_score.compute()
        self.fbeta.update(output_train, labels_train)
        train_fbeta = self.fbeta.compute()
        self.accuracy.update(output_train, labels_train)
        train_accu = self.accuracy.compute()
        self.recall.update(output_train, labels_train)
        train_recall = self.recall.compute()
        self.precision.update(output_train, labels_train)
        train_precision = self.precision.compute()
        self.auroc.update(output_train, labels_train)
        train_auroc = self.auroc.compute()
        self.average_precision.update(output_train, labels_train)
        train_average_precision = self.average_precision.compute()

        labels_valid = labels[data.val_mask]
        output_valid = output[data.val_mask]
        val_loss = self.criterion(output_valid, labels_valid) / len(
            output_valid
        )
        self.f1_score.update(output_valid, labels_valid)
        val_f1 = self.f1_score.compute()
        self.fbeta.update(output_valid, labels_valid)
        val_fbeta = self.fbeta.compute()
        self.accuracy.update(output_valid, labels_valid)
        val_accu = self.accuracy.compute()
        self.recall.update(output_valid, labels_valid)
        val_recall = self.recall.compute()
        self.precision.update(output_valid, labels_valid)
        val_precision = self.precision.compute()
        self.auroc.update(output_valid, labels_valid)
        val_auroc = self.auroc.compute()
        self.average_precision.update(output_valid, labels_valid)
        val_average_precision = self.average_precision.compute()

        labels_test = labels[data.test_mask]
        output_test = output[data.test_mask]
        val_loss = self.criterion(output_test, labels_test) / len(
            output_test
        )
        test_loss = self.criterion(
            output_test, labels_test
        ) / len(output_test)
        self.f1_score.update(output_test, labels_test)
        test_f1 = self.f1_score.compute()
        self.fbeta.update(output_test, labels_test)
        test_fbeta = self.fbeta.compute()
        self.accuracy.update(output_test, labels_test)
        test_accu = self.accuracy.compute()
        self.recall.update(output_test, labels_test)
        test_recall = self.recall.compute()
        self.precision.update(output_test, labels_test)
        test_precision = self.precision.compute()
        self.auroc.update(output_test, labels_test)
        test_auroc = self.auroc.compute()
        self.average_precision.update(output_test, labels_test)
        test_average_precision = self.average_precision.compute()

        self.scheduler_step(train_loss)
        return (
            (train_loss / len(output_train)).item(),
            train_f1,
            train_fbeta,
            train_accu,
            train_recall,
            train_precision,
            train_auroc,
            train_average_precision,
            val_loss.item(),
            val_f1,
            val_fbeta,
            val_accu,
            val_recall,
            val_precision,
            val_auroc,
            val_average_precision,
            test_loss.item(),
            test_f1,
            test_fbeta,
            test_accu,
            test_recall,
            test_precision,
            test_auroc,
            test_average_precision,
        )


    def train_eval_epoch_1grid_mask_nans(self, dataset):
        self.model.train()
        # for _, (data) in enumerate(data_loader):
        data = dataset[0]
        data.to(self.device)
        self.optimizer.zero_grad()
        output = torch.squeeze(
            self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        )
        labels = data.y
        labels_train = labels[data.train_mask]
        output_train = output[data.train_mask]
        train_mask_nan = torch.isnan(labels_train)
        labels_train = labels_train[~train_mask_nan]
        output_train = output_train[~train_mask_nan]
        train_loss = self.criterion(output_train, labels_train)
        train_loss.backward()
        self.gradient_clipping(self.model)
        self.optimizer.step()

        self.f1_score.update(output_train, labels_train)
        train_f1 = self.f1_score.compute()
        self.fbeta.update(output_train, labels_train)
        train_fbeta = self.fbeta.compute()
        self.accuracy.update(output_train, labels_train)
        train_accu = self.accuracy.compute()
        self.recall.update(output_train, labels_train)
        train_recall = self.recall.compute()
        self.precision.update(output_train, labels_train)
        train_precision = self.precision.compute()
        self.auroc.update(output_train, labels_train)
        train_auroc = self.auroc.compute()
        self.average_precision.update(output_train, labels_train)
        train_average_precision = self.average_precision.compute()

        labels_valid = labels[data.val_mask]
        output_valid = output[data.val_mask]
        valid_mask_nan = torch.isnan(labels_valid)
        labels_valid = labels_valid[~valid_mask_nan]
        output_valid = output_valid[~valid_mask_nan]
        val_loss = self.criterion(output_valid, labels_valid) / len(
            output_valid
        )
        self.f1_score.update(output_valid, labels_valid)
        val_f1 = self.f1_score.compute()
        self.fbeta.update(output_valid, labels_valid)
        val_fbeta = self.fbeta.compute()
        self.accuracy.update(output_valid, labels_valid)
        val_accu = self.accuracy.compute()
        self.recall.update(output_valid, labels_valid)
        val_recall = self.recall.compute()
        self.precision.update(output_valid, labels_valid)
        val_precision = self.precision.compute()
        self.auroc.update(output_valid, labels_valid)
        val_auroc = self.auroc.compute()
        self.average_precision.update(output_valid, labels_valid)
        val_average_precision = self.average_precision.compute()

        labels_test = labels[data.test_mask]
        output_test = output[data.test_mask]
        test_mask_nan = torch.isnan(labels_test)
        labels_test = labels_test[~test_mask_nan]
        output_test = output_test[~test_mask_nan]
        val_loss = self.criterion(output_test, labels_test) / len(
            output_test
        )
        test_loss = self.criterion(
            output_test, labels_test
        ) / len(output_test)
        self.f1_score.update(output_test, labels_test)
        test_f1 = self.f1_score.compute()
        self.fbeta.update(output_test, labels_test)
        test_fbeta = self.fbeta.compute()
        self.accuracy.update(output_test, labels_test)
        test_accu = self.accuracy.compute()
        self.recall.update(output_test, labels_test)
        test_recall = self.recall.compute()
        self.precision.update(output_test, labels_test)
        test_precision = self.precision.compute()
        self.auroc.update(output_test, labels_test)
        test_auroc = self.auroc.compute()
        self.average_precision.update(output_test, labels_test)
        test_average_precision = self.average_precision.compute()

        self.scheduler_step(train_loss)
        return (
            (train_loss / len(output_train)).item(),
            train_f1,
            train_fbeta,
            train_accu,
            train_recall,
            train_precision,
            train_auroc,
            train_average_precision,
            val_loss.item(),
            val_f1,
            val_fbeta,
            val_accu,
            val_recall,
            val_precision,
            val_auroc,
            val_average_precision,
            test_loss.item(),
            test_f1,
            test_fbeta,
            test_accu,
            test_recall,
            test_precision,
            test_auroc,
            test_average_precision,
        )

    def aggregate_list_from_config(self, config, key_word, index_start, index_end):
        new_list = [config[key_word + str(index_start)]]
        for i in range(index_start + 1, index_end + 1):
            index_name = key_word + str(i)
            new_list.append(config[index_name])
        return new_list

    def make_list_number_of_channels(self, config):
        key_word = "num_channels"
        index_start = 1
        index_end = config["num_layers"] + 1
        num_channels = self.aggregate_list_from_config(
            config, key_word, index_start, index_end
        )
        return num_channels

    def make_list_Tag_hops(self, config):
        key_word = "TAG::K_hops"
        index_start = 1
        index_end = config["num_layers"]
        list_k_hops = self.aggregate_list_from_config(
            config, key_word, index_start, index_end
        )
        return list_k_hops

    def make_list_Arma_internal_stacks(self, config):
        key_word = "ARMA::num_internal_stacks"
        index_start = 1
        index_end = config["num_layers"]
        list_internal_stacks = self.aggregate_list_from_config(
            config, key_word, index_start, index_end
        )
        return list_internal_stacks

    def make_list_Arma_internal_layers(self, config):
        key_word = "ARMA::num_internal_layers"
        index_start = 1
        index_end = config["num_layers"]
        list_internal_layers = self.aggregate_list_from_config(
            config, key_word, index_start, index_end
        )
        return list_internal_layers
