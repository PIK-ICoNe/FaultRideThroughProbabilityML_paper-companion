import numpy as np

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



from torch_geometric.nn import GCNConv, ARMAConv, SAGEConv, TAGConv, TransformerConv, GATv2Conv
from torch_geometric.nn import Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import to_hetero


from torchmetrics import F1Score, FBetaScore, Recall, Precision, R2Score, Accuracy

class ArmaNet_bench(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_layers=4, num_stacks=3, final_sigmoid_layer=True):
        super(ArmaNet_bench, self).__init__()
        self.conv1 = ARMAConv(num_node_features, 16, num_stacks=num_stacks,
                              num_layers=num_layers, shared_weights=True, dropout=0.25)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2 = ARMAConv(16, num_classes, num_stacks=num_stacks,
                              num_layers=num_layers, shared_weights=True, dropout=0.25, act=None)
        self.conv2_bn = nn.BatchNorm1d(num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.final_sigmoid_layer = final_sigmoid_layer
        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, batch):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x=x, edge_index=edge_index,
                       edge_weight=edge_weight.float())
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index=edge_index,
                       edge_weight=edge_weight.float())
        # x = self.endLinear(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.endLinear.reset_parameters()


class GATModel(nn.Module):
    def __init__(self, num_layers, hidden_channels, heads, dropout, add_self_loops,  edge_dim, v2, linear_layer_after_conv, ll_after_conv_dim, final_linear_layer, final_sigmoid_layer, hetero=False):
        super(GATModel, self).__init__()
        self.linear_layer_after_conv = linear_layer_after_conv
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer
        self.convlist = nn.ModuleList()
        if hetero == "hetero":
            self.hetero=True
        else:
            self.hetero = False
        
        
        for i in range(0, num_layers):
            if v2 == True:
                conv = GATv2Conv((-1,-1), out_channels=hidden_channels, heads = heads, dropout = dropout, add_self_loops=add_self_loops, edge_dim=edge_dim)
            else:
                conv = GATConv((-1,-1), out_channels=hidden_channels, heads = heads, dropout = dropout, add_self_loops=add_self_loops, edge_dim=edge_dim)
            self.convlist.append(conv)
        if v2 == True:
            conv = GATv2Conv((-1,-1), out_channels=hidden_channels, heads = 1, dropout = dropout, add_self_loops=add_self_loops, edge_dim=edge_dim)
        else:
            conv = GATConv((-1,-1), out_channels=hidden_channels, heads = 1, dropout = dropout, add_self_loops=add_self_loops, edge_dim=edge_dim)
        self.convlist.append(conv)
        if linear_layer_after_conv == True:
            self.ll_after_conv = nn.Linear(hidden_channels, ll_after_conv_dim)
        if final_linear_layer == True:
            if linear_layer_after_conv == True:
                self.final_ll = nn.Linear(ll_after_conv_dim,1)
            else:
                self.final_ll = nn.Linear(hidden_channels,1)
        if final_sigmoid_layer:
            self.sigmoid_layer = nn.Sigmoid()
        
    def forward(self, x, edge_index, edge_attr, batch):
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](x, edge_index, edge_attr)
        if self.linear_layer_after_conv:
            x = self.ll_after_conv(x)
        if self.final_linear_layer == True:
            x = self.final_ll(x)
        if self.final_sigmoid_layer:
            x = self.sigmoid_layer(x)
        return x
    # def forward(self, data):
    #     if self.hetero == True:
    #         x, edge_index, edge_attr = data.x_dict, data.edge_index_dict, data.edge_attr_dict
    #     else:
    #         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    #     for i, _ in enumerate(self.convlist):
    #         x = self.convlist[i](x, edge_index, edge_attr)
    #     if self.linear_layer_after_conv:
    #         x = self.ll_after_conv(x)
    #     if self.final_linear_layer == True:
    #         x = self.final_ll(x)
    #     if self.final_sigmoid_layer:
    #         x = self.sigmoid_layer(x)
    #     return x



class TAGConvModule(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, activation, K, dropout):
        super(TAGConvModule, self).__init__()
        self.activation = activation
        self.conv = TAGConv(input_channels, hidden_channels, K = K)
        # self.conv = TAGConv(-1, hidden_channels, K = K)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv(x, edge_index=edge_index)#,
                      #edge_weight=edge_weight.float())
        x = self.dropout(x)
        if self.activation == "ReLU":
            return F.relu(x)
        elif self.activation == None:
            return x
            
class TAGModel(nn.Module):
    def __init__(self, num_layers, hidden_channels, K, activation, dropout, linear_layer_after_conv, ll_after_conv_dim, final_linear_layer, final_sigmoid_layer):
        super(TAGModel, self).__init__()
        self.dropout = dropout
        self.final_linear_layer = final_linear_layer
        self.final_sigmoid_layer = final_sigmoid_layer
        self.convlist = nn.ModuleList()
        conv = TAGConvModule(8, hidden_channels, activation,K, dropout)
        self.convlist.append(conv)
        for i in range(1, num_layers):
            conv = TAGConvModule(hidden_channels, hidden_channels, activation,K, dropout)
            self.convlist.append(conv)

        if final_linear_layer == True:
            self.linear_layer = nn.Linear(hidden_channels,1)
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


class GNNmodule(nn.Module):
    def __init__(self, config, criterion_positive_weight=False, config_ray = False):
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
                self.critierion_positive_weight).to(self.device)

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
        if model_name == "ArmaNet_bench":
            model = ArmaNet_bench(
                final_sigmoid_layer= final_sigmoid_layer)
        elif model_name == "GAT":
            model = GATModel(config["num_layers"], config["hidden_channels"], config["heads"], config["dropout_n"], config["add_self_loops"],  config["edge_dim"], config["GAT::v2"], config["linear_layer_after_conv"], config["ll_after_conv_dim"], config["final_linear_layer"], config["final_sigmoid_layer"], config["grid_type"])
        elif model_name == "TAG":
            model = TAGModel(config["num_layers"], config["hidden_channels"], config["TAG::K"], config["activation"], config["dropout_n"], config["linear_layer_after_conv"], config["ll_after_conv_dim"], config["final_linear_layer"], config["final_sigmoid_layer"])
        else:
            print("error: model type unkown")
        if self.grid_type == "hetero":
            data = config["hetero::datasample"]
            # model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            model = to_hetero(model, data.metadata(), aggr='sum')
        # model.double()
        if config["model_name"] != "GAT":
            # model = torch.compile(model)
            print("did not precompile model")
        if self.dtype=="float64":
            model.double()
        if self.dtype=="float16":
            model.half()

        model.to(self.device)

        self.model = model


        # criterion
        if config["criterion"] == "MSELoss":
            if criterion_positive_weight == True:
                self.criterion = nn.MSELoss(reduction="none")
            else:
                self.criterion = nn.MSELoss()
        if config["criterion"] == "BCEWithLogitsLoss":
            if criterion_positive_weight == False:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(criterion_positive_weight))
                print("positive_weigt used for criterion: ",
                      criterion_positive_weight)
        if config["criterion"] == "BCELoss":
            self.criterion = nn.BCELoss()
        if config["criterion"] == "CELoss":
            self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        # set opimizer
        if config["optim::optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config["optim::LR"], momentum=config["optim::momentum"], weight_decay=config["optim::weight_decay"])
        if config["optim::optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(
            ), lr=config["optim::LR"], weight_decay=config["optim::weight_decay"])
        self.optimizer = optimizer

        # scheduler
        scheduler_name = config["optim::scheduler"]
        self.scheduler_name = scheduler_name
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=config["optim::ReducePlat_patience"], factor=config["optim::LR_reduce_factor"])
        elif scheduler_name == "stepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config["optim::stepLR_step_size"], gamma=config["optim::LR_reduce_factor"])
        elif scheduler_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=.1, last_epoch=-1)
        elif scheduler_name == "OneCycleLR":
            steps_per_epoch = config_ray["len_trainloader"]
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["optim::max_LR"], steps_per_epoch=steps_per_epoch, epochs=config["epochs"], div_factor = config["optim::div_factor"], anneal_strategy = config["optim::anneal_strategy"])
        elif scheduler_name == "None":
            scheduler = None
        elif scheduler_name == None:
            scheduler = None
        self.scheduler = scheduler

        self.gradient_clipping = config["gradient_clipping"]
        self.gradient_clipping_max_norm = config["gradient_clipping_::max_norm"]

        # metrics
        self.init_metrics(config)


    def init_metrics(self,config):
        num_classes = config["num_classes"]
        if num_classes > 1:
            multiclass = True
        else:
            multiclass = False
        # self.accuracy = Accuracy(task="multiclass", num_classes = num_classes).to(self.device)
        if config["task_type"] == "classification":
            self.accuracy = Accuracy().to(self.device)
            self.f1_score = F1Score(multiclass = multiclass, num_classes=num_classes).to(self.device)
            self.fbeta = FBetaScore(multiclass = multiclass, num_classes=num_classes, beta=self.beta).to(self.device)    
            self.recall = Recall(multiclass = multiclass, num_classes=num_classes).to(self.device)
            self.precision = Precision(multiclass = multiclass, num_classes=num_classes).to(self.device)

        if config["task_type"] == "regression":
            self.r2_score = R2Score().to(self.device)

        if config["task_type"] == "regressionThresholding":
            self.accuracy = Accuracy().to(self.device)
            self.r2_score = R2Score().to(self.device)
            self.f1_score = F1Score(multiclass = False).to(self.device)
            self.fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)    
            self.recall = Recall(multiclass=False).to(self.device)
            self.precision = Precision(multiclass=False).to(self.device)

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

    def train_epoch_regression(self, data_loader, threshold):
        self.model.train()
        loss = 0.
        correct = 0
        mse_trained = 0.
        all_labels = torch.IntTensor(0).to(self.device)
        all_predictions = torch.Tensor(0).to(self.device)
        for _, (data) in enumerate(data_loader):
            data.to(self.device)
            self.optimizer.zero_grad()
            output = torch.squeeze(self.model(data.x, data.edge_index, data.edge_attr, data.batch))
            labels = data.y
            temp_loss = self.criterion(output, labels.float())
            temp_loss.backward()
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clipping_max_norm)
            self.optimizer.step()
            correct += torch.sum((torch.abs(output - labels) < threshold))
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_predictions = torch.cat([all_predictions, output])
        R2 = self.r2_score(all_predictions, all_labels)
        # accuracy
        accuracy = 100 * correct / all_labels.shape[0]
        self.scheduler_step(loss)
        return loss, accuracy.item(), R2.item()

    def train_epoch_classification(self, data_loader):
        self.model.train()
        loss = 0.
        correct = 0
        all_labels = torch.IntTensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for _, (data) in enumerate(data_loader):
            data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.edge_attr, data.batch).squeeze()
            labels = data.y
            temp_loss = self.criterion(output, labels.float())
            temp_loss.backward()
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clipping_max_norm)
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_outputs = torch.cat([all_outputs, output])
        f1 = self.f1_score(all_outputs, all_labels)
        fbeta = self.fbeta(all_outputs, all_labels)
        accu = self.accuracy(all_outputs, all_labels)
        recall = self.recall(all_outputs, all_labels)
        precision = self.precision(all_outputs, all_labels)
        self.scheduler_step(loss)
        return loss, accu.item(), f1.item(), fbeta.item(), recall.item(), precision.item()

    def eval_model_regression(self, data_loader, threshold):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            mse_trained = 0.
            all_labels = torch.IntTensor(0).to(self.device)
            all_predictions = torch.Tensor(0).to(self.device)
            for data in data_loader:
                data.to(self.device)
                labels = data.y
                output = torch.squeeze(self.model(data.x, data.edge_index, data.edge_attr, data.batch))
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                correct += torch.sum((torch.abs(output - labels) < threshold))
                all_predictions = torch.cat([all_predictions, output])
                all_labels = torch.cat([all_labels, labels])
            accuracy = 100 * correct / all_labels.shape[0]
        R2 = self.r2_score(all_predictions, all_labels)
        return loss, accuracy.item(), R2.item()

    def eval_model_classification(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            all_labels = torch.IntTensor(0).to(self.device)
            all_outputs = torch.Tensor(0).to(self.device)
            for data in data_loader:
                data.to(self.device)
                labels = data.y
                output = self.model(data.x, data.edge_index, data.edge_attr, data.batch).squeeze()
                temp_loss = self.criterion(output, labels.long())
                loss += temp_loss.item()
                all_labels = torch.cat([all_labels, labels])
                all_outputs = torch.cat([all_outputs, output])
        f1 = self.f1_score(all_outputs, all_labels)
        fbeta = self.fbeta(all_outputs, all_labels)
        accuracy = self.accuracy(all_outputs, all_labels)
        recall = self.recall(all_outputs, all_labels)
        precision = self.precision(all_outputs, all_labels)
        return loss, accuracy.item(), f1.item(), fbeta.item(), recall.item(), precision.item()

    def train_epoch_regression_hetero(self, data_loader, threshold):
        self.model.train()
        loss = 0.
        correct = 0
        mse_trained = 0.
        all_labels = torch.IntTensor(0).to(self.device)
        all_predictions = torch.Tensor(0).to(self.device)
        for _, (data) in enumerate(data_loader):
            data.to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model(
                data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict)
            # outputs = self.model.forward(batch)
            predictions = torch.squeeze(
                torch.cat([predictions["load"], predictions["normalForm"]]))
            labels = torch.cat([data["load"].y, data["normalForm"].y])
            temp_loss = self.criterion(predictions, labels.float())
            temp_loss.backward()
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clipping_max_norm)
            self.optimizer.step()
            correct += torch.sum((torch.abs(predictions - labels) < threshold))
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_predictions = torch.cat([all_predictions, predictions])
        R2 = self.r2_score(all_predictions, all_labels)
        # accuracy
        accuracy = 100 * correct / all_labels.shape[0]
        self.scheduler_step(loss)
        return loss, accuracy.item(), R2.item()

    def eval_model_regression_hetero(self, data_loader, threshold):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            mse_trained = 0.
            all_labels = torch.IntTensor(0).to(self.device)
            all_predictions = torch.Tensor(0).to(self.device)
            for data in data_loader:
                data.to(self.device)
                labels = torch.cat([data["load"].y, data["normalForm"].y])
                predictions = self.model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict)
                predictions = torch.squeeze(
                torch.cat([predictions["load"], predictions["normalForm"]]))
                temp_loss = self.criterion(predictions, labels)
                loss += temp_loss.item()
                correct += torch.sum((torch.abs(predictions - labels) < threshold))
                all_predictions = torch.cat([all_predictions, predictions])
                all_labels = torch.cat([all_labels, labels])
            accuracy = 100 * correct / all_labels.shape[0]
        R2 = self.r2_score(all_predictions, all_labels)
        return loss, accuracy.item(), R2.item()


    def aggregate_list_from_config(self, config, key_word, index_start, index_end):
        new_list = [config[key_word+str(index_start)]]
        for i in range(index_start+1, index_end+1):
            index_name = key_word + str(i)
            new_list.append(config[index_name])
        return new_list

    def make_list_number_of_channels(self, config):
        key_word = "num_channels"
        index_start = 1
        index_end = config["num_layers"] + 1
        num_channels = self.aggregate_list_from_config(
            config, key_word, index_start, index_end)
        return num_channels

    def make_list_Tag_hops(self, config):
        key_word = "TAG::K_hops"
        index_start = 1
        index_end = config["num_layers"]
        list_k_hops = self.aggregate_list_from_config(
            config, key_word, index_start, index_end)
        return list_k_hops

    def make_list_Arma_internal_stacks(self, config):
        key_word = "ARMA::num_internal_stacks"
        index_start = 1
        index_end = config["num_layers"]
        list_internal_stacks = self.aggregate_list_from_config(
            config, key_word, index_start, index_end)
        return list_internal_stacks

    def make_list_Arma_internal_layers(self, config):
        key_word = "ARMA::num_internal_layers"
        index_start = 1
        index_end = config["num_layers"]
        list_internal_layers = self.aggregate_list_from_config(
            config, key_word, index_start, index_end)
        return list_internal_layers
