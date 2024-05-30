
from torchmetrics import F1Score, FBetaScore, Recall, Precision, R2Score

from sklearn.preprocessing import StandardScaler


import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class HiddenLayerModule(torch.nn.Module):
    def __init__(self, dim_in, dim_out, activation, dropout=False):
        super(HiddenLayerModule, self).__init__()
        self.activation = activation
        self.layer = nn.Linear(dim_in, dim_out)
        self.activation = activation
        if dropout != False:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = F.relu(x)
        if self.dropout != False:
            x = self.dropout(x)
        return x


class MLP01(torch.nn.Module):
    def __init__(self, num_classes=1, num_input_features=6, num_hidden_layers=1, num_hidden_unit_per_layer=30, final_sigmoid_layer=False):
        super(MLP01, self).__init__()
        self.input_layer = nn.Linear(
            num_input_features, num_hidden_unit_per_layer)
        self.internal_layers = nn.ModuleList()

        for i in range(0, num_hidden_layers):
            hidden_layer = HiddenLayerModule(
                num_hidden_unit_per_layer, num_hidden_unit_per_layer, activation=True)
            self.internal_layers.append(hidden_layer)

        self.output_layer = nn.Linear(num_hidden_unit_per_layer, num_classes)

        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()
        self.final_sigmoid_layer = final_sigmoid_layer

    def forward(self, data):
        x = self.input_layer(data)
        x = F.relu(x)

        for i, _ in enumerate(self.internal_layers):
            x = self.internal_layers[i](x)
        x = self.output_layer(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class MLP02(torch.nn.Module):
    def __init__(self, num_classes=1, num_input_features=6, num_hidden_layers=1, num_hidden_unit_per_layer=30, dropout=.5, final_sigmoid_layer=False):
        super(MLP02, self).__init__()
        self.input_layer = nn.Linear(
            num_input_features, num_hidden_unit_per_layer)
        self.internal_layers = nn.ModuleList()

        for i in range(0, num_hidden_layers):
            hidden_layer = HiddenLayerModule(
                num_hidden_unit_per_layer, num_hidden_unit_per_layer, activation=True, dropout=dropout)
            self.internal_layers.append(hidden_layer)

        self.output_layer = nn.Linear(num_hidden_unit_per_layer, num_classes)

        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()
        self.final_sigmoid_layer = final_sigmoid_layer

    def forward(self, data):
        x = self.input_layer(data)
        x = F.relu(x)

        for i, _ in enumerate(self.internal_layers):
            x = self.internal_layers[i](x)
        x = self.output_layer(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class MLPModule(nn.Module):
    def __init__(self, config, criterion_positive_weight=False):
        super(MLPModule, self).__init__()
        cuda = config["cuda"]
        if "use_dtype_double" in config.keys():
            use_dtype_double = config["use_dtype_double"]
        else:
            use_dtype_double = True
        if "Fbeta::beta" in config:
            self.beta = config["Fbeta::beta"]
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.cuda = True
            print("cuda available:: send model to GPU")
        elif cuda and use_dtype_double == False:
            if torch.backends.mps.is_available() and config["use_dtype_double"] == False:
                self.device = torch.device("mps")
                print("cuda available:: send model to GPU (MPS)")
        else:
            self.cuda = False
            self.device = torch.device("cpu")
            print("cuda unavailable:: train model on cpu")

        # seeds
        torch.manual_seed(config["manual_seed"])
        torch.cuda.manual_seed(config["manual_seed"])
        np.random.seed(config["manual_seed"])
        if self.cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # set model
        num_classes = config["MLP::num_classes"]
        num_input_features = config["MLP::num_input_features"]
        num_hidden_layers = config["MLP::num_hidden_layers"]
        num_hidden_unit_per_layer = config["MLP::num_hidden_unit_per_layer"]
        final_sigmoid_layer = config["final_sigmoid_layer"]
        if config["model_name"] == "MLP" or config["model_name"] == "MLP01":
            model = MLP01(num_classes, num_input_features, num_hidden_layers,
                          num_hidden_unit_per_layer, final_sigmoid_layer)
        elif config["model_name"] == "MLP02":
            model = MLP02(num_classes, num_input_features, num_hidden_layers,
                          num_hidden_unit_per_layer, config["MLP::dropout"], final_sigmoid_layer)
        if use_dtype_double == True:
            model.double()
        model.to(self.device)

        self.model = model

        # criterion
        if config["criterion"] == "MSELoss":
            self.criterion = nn.MSELoss()
        if config["criterion"] == "BCEWithLogitsLoss":
            if criterion_positive_weight == False:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(criterion_positive_weight))
                print("positive_weigt used for criterion: ",
                      criterion_positive_weight.item())
        if config["criterion"] == "BCELoss":
            self.criterion = nn.BCELoss()
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
        elif scheduler_name == "None":
            scheduler = None
        elif scheduler_name == None:
            scheduler = None
        self.scheduler = scheduler

    def forward(self, x):
        y = self.model(x)
        return y

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
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
        # scheduler = self.scheduler
        loss = 0.
        correct = 0
        mse_trained = 0.
        all_labels = torch.Tensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for iter, (input_data, labels) in enumerate(data_loader):
            input_data = input_data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            output = torch.squeeze(self.model.forward(input_data))
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            correct += torch.sum((torch.abs(output - labels) < threshold))
            loss += temp_loss.item()
            mse_trained += torch.sum((output - labels) ** 2)
            all_labels = torch.cat([all_labels, labels])
            all_outputs = torch.cat([all_outputs, output])
        r2score = R2Score().to(self.device)
        R2 = r2score(all_outputs, all_labels)
        # accuracy
        accuracy = 100 * correct / all_labels.flatten().shape[0]
        self.scheduler_step(loss)
        return loss, accuracy, R2.item()

    def train_epoch_regressionThresholding(self, data_loader):
        self.model.train()
        # scheduler = self.scheduler
        loss = 0.
        mse_trained = 0.
        all_labels = torch.Tensor(0).to(self.device)
        all_labels_threshold = torch.Tensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for iter, (input_data, labels, labels_threshold) in enumerate(data_loader):
            input_data = input_data.to(self.device)
            labels = labels.to(self.device)
            labels_threshold = labels_threshold.to(self.device)
            self.optimizer.zero_grad()
            output = torch.squeeze(self.model.forward(input_data))
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_labels_threshold = torch.cat(
                [all_labels_threshold, labels_threshold])
            all_outputs = torch.cat([all_outputs, output])
        r2score = R2Score().to(self.device)
        R2 = r2score(all_outputs, all_labels)

        all_labels_threshold = all_labels_threshold.int()
        outputs_classification = torch.where(all_outputs < 15., 0., 1.)
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(outputs_classification, all_labels_threshold)
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(outputs_classification, all_labels_threshold)
        self.scheduler_step(loss)
        return loss, R2.item(), fbeta.item(), recall.item()

    def train_epoch_classification(self, data_loader):
        self.model.train()
        loss = 0.
        correct = 0
        all_labels = torch.Tensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for _, (input_data, labels) in enumerate(data_loader):
            input_data = input_data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            output = torch.squeeze(self.model.forward(input_data))
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_outputs = torch.cat([all_outputs, output])
        sigmoid_layer = torch.nn.Sigmoid().to(self.device)
        all_outputs_bin = sigmoid_layer(all_outputs) > .5
        correct += (all_outputs_bin == all_labels).sum()
        f1 = F1Score(multiclass=False).to(self.device)
        f1 = f1(all_outputs_bin, all_labels.int())
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(all_outputs_bin, all_labels.int())
        accuracy = 100 * correct / all_labels.flatten().shape[0]
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(all_outputs_bin, all_labels.int())
        precision = Precision(multiclass=False)
        precision = precision.to(self.device)
        precision = precision(all_outputs_bin, all_labels.int())
        self.scheduler_step(loss)
        return loss, accuracy, f1.item(), fbeta.item(), recall.item(), precision.item()

    def eval_model_regression(self, data_loader, threshold):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            all_outputs = torch.Tensor(0).to(self.device)
            for _, (input_data, labels) in enumerate(data_loader):
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)
                output = torch.squeeze(self.model(input_data))
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                correct += torch.sum((torch.abs(output - labels) < threshold))
                mse_trained += torch.sum((output - labels) ** 2)
                all_labels = torch.cat([all_labels, labels])
                all_outputs = torch.cat([all_outputs, output])
            accuracy = 100 * correct / all_labels.flatten().shape[0]

        r2score = R2Score().to(self.device)
        R2 = r2score(all_outputs, all_labels)
        return loss, accuracy, R2.item()

    def eval_model_regressionThresholding(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            all_labels_threshold = torch.Tensor(0).to(self.device)
            all_outputs = torch.Tensor(0).to(self.device)
            for iter, (input_data, labels, labels_threshold) in enumerate(data_loader):
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)
                labels_threshold = labels_threshold.to(self.device)
                output = torch.squeeze(self.model(input_data))
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                mse_trained += torch.sum((output - labels) ** 2)
                all_labels = torch.cat([all_labels, labels])
                all_labels_threshold = torch.cat(
                    [all_labels_threshold, labels_threshold])
                all_outputs = torch.cat([all_outputs, output])
        r2score = R2Score().to(self.device)
        R2 = r2score(all_outputs, all_labels)

        all_labels_threshold = all_labels_threshold.int()
        outputs_classification = torch.where(all_outputs < 15., 0., 1.)
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(outputs_classification, all_labels_threshold)
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(outputs_classification, all_labels_threshold)
        self.scheduler_step(loss)

        return loss, R2.item(), fbeta.item(), recall.item()

    def eval_model_classification(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            all_outputs = torch.Tensor(0).to(self.device)
            for iter, (input_data, labels) in enumerate(data_loader):
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)
                output = torch.squeeze(self.model(input_data))
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                all_labels = torch.cat([all_labels, labels])
                all_outputs = torch.cat([all_outputs, output])
        sigmoid_layer = torch.nn.Sigmoid().to(self.device)
        all_outputs_bin = sigmoid_layer(all_outputs) > .5
        correct += (all_outputs_bin == all_labels).sum()
        f1 = F1Score(multiclass=False).to(self.device)
        f1 = f1(all_outputs_bin, all_labels.int())
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(all_outputs_bin, all_labels.int())
        accuracy = 100 * correct / all_labels.flatten().shape[0]
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(all_outputs_bin, all_labels.int())
        precision = Precision(multiclass=False)
        precision = precision.to(self.device)
        precision = precision(all_outputs_bin, all_labels.int())
        return loss, accuracy, f1.item(), fbeta.item(), recall.item(), precision.item()
