from ray.tune import Trainable

from torch_geometric.loader import DataLoader

from gnn_models import GNNmodule
from surv_nf import init_surv_nf_dataset


import torch
import sys
import json
from pathlib import Path


class NN_tune_trainable(Trainable):
    def setup(self, config):
        print("task_type: ", config["task_type"])
        self.config = config
        self.seed = config["manual_seed"]
        self.cuda = config["cuda"]
        self.task_type = config["task_type"]
        task_type = self.task_type
        self.grid_type = config["grid_type"]

        if self.cuda and torch.cuda.is_available():
            pin_memory = True
        else:
            pin_memory = False

        if "num_workers" in config.keys():
            num_workers = config["num_workers"]
        else:
            num_workers = 1

        # data set
        if config["dataset_name"] == "surv_nf":
            self.train_set, self.valid_set, self.test_set = init_surv_nf_dataset(config)
        if config["ieee_eval"]:
            self.ieee_set = init_surv_nf_dataset(config,ieee=True)
            self.ieee_loader = DataLoader(self.ieee_set, batch_size=config["test_set::batchsize"], shuffle=config["test_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
        

        self.train_loader = DataLoader(
            self.train_set, batch_size=config["train_set::batchsize"], shuffle=config["train_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
        self.valid_loader = DataLoader(
            self.valid_set, batch_size=config["valid_set::batchsize"], shuffle=config["valid_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(
            self.test_set, batch_size=config["test_set::batchsize"], shuffle=config["test_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
        cfg_ray = {"len_trainloader": len(self.train_loader)}
        if "criterion::positive_weight" in config.keys():
            if config["criterion::positive_weight"] == True:
                train_set_positive_weight = self.train_set.positive_weight.clone().detach()
                self.NN = GNNmodule(
                    config, train_set_positive_weight.numpy(), cfg_ray)
            else:
                self.NN = GNNmodule(config, config_ray = cfg_ray)
        else:
            self.NN = GNNmodule(config, config_ray=cfg_ray)
    
    def step(self):
        if self.config["dataset_name"] == "surv_nf":
            return self.step_surv_nf()
        if self.task_type == "regression":
            return self.step_regression()
        elif self.task_type == "classification":
            return self.step_classification()
        elif self.task_type == "regressionThresholding":
            return self.step_regressionThresholding()

    def step_regression(self):
        threshold = self.config["eval::threshold"]
        # train
        loss_train, acc_train, R2_train = self.NN.train_epoch_regression(
            self.train_loader, threshold)
        # valid
        loss_valid, acc_valid, R2_valid = self.NN.eval_model_regression(
            self.valid_loader, threshold)
        # test
        loss_test, acc_test, R2_test = self.NN.eval_model_regression(
            self.test_loader, threshold)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "train_R2": R2_train,
            "test_loss": loss_test,
            "test_acc": acc_test,
            "test_R2": R2_test,
            "valid_loss": loss_valid,
            "valid_acc": acc_valid,
            "valid_R2": R2_valid,
        }

        return result_dict

    def step_regressionThresholding(self):
        threshold = self.config["eval::threshold"]
        # train
        loss_train, R2_train, fbeta_train, recall_train = self.NN.train_epoch_regressionThresholding(
            self.train_loader)
        # valid
        loss_valid, R2_valid, fbeta_valid, recall_valid = self.NN.eval_model_regressionThresholding(
            self.valid_loader)
        # test
        loss_test, R2_test, fbeta_test, recall_test = self.NN.eval_model_regressionThresholding(
            self.test_loader)

        result_dict = {
            "train_loss": loss_train,
            "train_R2": R2_train,
            "train_fbeta": fbeta_train,
            "train_recall": recall_train,
            "valid_loss": loss_valid,
            "valid_R2": R2_valid,
            "valid_fbeta": fbeta_valid,
            "valid_recall": recall_valid,
            "test_loss": loss_test,
            "test_R2": R2_test,
            "test_fbeta": fbeta_test,
            "test_recall": recall_test,
        }

        return result_dict

    def step_classification(self):
        threshold = self.config["eval::threshold"]
        # train
        loss_train, acc_train, f1_train, fbeta_train, recall_train, precision_train = self.NN.train_epoch_classification(
            self.train_loader)
        # valid
        loss_valid, acc_valid, f1_valid, fbeta_valid, recall_valid, precision_valid = self.NN.eval_model_classification(
            self.valid_loader)
        # test
        loss_test, acc_test, f1_test, fbeta_test, recall_test, precision_test = self.NN.eval_model_classification(
            self.test_loader)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "train_f1": f1_train,
            "train_fbeta": fbeta_train,
            "train_recall": recall_train,
            "train_precision": precision_train,
            "valid_loss": loss_valid,
            "valid_acc": acc_valid,
            "valid_f1": f1_valid,
            "valid_fbeta": fbeta_valid,
            "valid_recall": recall_valid,
            "valid_precision": precision_valid,
            "test_loss": loss_test,
            "test_acc": acc_test,
            "test_f1": f1_test,
            "test_fbeta": fbeta_test,
            "test_recall": recall_test,
            "test_precision": precision_test
        }

        return result_dict

    def step_regression_nf_hetero(self):
        threshold = self.config["eval::threshold"]
        loss_train, acc_train, R2_train = self.NN.train_epoch_regression_hetero(
            self.train_loader, threshold)
        # valid
        loss_valid, acc_valid, R2_valid = self.NN.eval_model_regression_hetero(
            self.valid_loader, threshold)
        # test
        loss_test, acc_test, R2_test = self.NN.eval_model_regression_hetero(
            self.test_loader, threshold)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "train_R2": R2_train,
            "test_loss": loss_test,
            "test_acc": acc_test,
            "test_R2": R2_test,
            "valid_loss": loss_valid,
            "valid_acc": acc_valid,
            "valid_R2": R2_valid,
        }

        return result_dict



    def step_surv_nf(self):
        threshold = self.config["eval::threshold"]
        if self.grid_type == "homo":
            result_dict = self.step_regression()
        else:
            result_dict = self.step_regression_nf_hetero()
        if self.config["ieee_eval"]:
            if self.grid_type == "homo":
                loss_ieee, acc_ieee, R2_ieee = self.NN.eval_model_regression(self.ieee_loader, threshold)
            else:
                loss_ieee, acc_ieee, R2_ieee = self.NN.eval_model_regression_hetero(self.ieee_loader, threshold)
            result_dict["ieee_loss"] = loss_ieee
            result_dict["ieee_acc"] = acc_ieee
            result_dict["ieee_R2"] = R2_ieee
        return result_dict

        


    def save_checkpoint(self, experiment_dir):
        # save model state dict
        path = Path(experiment_dir).joinpath("model_state_dict")
        torch.save(self.NN.model.state_dict(), path)
        # save optimizer state dict
        path = Path(experiment_dir).joinpath("opt_state_dict")
        torch.save(self.NN.optimizer.state_dict(), path)
        # save scheduler state dict
        if self.NN.scheduler != None:
            path = Path(experiment_dir).joinpath("scheduler_state_dict")
            torch.save(self.NN.scheduler.state_dict(), path)

        return experiment_dir

    def load_checkpoint(self, experiment_dir):
        # load model state dict
        path = Path(experiment_dir).joinpath("model_state_dict")
        checkpoint = torch.load(path)
        self.NN.model.load_state_dict(checkpoint)
        # load optimizer state dict
        path = Path(experiment_dir).joinpath("opt_state_dict")
        checkpoint = torch.load(path)
        self.NN.optimizer.load_state_dict(checkpoint)
        # load scheduler state dict
        if self.NN.scheduler != None:
            path = Path(experiment_dir).joinpath("scheduler_state_dict")
            checkpoint = torch.load(path)
            self.NN.scheduler.load_state_dict(checkpoint)
