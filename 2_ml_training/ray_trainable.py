import ray.tune as tune
from ray.tune import Trainable

from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader

from gnn_models import GNNmodule
from surv_nf import init_surv_nf_dataset


import torch
import sys
import json
from pathlib import Path
import math


class NN_tune_trainable(Trainable):
    def setup(self, config):
        print("task_type: ", config["task_type"])
        self.config = config
        self.seed = config["manual_seed"]
        self.cuda = config["cuda"]
        self.task_type = config["task_type"]
        task_type = self.task_type
        self.grid_type = config["grid_type"]

        if "dataset::single_grid" not in config:
            self.use_single_grid = False
        else:
            self.use_single_grid = config["dataset::single_grid"]
       
        if "dataloader::neighbor_loader" not in config:
            self.dataloader_neighbor_loader = False
        elif config["dataloader::neighbor_loader"] == True:
            self.dataloader_neighbor_loader = True

        if "dataloader::neighbor_loader::num_neighbors" in config:
            num_neighbors = config["dataloader::neighbor_loader::num_neighbors"]
        if "dataloader::neighbor_loader::iterations" in config:
            num_iterations = config["dataloader::neighbor_loader::iterations"]

        # if self.cuda and torch.cuda.is_available():
        #     pin_memory = True
        # else:
        pin_memory = False

        if "num_workers" in config.keys():
            num_workers = config["num_workers"]
        else:
            num_workers = 1

        self.ood_eval = ood_eval = config["ood_eval"]
        # data set
        if config["dataset::name"] == "snbs_homo":
            self.train_set, self.valid_set, self.test_set = init_snbs_hom_dataset(
                config
            )
        if config["dataset::name"] == "ws_snbs":
            self.train_set, self.valid_set, self.test_set = init_ws_snbs_dataset(config)
            if ood_eval:
                self.ood_dataset = init_ws_snbs_dataset(config, ood_eval=True)
        if config["dataset::name"] == "TU-collab":
            self.train_set, self.valid_set, self.test_set = init_TU_dataset_collab(
                config
            )
        if config["dataset::name"] == "ogbg-molhiv":
            self.train_set, self.valid_set, self.test_set = init_ogb_dataset(config)
        if config["dataset::name"] == "ogbg-molpcba":
            self.train_set, self.valid_set, self.test_set = init_ogb_dataset(config)
        if config["dataset::name"] == "ogbg-ppa":
            self.train_set, self.valid_set, self.test_set = init_ogb_dataset(config)
        if config["dataset::name"] == "ogbn-proteins":
            self.dataset = init_ogb_dataset(config)
        elif config["dataset::name"] == "ogbn-arxiv":
            self.dataset = init_ogb_dataset(config)
        elif config["dataset::name"][0:4] == "lrgb":
            self.train_set, self.valid_set, self.test_set = init_lrgb_dataset(config)
        if config["dataset::name"] == "surv_nf":
            self.train_set, self.valid_set, self.test_set = init_surv_nf_dataset(config)
            if ood_eval:
                self.ood_dataset = init_surv_nf_dataset(config, ieee_eval=True)
        if config["dataset::name"][0:5] == "linkx":
            self.dataset = init_linkx_dataset(config)

        # if config["ood_eval"]:
        #     self.ieee_set = init_surv_nf_dataset(config, ieee=True)
        #     self.ieee_loader = DataLoader(
        #         self.ieee_set,
        #         batch_size=config["test_set::batchsize"],
        #         shuffle=config["test_set::shuffle"],
        #         num_workers=num_workers,
        #         pin_memory=pin_memory,
        #     )
        if ood_eval:
                self.ood_loader = DataLoader(
                    self.ood_dataset,
                    batch_size=config["test_set::batchsize"],
                    shuffle=config["test_set::shuffle"],
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
        if "dataset::single_grid" not in config:
            config["dataset::single_grid"] = False
        
        if self.dataloader_neighbor_loader == False:
            if config["dataset::single_grid"] == False:
                self.train_loader = DataLoader(
                    self.train_set,
                    batch_size=config["train_set::batchsize"],
                    shuffle=config["train_set::shuffle"],
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                self.valid_loader = DataLoader(
                    self.valid_set,
                    batch_size=config["valid_set::batchsize"],
                    shuffle=config["valid_set::shuffle"],
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                self.test_loader = DataLoader(
                    self.test_set,
                    batch_size=config["test_set::batchsize"],
                    shuffle=config["test_set::shuffle"],
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                cfg_ray = {"len_trainloader": len(self.train_loader)}
        else:
            data = self.dataset[0]
            self.train_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbors] * num_iterations,
                input_nodes=data.train_mask,
                batch_size=config["train_set::batchsize"],
            )
            self.valid_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbors] * num_iterations,
                input_nodes=data.val_mask,
                batch_size=config["valid_set::batchsize"],
            )
            self.test_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbors] * num_iterations,
                input_nodes=data.test_mask,
                batch_size=config["test_set::batchsize"],
            )
            cfg_ray = {"len_trainloader": 1}
    
        if config["dataset::single_grid"] == True:
            cfg_ray = {"len_trainloader": 1}
        else:
            cfg_ray = {"len_trainloader": len(self.train_loader)}

        if "criterion::positive_weight" in config.keys():
            if config["criterion::positive_weight"] == True:
                train_set_positive_weight = (
                    self.train_set.positive_weight.clone().detach()
                )
                self.NN = GNNmodule(config, train_set_positive_weight.numpy(), cfg_ray)
            else:
                self.NN = GNNmodule(config, config_ray=cfg_ray)
        else:
            self.NN = GNNmodule(config, config_ray=cfg_ray)

        if "checkpoint::load_checkpoint" in config.keys():
            if config["checkpoint::load_checkpoint"] == True:
                print("Loading checkpoint from: ", config["checkpoint::experiment_dir"])
                self.load_checkpoint(config["checkpoint::experiment_dir"])

    def step(self):
        # if self.config["dataset::name"] == "surv_nf":
        #     return self.step_surv_nf()
        if self.use_single_grid != True:
            if self.task_type == "regression":
                return self.step_regression()
            elif self.task_type == "classification":
                return self.step_classification()
            elif self.task_type == "regressionThresholding":
                return self.step_regressionThresholding()
        else:
            return self.step_single_grid()

    def step_regression(self):
        # train
        loss_train, R2_train, MAE_train, MSE_train = self.NN.train_epoch_regression(
            self.train_loader
        )
        # valid
        loss_valid, R2_valid, MAE_valid, MSE_valid = self.NN.eval_model_regression(
            self.valid_loader
        )
        # test
        loss_test, R2_test, MAE_test, MSE_test = self.NN.eval_model_regression(self.test_loader)

        result_dict = {
            "train_loss": loss_train,
            "train_R2": R2_train,
            "train_MAE": MAE_train,
            "train_MSE": MSE_train,
            "test_loss": loss_test,
            "test_R2": R2_test,
            "test_MAE": MAE_test,
            "test_MSE": MSE_test,
            "valid_loss": loss_valid,
            "valid_R2": R2_valid,
            "valid_MAE": MAE_valid,
            "valid_MSE": MSE_valid
        }
        if self.ood_eval:
            loss_ood, R2_ood, mae_ood, mse_ood = self.NN.eval_model_regression(self.ood_loader)
            result_dict["ood_loss"] = loss_ood
            result_dict["ood_R2"] = R2_ood
            result_dict["ood_mae"] = mae_ood
            result_dict["ood_mse"] = mse_ood
        return result_dict

    def step_regressionThresholding(self):
        # train
        loss_train, R2_train, fbeta_train, recall_train = (
            self.NN.train_epoch_regressionThresholding(self.train_loader)
        )
        # valid
        loss_valid, R2_valid, fbeta_valid, recall_valid = (
            self.NN.eval_model_regressionThresholding(self.valid_loader)
        )
        # test
        loss_test, R2_test, fbeta_test, recall_test = (
            self.NN.eval_model_regressionThresholding(self.test_loader)
        )

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
        # train
        (
            loss_train,
            acc_train,
            f1_train,
            fbeta_train,
            recall_train,
            precision_train,
            auroc_train,
            average_precision_train,
        ) = self.NN.train_epoch_classification(self.train_loader)
        # valid
        (
            loss_valid,
            acc_valid,
            f1_valid,
            fbeta_valid,
            recall_valid,
            precision_valid,
            auroc_valid,
            average_precision_valid,
        ) = self.NN.eval_model_classification(self.valid_loader)
        # test
        (
            loss_test,
            acc_test,
            f1_test,
            fbeta_test,
            recall_test,
            precision_test,
            auroc_test,
            average_precision_test,
        ) = self.NN.eval_model_classification(self.test_loader)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "train_f1": f1_train,
            "train_fbeta": fbeta_train,
            "train_recall": recall_train,
            "train_precision": precision_train,
            "train_auroc": auroc_train,
            "train_average_precision": average_precision_train,
            "valid_loss": loss_valid,
            "valid_acc": acc_valid,
            "valid_f1": f1_valid,
            "valid_fbeta": fbeta_valid,
            "valid_recall": recall_valid,
            "valid_precision": precision_valid,
            "valid_auroc": auroc_valid,
            "valid_average_precision": average_precision_valid,
            "test_loss": loss_test,
            "test_acc": acc_test,
            "test_f1": f1_test,
            "test_fbeta": fbeta_test,
            "test_recall": recall_test,
            "test_precision": precision_test,
            "test_auroc": auroc_test,
            "test_average_precision": average_precision_test,
        }

        return result_dict

    def step_regression_nf_hetero(self):
        loss_train, R2_train = self.NN.train_epoch_regression_hetero(self.train_loader)
        # valid
        loss_valid, R2_valid = self.NN.eval_model_regression_hetero(self.valid_loader)
        # test
        loss_test, R2_test = self.NN.eval_model_regression_hetero(self.test_loader)

        result_dict = {
            "train_loss": loss_train,
            "train_R2": R2_train,
            "test_loss": loss_test,
            "test_R2": R2_test,
            "valid_loss": loss_valid,
            "valid_R2": R2_valid,
        }

        return result_dict

    # def step_surv_nf(self):
    #     if self.grid_type == "homo":
    #         result_dict = self.step_regression()
    #     else:
    #         result_dict = self.step_regression_nf_hetero()
    #     if self.config["ood_eval"]:
    #         if self.grid_type == "homo":
    #             loss_ood, R2_ood, _, _ = self.NN.eval_model_regression(self.ood_loader)
    #         else:
    #             loss_ood, R2_ood, _ = self.NN.eval_model_regression_hetero(
    #                 self.ood_loader
    #             )
    #         result_dict["ood_loss"] = loss_ood
    #         result_dict["ood_R2"] = R2_ood
    #     return result_dict

    def step_single_grid(self):
        (train_loss,
        train_f1,
        train_fbeta,
        train_accu,
        train_recall,
        train_precision,
        train_auroc,
        train_average_precision,
        valid_loss,
        valid_f1,
        valid_fbeta,
        valid_accu,
        valid_recall,
        valid_precision,
        valid_auroc,
        valid_average_precision,
        test_loss,
        test_f1,
        test_fbeta,
        test_accu,
        test_recall,
        test_precision,
        test_auroc,
        test_average_precision) = self.NN.train_eval_epoch_1grid(self.dataset)


        result_dict = {
                "train_loss": train_loss,
                "train_acc": train_accu,
                "train_f1": train_f1,
                "train_fbeta": train_fbeta,
                "train_recall": train_recall,
                "train_precision": train_precision,
                "train_auroc": train_auroc,
                "train_average_precision": train_average_precision,
                
                "valid_loss": valid_loss,
                "valid_acc": valid_accu,
                "valid_f1": valid_f1,
                "valid_fbeta": valid_fbeta,
                "valid_recall": valid_recall,
                "valid_precision": valid_precision,
                "valid_auroc": valid_auroc,
                "valid_average_precision": valid_average_precision,
                
                "test_loss": test_loss,
                "test_acc": test_accu,
                "test_f1": test_f1,
                "test_fbeta": test_fbeta,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "test_auroc": test_auroc,
                "test_average_precision": test_average_precision,
            }

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


class nan_stopper(tune.Stopper):
    def __init__(self):
        print("nan stopper initialized")

    def __call__(self, trial_id, result):
        if math.isnan(result["valid_loss"]):
            return True
        else:
            False

    def stop_all(self):
        return False
