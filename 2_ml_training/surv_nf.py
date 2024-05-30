import os
import numpy as np
import h5py

import torch as torch
from torch.utils.data import Dataset

from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset as gDataset
from torch_geometric.data import Data as gData
from torch_geometric.nn import to_hetero


class dataset_v2(gDataset):
    def __init__(self, dataset_path, target_name, grid_type, slice_index, scaling=False, dtype="float32"):
        super(dataset_v2, self).__init__()
        self.dataset_path = Path(PATHMISSING)
        self.target_name = target_name
        self.grid_type = grid_type
        self.dtype = dtype
        self.slice_index = slice_index
        self.scaling = scaling
        self.data = {}
        self.mask = {}
        if grid_type == "homo":
            self.read_grid_targets_homo()
        elif grid_type == "hetero":
            self.read_grid_targets_hetero()

    def read_grid_targets_homo(self):
        self.use_homo_grid_data = True
        if self.scaling == "normalize":
            file_to_read = '/ml_input_grid_data_homo_norm.h5'
        elif self.scaling == "standardize":
            file_to_read = '/ml_input_grid_data_homo_std.h5'
        elif self.scaling == False:
            file_to_read = '/ml_input_grid_data_homo.h5'
        else:
            print("Error: Invalid scaling chosen")

        file_to_read = str(self.dataset_path) + file_to_read
        f = h5py.File(file_to_read, 'r')
        dset_grids = f['grids']
        index_start = self.slice_index.start
        index_stop = self.slice_index.stop
        for index_grid in range(index_start, index_stop+1):
            edge_index = (np.array(dset_grids[str(index_grid)].get(
                "edge_index"), dtype='int64') - 1).transpose()
            node_features = np.array(dset_grids[str(index_grid)].get(
                "node_features"), dtype=self.dtype).transpose()
            line_properties = np.array(
                dset_grids[str(index_grid)].get("edge_attr"), dtype=self.dtype)
            targets = np.array(dset_grids[str(index_grid)]["targets"].get(
                self.target_name), dtype=self.dtype)
            mask = np.array(dset_grids[str(index_grid)].get("mask"), dtype='int32')
            self.data[index_grid - index_start] = gData(x=(torch.tensor(node_features)), edge_index=torch.tensor(
                edge_index), edge_attr=torch.tensor(line_properties), y=torch.tensor(targets.transpose()), mask = torch.tensor(mask > 0.1))
            # self.mask[index_grid - index_start] = torch.tensor(mask > 0.1)

    def read_grid_targets_hetero(self):
        if self.scaling == "normalize":
            file_to_read = '/ml_input_grid_data_hetero_norm.h5'
        elif self.scaling == "standardize":
            file_to_read = '/ml_input_grid_data_hetero_std.h5'
        elif self.scaling == False:
            file_to_read = '/ml_input_grid_data_hetero.h5'
        else:
            print("Error: Invalid scaling chosen")
        file_to_read = str(self.dataset_path) + file_to_read
        f = h5py.File(file_to_read, 'r')
        dset_grids = f['grids']
        index_start = self.slice_index.start
        index_stop = self.slice_index.stop
        for index_grid in range(index_start, index_stop+1):
            data = HeteroData()
            one_grid_dset = dset_grids[str(index_grid)]
            node_features_group = one_grid_dset["node_features_group"]
            name_node_types = list(node_features_group)
            num_node_types = len(name_node_types)
            for i in range(num_node_types):
                node_type = name_node_types[i]
                node_features = np.array(
                    node_features_group[node_type].get("features"), dtype=self.dtype)
                if node_type != "SlackAlgebraic":
                    targets = np.array(node_features_group[node_type]["targets"].get(
                        self.target_name), dtype=self.dtype)
                    data[node_type].y = torch.tensor(targets.transpose())
                    data[node_type].x = torch.tensor(node_features.transpose())
                else:
                    data[node_type].x = torch.tensor(
                        node_features.transpose()).unsqueeze(0).unsqueeze(0)
            line_features_group = one_grid_dset["line_features_group"]
            num_line_types = len(list(line_features_group))
            for i in range(num_line_types):
                line_properties = line_features_group[str(i+1)]
                first_node_type = line_properties.get(
                    'first_node_type')[()].decode()
                second_node_type = line_properties.get(
                    'second_node_type')[()].decode()
                line_type = line_properties.get('line_type')[()].decode()
                edge_index = np.array(line_properties.get(
                    'edge_index'), dtype='int64') - 1
                edge_attr = np.array(line_properties.get(
                    'edge_attr'), dtype=self.dtype)
                data[first_node_type, line_type, second_node_type].edge_index = torch.tensor(
                    edge_index.transpose())
                data[first_node_type, line_type, second_node_type].edge_attr = torch.tensor(
                    edge_attr.transpose())
            self.data[index_grid-index_start] = data
        print("setting up hetero dataset finished")

    def len(self):
        return len(self.data)

    def get(self, index):
        return self.data[index]



class dataset_v2_mlp(Dataset):
    def __init__(self, dataset_path, target_name, slice_index, scaling=False, dtype="float32"):
        super(dataset_v2_mlp, self).__init__()
        self.scaling = scaling
        self.dataset_path = Path(PATHMISSING)
        self.target_name = target_name
        self.dtype = dtype
        self.slice_index = slice_index
        self.scaling = scaling
        self.input_data = {}
        self.targets = {}
        self.mask = {}
        self.read_features_targets()

    def read_features_targets(self):
        if self.scaling == "normalize":
            file_to_read =     '/ml_input_grid_data_homo_norm.h5'
        elif self.scaling == "standardize":
            file_to_read =     '/ml_input_grid_data_homo_std.h5'
        elif self.scaling == False:
            file_to_read =     '/ml_input_grid_data_homo.h5'
        else :
            print("Error: Invalid scaling chosen")


        file_to_read = str(self.dataset_path) + file_to_read
        f = h5py.File(file_to_read, 'r')
        dset_grids = f['grids']
        index_start = self.slice_index.start
        index_stop = self.slice_index.stop
        for index_grid in range(index_start, index_stop+1):
            node_features = np.array(dset_grids[str(index_grid)].get(
                "node_features"), dtype=self.dtype).transpose()
            targets = np.array(dset_grids[str(index_grid)]["targets"].get(
                self.target_name), dtype=self.dtype)
            self.input_data[index_grid - index_start] = torch.tensor(node_features)
            self.targets[index_grid - index_start] = torch.tensor(targets.transpose())
            mask = np.array(dset_grids[str(index_grid)].get("mask"), dtype='int32')
            self.mask[index_grid - index_start] = torch.tensor(mask > 0.1)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.targets[idx], self.mask[idx]


def init_surv_nf_dataset(config, ieee=False):

    if "MLP::use_MLP" in config.keys():
        use_MLP = config["MLP::use_MLP"]
    else:
        use_MLP = False

    ieee_eval = config["ieee_eval"]
    scaling = config["scaling"]
    train_slice = config["train_slice"]
    valid_slice = config["valid_slice"]
    test_slice = config["test_slice"]
    task = config["task"]
    
    if use_MLP:
        if ieeee == False:
            train_set = dataset_v2_mlp(config[dataset_path = Path(PATHMISSING)
            valid_set = dataset_v2_mlp(config[dataset_path = Path(PATHMISSING)
            test_set = dataset_v2_mlp(config[dataset_path = Path(PATHMISSING)
            return train_set, valid_set, test_set
        else:
            ieee_set = dataset_v2_mlp(config["dataset::ieee_path"], task, slice_index=test_slice, scaling=scaling, dtype=config["dtype"])
            return ieee_set

    else:
        if ieee == False:
            train_set = dataset_v2(config[dataset_path = Path(PATHMISSING)
                                        slice_index=train_slice, scaling=config["scaling"],  dtype=config["dtype"])
            valid_set = dataset_v2(config[dataset_path = Path(PATHMISSING)
                                slice_index=valid_slice, scaling=config["scaling"],  dtype=config["dtype"])
            test_set = dataset_v2(config[dataset_path = Path(PATHMISSING)
                                        slice_index=test_slice, scaling=config["scaling"],  dtype=config["dtype"])             
            if config["grid_type"] == "hetero":
                config["hetero::datasample"] = train_set[0]
            return train_set, valid_set, test_set
        else:
            ieee_set = dataset_v2(config["dataset::ieee_path"], task, config["grid_type"],
                                       slice_index=slice(1,1), scaling=config["ieee_scaling"],  dtype=config["dtype"])
            return ieee_set

