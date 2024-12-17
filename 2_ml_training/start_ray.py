from pathlib import Path
from sys import argv
import math

import ray as ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper
from ray import tune, train
from ray.air import CheckpointConfig
from ray_trainable import NN_tune_trainable, nan_stopper

regular_dataset_path = Path("/home/nauck/joined_work/dataset_v2/ml/datasets/2024_12_02/synthetic")
dataset_ieee_path = Path("/home/nauck/joined_work/dataset_v2/ml/datasets/2024_12_02/ieee")
result_path = Path("/home/nauck/joined_work/dataset_v2/ml/run/DBGNN/bo_op05_new_code").as_posix()
tmp_dir = "/p/tmp/nauck/ray_tmp/"
N_cpus = int(argv[1])
port_dashboard = int(argv[2])
tmp_dir = tmp_dir + argv[3]
ray.init(_temp_dir=tmp_dir,num_cpus=N_cpus, num_gpus = 2, include_dashboard=True,dashboard_port=port_dashboard)

cfg = {}

# dataset surv_nf 
cfg["dataset::path"] = regular_dataset_path
cfg["dataset::ood_path"] = dataset_ieee_path
cfg["result::path"] = result_path
cfg["dataset::name"] = "surv_nf" 
cfg["input_features_node_dim"] = 9
cfg["input_features_edge_dim"] = 5
cfg["output_features_node_dim"] = 1
cfg["num_classes"] = 1
cfg["pool"] = False
cfg["task"] = "surv"
cfg["grid_type"] = "homo"
cfg["task_type"] = "regression"
cfg["criterion"] = "MSELoss"
cfg["list_metrics"] = ["r2"]
metric = "valid_loss"
metric_mode = "min"
metric_checkpoint = metric

## dataset nf properties
cfg["scaling"] = False#"normalize"
cfg["ieee_scaling"] = False#"normalize"
cfg["dtype"] = "float32"
cfg["grid_type"] = "homo"#"hetero"
cfg["ood_eval"] = True 
cfg["train_slice"] = slice(1, 700)
cfg["valid_slice"] = slice(701, 850)
cfg["test_slice"] = slice(851, 1000)

# dataset batch sizes
cfg["train_set::batchsize"] = 50 #800
cfg["test_set::batchsize"] = 500
cfg["valid_set::batchsize"] = 500
cfg["train_set::shuffle"] = True
cfg["test_set::shuffle"] = False
cfg["valid_set::shuffle"] = False




# model settings
cfg["model_name"] = "DBGNN"# "GAT"#"TAG"#"DBGNN"
cfg["num_layers"] = 2
cfg["final_linear_layer"] = False#True
cfg["final_sigmoid_layer"] = False
cfg["bias_zero"] = True
cfg["activation_name_n"] = "LeakyReLU"#"ReLU"
cfg["activation_name_e"] = "LeakyReLU"#"ReLU"
cfg["dropout_n"] = 0.014877#tune.loguniform(.014, .4)#0.014150743970295889#tune.loguniform(1E-4,2E-1)#0
cfg["dropout_e"] = 0.002454#tune.loguniform(.0019, .4)#0.0018562467513318446#tune.loguniform(4E-4,2E-1)#0

# ray settings
cfg["save_after_epochs"] = 10000
cfg["checkpoint_freq"] = 10000
cfg["num_samples"] = 5 
cfg["ray_name"] = "DBGNN"

cfg["DBGNN::dense_after_linDB"] = False
cfg["DBGNN::in_channels_n"] = cfg["input_features_node_dim"] 
cfg["DBGNN::out_channels_n"] = cfg["num_classes"]
cfg["DBGNN::hidden_channels_n"] = 120
cfg["DBGNN::in_channels_e"] = cfg["input_features_edge_dim"]
cfg["DBGNN::hidden_channels_e"] = 120
cfg["DBGNN::num_steps"] = 30 
cfg["DBGNN::Δ"] = 1e-4#tune.loguniform(1E-5, 1E-1)
cfg["DBGNN::scale_features"] = False

cfg["skip_connection_n"] = True#False
cfg["skip_connection_e"] = True#False

# training settings
cfg["cuda"] = True
#cfg["num_workers"] = 1
#cfg["num_threads"] = 2
# cfg["manual_seed"] = 1
cfg["manual_seed"] = tune.choice([1,2,3,4,5])
cfg["epochs"] = 5000
cfg["optim::optimizer"] = "adamW"#"SGD"
# cfg["optim::LR"] = 0.193048 # 1.1
# cfg["optim::LR"] = 0.3
cfg["optim::LR"] = 4.350367242636451e-05#tune.loguniform(1e-5, 1e-4)#6.461370779377669e-05
# cfg["optim::LR"] = tune.choice([1.1])
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 1E-7#tune.loguniform(1E-9, 10) 
cfg["optim::scheduler"] = "OneCycleLR" 
cfg["optim::anneal_strategy"] = "cos"
cfg["optim::div_factor"] = 413759#15290378#tune.randint(1000,20E6)
cfg["optim::final_div_factor"] = 1241798872312.98#4850042250280.317383#tune.loguniform(1E9,1E13) 
cfg["optim::max_LR"] = 863.4449718452441#2.056397#tune.loguniform(1E-1,1E3)
cfg["optim::ReducePlat_patience"] = 20
cfg["optim::LR_reduce_factor"] = .7
cfg["optim::stepLR_step_size"] = 30
# cfg["optim::scheduler"] = "stepLR"
cfg["search_alg"] = "Optuna"


## gradient clipping
cfg["gradient_clipping::grad_norm"] = True
cfg["gradient_clipping::grad_norm::max_norm"] = 100
cfg["gradient_clipping::grad_value"] = True
cfg["gradient_clipping::grad_value::clip_value"] = 100


# evaluation
cfg["eval::threshold"] = .1

asha_scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric=metric,
    mode=metric_mode,
    max_t=cfg["epochs"],
    grace_period=250,
    #reduction_factor=3,
    #brackets=5,
)

optuna_search = OptunaSearch(
    metric=metric,
    mode=metric_mode,
    points_to_evaluate=[{"manual_seed": 1}, {"manual_seed": 2}, {"manual_seed": 3}, {"manual_seed": 4}, {"manual_seed": 5}],
    # points_to_evaluate=[{"optim::LR": 6.461370779377669e-05}]
    # points_to_evaluate = [{"optim::div_factor": 222315, "optim::max_LR": 3.689683, "optim::final_div_factor": 752185674861.16394}]
    # points_to_evaluate = [{"optim::div_factor": 32, "optim::max_LR": 6.1E-4, "optim::final_div_factor": 5.8E5}]
    #points_to_evaluate = [{'dropout_n': 1E-4, 'dropout_e': 1E-4}]
)
tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]), TrialPlateauStopper(metric=metric, num_results=500, std=0.001, grace_period=1000), nan_stopper())
#tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]))

#checkpoint_freq = cfg["checkpoint_freq"]
num_samples = cfg["num_samples"]
name = cfg["ray_name"]

checkpoint_config = CheckpointConfig(
   num_to_keep=1, checkpoint_frequency = 1, checkpoint_score_attribute=metric, checkpoint_score_order=metric_mode, checkpoint_at_end= True,
)

analysis = tune.run(
    NN_tune_trainable,
    name=name,
    stop=tune_stop,
    config=cfg,
    num_samples=num_samples,
    storage_path=result_path,
    search_alg=optuna_search,
    # scheduler=asha_scheduler,
    checkpoint_config=checkpoint_config,
    resources_per_trial={'cpu': 1. ,'gpu': .3},
    max_failures=1,
    resume=False,
)



print('best config: ', analysis.get_best_config(metric=metric, mode=metric_mode))


ray.shutdown()
print("finished")
