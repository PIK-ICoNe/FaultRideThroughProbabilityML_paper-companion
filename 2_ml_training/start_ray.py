from pathlib import Path
import ray as ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper
from ray import tune
import ray as ray
from ray_trainable import NN_tune_trainable
from sys import argv
from optuna.distributions import FloatDistribution

dataset_regular_path = Path(PATHMISSING)
dataset_ieee_path = Path(PATHMISSING)
dataset_path = Path(PATHMISSING)
tmp_dir = Path(PATHMISSING)
N_cpus = int(argv[1])
port_dashboard = int(argv[2])
tmp_dir = Path(PATHMISSING)
ray.init(_temp_dir=tmp_dir,num_cpus=N_cpus, num_gpus = 1, include_dashboard=True,dashboard_port=port_dashboard)

cfg = {}
cfg[dataset_path = Path(PATHMISSING)
cfg["dataset::ieee_path"] = dataset_ieee_path
cfg["dataset_name"] = "surv_nf"
cfg["train_slice"] = slice(1, 700)
cfg["valid_slice"] = slice(701, 850)
cfg["test_slice"] = slice(851, 1000)


cfg["train_set::batchsize"] = 700#tune.randint(1,600)
cfg["train_set::shuffle"] = False
cfg["valid_set::batchsize"] = 10000
cfg["valid_set::shuffle"] = False
cfg["test_set::batchsize"] = 10000
cfg["test_set::shuffle"] = False


cfg["task"] = "surv"
cfg["task_type"] = "regression"
cfg["criterion"] = "MSELoss"
cfg["num_classes"] = 1
cfg["pool"] = False


## dataset nf properties
cfg["scaling"] = "standardize"#tune.choice(["normalize", False, "standardize"]) #"standardize"
cfg["ieee_scaling"] = "standardize"#tune.choice(["normalize", False, "standardize"])# "standardize"
cfg["dtype"] = "float32"
cfg["grid_type"] = "homo" #"hetero"
cfg["ieee_eval"] = True
# model settings
cfg["model_name"] = "TAG"#"Transformer_simple"  # "GAT_simple"
cfg["num_layers"] =  3#tune.randint(2,6)
cfg["hidden_channels"] = 304#tune.randint(200,500)#400 # 16
cfg["edge_dim"] = 5

cfg["dropout_n"] = 0.34508926764854975#tune.uniform(.1,.9)
cfg["dropout_e"] = 0
cfg["linear_layer_after_conv"] = True# False #True

### other models
cfg["Transformer::beta"] = True
cfg["TAG::K"] = 3#tune.randint(3,6)
cfg["activation"] = None#"ReLU"
cfg["ll_after_conv_dim"] = 500
cfg["final_linear_layer"] = True
cfg["final_sigmoid_layer"] = False# True#False #True

### GAT
cfg["heads"] = 2
cfg["add_self_loops"] = False
cfg["GAT::v2"] = True


## MLP
# cfg["MLP::use_MLP"] = False
# cfg["MLP::num_target_classes"] = 1
# cfg["MLP::num_input_features"] = 5
# cfg["MLP::num_hidden_layers"] = 1
# cfg["MLP::num_hidden_unit_per_layer"] = 30

# training settings
cfg["cuda"] = True


cfg["manual_seed"] = tune.choice([1, 2, 3, 4, 5])
cfg["optim::LR"] = 0.034851515880887085 #tune.loguniform(1e-2, 5e-1) #.01
cfg["optim::optimizer"] = "SGD"
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 1e-9
cfg["optim::scheduler"] = "None"
cfg["eval::threshold"] = .1
cfg["epochs"] = 200000

## gradient clipping
cfg["gradient_clipping"] = True
cfg["gradient_clipping_::max_norm"] = 10

# ray settings
cfg["save_after_epochs"] = 1000
cfg["checkpoint_freq"] = 1000
cfg["num_samples"] = 5 
cfg["ray_name"] = "TAG"



metric = "valid_loss"
metric_checkpoint = "valid_R2"
asha_scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric=metric,
    mode="min",
    max_t=cfg["epochs"],
    # grace_period=150,
    reduction_factor=3,
    brackets=5,
)

optuna_search = OptunaSearch(
    metric="valid_loss",
    mode="min",
    points_to_evaluate=[{"manual_seed": 1}, {"manual_seed": 2}, {"manual_seed": 3}, {"manual_seed": 4}, {"manual_seed": 5}]
    #points_to_evaluate=[{"ieee_scaling": False, "scaling": False}, {"ieee_scaling": "standardize", "scaling": "standardize"},{"ieee_scaling": "normalize", "scaling": "normalize"}]
)

tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]))
# tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]), TrialPlateauStopper(
#    metric=metric, num_results=100, std=0.005, grace_period=150))


checkpoint_freq = cfg["checkpoint_freq"]
name = cfg["ray_name"]


analysis = tune.run(
    NN_tune_trainable,
    name=name,
    stop=tune_stop,
    config=cfg,
    num_samples=cfg["num_samples"],
    local_dir=result_path,
    search_alg=optuna_search,
    # checkpoint_freq=checkpoint_freq,
    keep_checkpoints_num=1,
    checkpoint_score_attr=metric_checkpoint,
    checkpoint_freq=1,
    checkpoint_at_end=True,
    resources_per_trial={'cpu': 1. ,'gpu': .2},
    # resources_per_trial={'cpu': 1., 'gpu': 0},
    max_failures=1,
    # scheduler=asha_scheduler,
    resume=True,
    reuse_actors=False,
)

print('best config: ', analysis.get_best_config(metric=metric, mode="min"))
# ray.shutdown()
print("finished")
