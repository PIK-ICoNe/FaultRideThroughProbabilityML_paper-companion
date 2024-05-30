# Paper companion for paper: Predicting fault-ride-through probability of inverter-dominated power grids using machine learning


This GitHub repository includes code and data that allow you to explore the codebase and potentially reuse parts of it for future projects. You can find the complete code and data for reproducing the results at: https://zenodo.org/record/11193718.

```
├── 1_simulation 
    ├── Manifest.toml
    ├── Project.toml
    ├── scripts
    ├── src
    └── test
├── 2_machine_learning
└── Readme.md
```

The directory ```1_simulation``` contains the code to generate the datasets and the following scripts can be used to generate synthetic grids, conduct dynamical simulations and prepare the data for ML.
The paths need to be set, to execute the scripts properly.

1. ```generate_grids.jl``` or ```generate_ieee_grid.jl```
2. ```run_dynamic_simulation.jl``` or ```run_ieee96_sim.jl```
3. ```analyze_simualtions_prepare_ml.jl```

To code for training the TAG model can be found in ```2_machine_learning```. The training can be started by running ```python start_ray.py```, after manipulating the file paths and installing the necessary packages (see conda yml file in ```2_ml_training/env```). 
