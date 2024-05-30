using Revise

using Pkg
package_dir = joinpath(@__DIR__, "..")
Pkg.activate(package_dir)


using HDF5
using Plots
using PowerDynamics
using DataFrames
using GNN_BS_new_dataset

using PyPlot
using Statistics


mutable struct results_one_setup
    indices_node_type
    results
    statistics
end

struct node_type_indices
    load
    normalform1
    normalform2
    normalform3
    slack
end


function get_statistical_results(data_dir, grid_index_start, grid_index_end, thresholds)
    grid_data_nodes, _ = read_grids_data(data_dir, grid_index_start, grid_index_end)
    df_all_grids_node_features = combine_data_one_df(grid_data_nodes)
    all_node_types_int = df_all_grids_node_features.node_type_int
    node_types = node_type_indices(get_idx_of_nodes(all_node_types_int)...)
    results = obtain_results(data_dir,grid_index_start,grid_index_end)
    statistics = compute_statistics(results,thresholds,false);
    return results_one_setup(node_types,results,statistics)
end

thresholds_1 = all_thresholds(
    -2. * 2*pi, #threshold_ω_surv_low
     2. * 2*pi, # threshold_ω_surv_high
     .1,
     .1
);


grid_ieee = get_statistical_results("./data/",1,1, thresholds_1);