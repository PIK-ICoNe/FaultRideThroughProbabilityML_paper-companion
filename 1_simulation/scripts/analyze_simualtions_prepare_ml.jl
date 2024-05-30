using Pkg
package_dir = joinpath(@__DIR__, "../")
Pkg.activate(package_dir)

using Revise
using GNN_BS_new_dataset
using HDF5
using DataFrames
using PowerDynamics

struct dataset_properties
    path::Any
    index_start::Int32
    index_end::Int32
end


function get_stats_for_scaling(ds_props)
    data_dir, index_start, index_end = ds_props.path, ds_props.index_start, ds_props.index_end

    grid_data_nodes, grid_data_lines = read_grids_data(data_dir, index_start, index_end)
    # get node features
    df_all_grids_node_features = combine_data_one_df(grid_data_nodes)
    df_all_grids_line_properties = combine_data_one_df(grid_data_lines)


    # get node types and their indices
    all_node_types001 = df_all_grids_node_features.node_type_int
    # get idx of nodes
    idx_loads, idx_normalform1, idx_normalform2, idx_normalform3, idx_slack = get_idx_of_nodes(all_node_types001)

    # get min_max of node features
    idx_all_nodes = 1:size(all_node_types001, 1)
    idx_normalform_all = vcat(idx_normalform1, idx_normalform2, idx_normalform3)
    indices_scaling = Dict()
    indices_scaling["P"] = idx_all_nodes
    indices_scaling["Q"] = idx_all_nodes
    indices_scaling["sum_y_real"] = idx_all_nodes
    indices_scaling["sum_y_imag"] = idx_all_nodes
    indices_scaling["sum_y_shunt_mk_imag"] = idx_all_nodes
    indices_scaling["Bᵤ_imag"] = idx_normalform_all
    indices_scaling["Cᵤ_real"] = idx_normalform_all
    indices_scaling["Hᵤ_real"] = idx_normalform_all
    indices_scaling["Bₓ_real"] = idx_normalform_all
    indices_scaling["Gₓ_real"] = idx_normalform_all

    scaling_statistics_nodes = get_statistics_features_df(df_all_grids_node_features, ["P", "Q", "Bᵤ_imag", "Cᵤ_real", "Hᵤ_real", "Bₓ_real", "Gₓ_real", "sum_y_real", "sum_y_imag", "sum_y_shunt_mk_imag"], indices_scaling)
    scaling_statistics_lines = get_statistics_features_df(df_all_grids_line_properties, ["y_real", "y_imag", "y_shunt_mk_imag", "power_flow_P", "power_flow_Q"])

    return scaling_statistics_nodes, scaling_statistics_lines
end

function save_data(ds_props, thresholds, scaling_variables_nodes, scaling_variables_lines, desired_variables_per_node_type, desired_variables_per_line,add_node_type_load, add_node_type_slack)
    ## analysis of dynamical results
    data_dir, start_index, end_index = ds_props.path, ds_props.index_start, ds_props.index_end

    ## read in all data
    grid_data_nodes, grid_data_lines = read_grids_data(data_dir, start_index, end_index)

    imputation_norm = [Dict("node_type_int" => 0, "variable" => "Bₓ_real","value" => normalize_array(scaling_variables_nodes["Bₓ_real"].mean,scaling_variables_nodes["Bₓ_real"])),
    Dict("node_type_int" => 4, "variable" => "Bₓ_real","value" => normalize_array(scaling_variables_nodes["Bₓ_real"].mean,scaling_variables_nodes["Bₓ_real"]))
    ]

    imputation_std = [Dict("node_type_int" => 0, "variable" => "Bₓ_real","value" => 0),
    Dict("node_type_int" => 4, "variable" => "Bₓ_real","value" => 0)
    ]

    imputation_noscaling = [Dict("node_type_int" => 0, "variable" => "Bₓ_real","value" => scaling_variables_nodes["Bₓ_real"].mean),
    Dict("node_type_int" => 4, "variable" => "Bₓ_real","value" => scaling_variables_nodes["Bₓ_real"].mean)
    ]

    scaling_dict_norm = Dict("nodes" => Dict("mode" => "normalize", "variables" => scaling_variables_nodes, "imputation"=> imputation_norm), "lines" => Dict("mode" => "normalize", "variables" => scaling_variables_lines))
    scaling_dict_std = Dict("nodes" => Dict("mode" => "standardize", "variables" => scaling_variables_nodes, "imputation" => imputation_std), "lines" => Dict("mode" => "standardize", "variables" => scaling_variables_lines))
    
    scaling_dict_noscaling = Dict("nodes" => Dict("mode" => false, "variables" => [], "imputation"=> imputation_noscaling), "lines" => Dict("mode" => false, "variables" => []))
    
    
    # imputation = [Dict("node_type_int" => "0", "variable" => "Bₓ_real","value" => scaling_variables_nodes["Bₓ_real"].mean),
    # Dict("node_type_int" => "4", "variable" => "Bₓ_real","value" => scaling_variables_nodes["Bₓ_real"].mean)
    # ]

    # prepare data for ml
    filename_ml_homo = joinpath(data_dir, "ml_input_grid_data_homo.h5")
    filename_ml_hetero = joinpath(data_dir, "ml_input_grid_data_hetero.h5")
    prepare_grid_data_homo(data_dir, filename_ml_homo, grid_data_lines, grid_data_nodes, desired_variables_per_node_type, desired_variables_per_line, add_node_type_load,add_node_type_slack, thresholds, scaling_dict_noscaling, true)
    prepare_grid_data_hetero(data_dir, filename_ml_hetero, grid_data_lines, grid_data_nodes, desired_variables_per_node_type, desired_variables_per_line, thresholds, scaling_dict_noscaling, true)

    filename_ml_homo = joinpath(data_dir, "ml_input_grid_data_homo_norm.h5")
    filename_ml_hetero = joinpath(data_dir, "ml_input_grid_data_hetero_norm.h5")
    prepare_grid_data_homo(data_dir, filename_ml_homo, grid_data_lines, grid_data_nodes, desired_variables_per_node_type, desired_variables_per_line, add_node_type_load, add_node_type_slack, thresholds, scaling_dict_norm, true)
    prepare_grid_data_hetero(data_dir, filename_ml_hetero, grid_data_lines, grid_data_nodes, desired_variables_per_node_type, desired_variables_per_line, thresholds, scaling_dict_norm, true)

    filename_ml_homo = joinpath(data_dir, "ml_input_grid_data_homo_std.h5")
    filename_ml_hetero = joinpath(data_dir, "ml_input_grid_data_hetero_std.h5")
    prepare_grid_data_homo(data_dir, filename_ml_homo, grid_data_lines, grid_data_nodes, desired_variables_per_node_type, desired_variables_per_line, add_node_type_load, add_node_type_slack, thresholds, scaling_dict_std, true)
    prepare_grid_data_hetero(data_dir, filename_ml_hetero, grid_data_lines, grid_data_nodes, desired_variables_per_node_type, desired_variables_per_line, thresholds, scaling_dict_std, true)
end

# desired variables for hetero grids
desired_variables_per_node_type = Dict()
desired_variables_per_node_type["load"] = ["P", "Q", "sum_y_real", "sum_y_imag", "sum_y_shunt_mk_imag"]
desired_variables_per_node_type["normalForm"] = ["P", "Q", "Bₓ_real", "sum_y_real", "sum_y_imag", "sum_y_shunt_mk_imag"]
desired_variables_per_node_type["SlackAlgebraic"] = ["P", "Q", "Bₓ_real", "sum_y_real", "sum_y_imag", "sum_y_shunt_mk_imag"]
desired_variables_per_line = ["y_real", "y_imag", "y_shunt_mk_imag", "power_flow_P", "power_flow_Q"]

thresholds_1 = all_thresholds(
    -2.0 * 2 * pi, #threshold_ω_surv_low
    2.0 * 2 * pi, # threshold_ω_surv_high,
    .1, #threshold_ω_snbs
    .1 #threshold_u_snbs
)
add_node_type_load = true
add_node_type_slack = true

ieee_dataset = dataset_properties("/Users/christiannauck/work/pik/data/dataset_v2/ieee", 1, 1)
regular_dataset = dataset_properties("/Users/christiannauck/work/pik/data/dataset_v2/70-80nodes", 1, 1000)
# training_dataset = dataset_properties("/Users/christiannauck/work/pik/data/dataset_v2/70-80nodes", 1, 600)


#get statistics for scaling
# scaling_variables_nodes, scaling_variables_lines = get_stats_for_scaling(training_dataset)
scaling_variables_nodes, scaling_variables_lines = get_stats_for_scaling(regular_dataset)

# save_data(ieee_dataset, thresholds_1, scaling_variables_nodes, scaling_variables_lines, desired_variables_per_node_type, desired_variables_per_line)
save_data(regular_dataset, thresholds_1, scaling_variables_nodes, scaling_variables_lines, desired_variables_per_node_type, desired_variables_per_line,add_node_type_load,add_node_type_slack)


## scale ieee differently
ieee_scaling_variables_nodes, ieee_scaling_variables_lines = get_stats_for_scaling(ieee_dataset)
save_data(ieee_dataset, thresholds_1, ieee_scaling_variables_nodes, ieee_scaling_variables_lines, desired_variables_per_node_type, desired_variables_per_line,add_node_type_load,add_node_type_slack)

