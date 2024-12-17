using Pkg
Pkg.activate("..")
using GNN_BS_new_dataset
using DataFrames
using CSV

thresholds_targets = all_thresholds(
    -2.0 * 2 * pi, #threshold_ω_surv_low
    2.0 * 2 * pi, # threshold_ω_surv_high,
    .1,
    .1
)

consider_infeasible = true

# Define directories and grid types
ieee_data_dir = "/home/nauck/joined_work/dataset_v2/ieee/data/"
grid_type_ieee = "ieee"

syn_data_dir = "/home/nauck/joined_work/dataset_v2/70-80nodes/data/"
syn_grid_type = "synthetic"

df_all_nodes = DataFrame()
df_all_lines = DataFrame()

# Process IEEE data
id_one = lpad(1, length(digits(10000)), '0')
collect_grid_sim_data!(df_all_nodes, df_all_lines, id_one, ieee_data_dir, grid_type_ieee, thresholds_targets, consider_infeasible)

# Process synthetic ensemble data
for grid_index in 1:1000
    id = lpad(grid_index, length(digits(10000)), '0')
    collect_grid_sim_data!(df_all_nodes, df_all_lines, id, syn_data_dir, syn_grid_type, thresholds_targets, consider_infeasible)
end

# Save the combined DataFrames to CSV files
CSV.write("data_frames/df_all_nodes.csv", df_all_nodes)
CSV.write("data_frames/df_all_lines.csv", df_all_lines)