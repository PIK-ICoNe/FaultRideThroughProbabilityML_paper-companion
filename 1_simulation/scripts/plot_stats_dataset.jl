using Revise

using Pkg
# package_dir = joinpath(@__DIR__, "../gnn_bs_new_dataset")
package_dir = joinpath(@__DIR__, "../")
Pkg.activate(package_dir)


using HDF5
using Plots
using Plots.PlotMeasures
using PowerDynamics
using DataFrames
using GNN_BS_new_dataset
using LaTeXStrings
using CSV

# using PyPlot
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


thresholds_targets = all_thresholds(
    -2.0 * 2 * pi, #threshold_ω_surv_low
    2.0 * 2 * pi, # threshold_ω_surv_high,
    .1,
    .1
)





# Define directories and grid types
ieee_data_dir = "../ieee/data/"
grid_type_ieee = "ieee"

syn_data_dir = "../70-80nodes/data/"
syn_grid_type = "synthetic"



path_csv_dir = "PATH_TO_CSV_DIR"

df_all_nodes = CSV.read(string(path_csv_dir,"df_all_nodes.csv"), DataFrame)
df_all_lines = CSV.read(string(path_csv_dir,"df_all_lines.csv"), DataFrame)

df_ieee_nodes = filter(row -> row.grid_type == "ieee", df_all_nodes)
df_syn_nodes = filter(row -> row.grid_type == "synthetic", df_all_nodes)

plot_properties = Dict()
plot_properties["nBins"] = range(0, 1, length=30)#30
plot_properties["xlims"] = [.0, 1.06]
plot_properties["ylims_nf"] = (0, .2) #[.0, 40]
plot_properties["ylims_load"] = (0, .2) #[.0, 30]
plot_properties["xlabel"] = "SURV"
plot_properties["ylabel"] = "density"
plot_properties["fontSizeGuide"] = 8
plot_properties["fontSizeTick"] = 6
plot_properties["fontSizeTitle"] = 10
right_margin = -0mm
left_margin = 2mm
color_pal = palette(:tab20b)

function make_subplot(plot_data, title, legend_name, plot_properties, color, y_lims, left_margin, right_margin = 0mm, xaxis = false, yaxis = false, xlabel = false, ylabel = false)
    nBins = plot_properties["nBins"]
    x_label = plot_properties["xlabel"]
    y_label = plot_properties["ylabel"]
    fsGuide = plot_properties["fontSizeGuide"]
    fsTitle = plot_properties["fontSizeTitle"]
    fsTick = plot_properties["fontSizeTick"]
    x_lims = plot_properties["xlims"]
    bottom_margin=0mm
    fig = histogram(plot_data, normalize=:probability, color=color, xlims = x_lims, y_lims = y_lims, bins=nBins, xaxis = xaxis, yaxis=yaxis, grid =false, bottom_margin = bottom_margin, right_margin = right_margin, top_margin = -.2mm, left_margin = left_margin, fg_legend = :false, label=legend_name)    
    plot!(xtickfontsize=fsTick,ytickfontsize=fsTick, xlabelfontsize=fsGuide, ylabelfontsize=fsGuide)
    if xlabel == true
#         xlabel!("survivability")
        # xlabel!(L"p_{frt}")
        xlabel!(L"p_{\mathrm{frt}}")
    end
    if ylabel == true
        ylabel!("normalized density")
    end
    if title !=false
        title!(title, titlefontsize=fsTitle)
    end
    return fig
end

println("grid_ieee_surv_mean: ", mean(df_ieee_nodes.surv))


# Filter the DataFrame to get indices for each node type
idx_ieee_normalform1 = findall(row -> row[:node_type] == "PowerDynamics.NormalForm{1}" && row[:B_x_real] == -2.0, eachrow(df_ieee_nodes))
idx_ieee_normalform2 = findall(row -> row[:node_type] == "PowerDynamics.NormalForm{1}" && row[:B_x_real] == -1.0, eachrow(df_ieee_nodes))
idx_ieee_normalform3 = findall(row -> row[:node_type] == "PowerDynamics.NormalForm{1}" && row[:B_x_real] == -0.2, eachrow(df_ieee_nodes))
idx_ieee_load = findall(row -> row[:node_type] == "PowerDynamics.PQAlgebraic", eachrow(df_ieee_nodes))

# Filter the DataFrame to get indices for each node type
idx_syn_normalform1 = findall(row -> row[:node_type] == "PowerDynamics.NormalForm{1}" && row[:B_x_real] == -2.0, eachrow(df_syn_nodes))
idx_syn_normalform2 = findall(row -> row[:node_type] == "PowerDynamics.NormalForm{1}" && row[:B_x_real] == -1.0, eachrow(df_syn_nodes))
idx_syn_normalform3 = findall(row -> row[:node_type] == "PowerDynamics.NormalForm{1}" && row[:B_x_real] == -0.2, eachrow(df_syn_nodes))
idx_syn_load = findall(row -> row[:node_type] == "PowerDynamics.PQAlgebraic", eachrow(df_syn_nodes))


# Create a struct to store the indices
struct NodeTypeIndices
    normalform1::Vector{Int}
    normalform2::Vector{Int}
    normalform3::Vector{Int}
    load::Vector{Int}
end

# Instantiate the struct with the indices
idx_ieee = NodeTypeIndices(idx_ieee_normalform1, idx_ieee_normalform2, idx_ieee_normalform3, idx_ieee_load)
idx_syn = NodeTypeIndices(idx_syn_normalform1, idx_syn_normalform2, idx_syn_normalform3, idx_syn_load)




# # 4x2 format

# Now you can use idx.normalform1, idx.normalform2, idx.normalform3, and idx.load
plot_ieee_nf1 = make_subplot(df_ieee_nodes.surv[idx_ieee.normalform1], false, "NF1", plot_properties, color_pal[13], (0,.2), left_margin, right_margin, true, true, false, true)
plot_ieee_nf2 = make_subplot(df_ieee_nodes.surv[idx_ieee.normalform2], false, "NF2", plot_properties, color_pal[17], (0,.41), 0mm, right_margin, true, true, false, true)
plot_ieee_nf3 = make_subplot(df_ieee_nodes.surv[idx_ieee.normalform3], false, "NF3", plot_properties, color_pal[1], (0,.3), left_margin, right_margin, true, true, false, true)
plot_ieee_load = make_subplot(df_ieee_nodes.surv[idx_ieee.load], false, "load", plot_properties, color_pal[5], (0,.2), left_margin, right_margin, true, true, true, true)


plot_syn_nf1 = make_subplot(df_syn_nodes.surv[idx_syn.normalform1], false, "NF1", plot_properties, color_pal[13], (0,.2), left_margin, right_margin, true, true, false, true)
plot_syn_nf2 = make_subplot(df_syn_nodes.surv[idx_syn.normalform2], false, "NF2", plot_properties, color_pal[17], (0,.41), 0mm, right_margin, true, true, false, true)
plot_syn_nf3 = make_subplot(df_syn_nodes.surv[idx_syn.normalform3], false, "NF3", plot_properties, color_pal[1], (0,.3), left_margin, right_margin, true, true, false, true)
plot_syn_load = make_subplot(df_syn_nodes.surv[idx_syn.load], false, "load", plot_properties, color_pal[5], (0,.2), left_margin, right_margin, true, true, true, true)


plot_fig = Plots.plot(
    plot_ieee_nf1, plot_syn_nf1,
    plot_ieee_nf2, plot_syn_nf2,
    plot_ieee_nf3, plot_syn_nf3,
    plot_ieee_load, plot_syn_load,
    layout = Plots.grid(4, 2),
    size=(500, 750),
#     margin=1Plots.mm,
#     lw=3,  
)
Plots.savefig("./../pics/ieee_sn_hist_surv_4x2.pdf")


# plot_syn_nf = make_subplot(df_syn_nodes.surv[vcat(idx_syn.normalform1, idx_syn.normalform2, idx_syn.normalform3)], false, "NF", plot_properties, color_pal[13], (0,.2), left_margin, right_margin, true, true, false, true)

# # 2x4 format
plot_ieee_nf1 = make_subplot(df_ieee_nodes.surv[idx_ieee.normalform1], false, "NF1", plot_properties, color_pal[13], (0,.2), left_margin, right_margin, true, true, false, true)
plot_ieee_nf2 = make_subplot(df_ieee_nodes.surv[idx_ieee.normalform2], false, "NF2", plot_properties, color_pal[17], (0,.41), 0mm, right_margin, true, true, false, false)
plot_ieee_nf3 = make_subplot(df_ieee_nodes.surv[idx_ieee.normalform3], false, "NF3", plot_properties, color_pal[1], (0,.3), left_margin, right_margin, true, true, false, false)
plot_ieee_load = make_subplot(df_ieee_nodes.surv[idx_ieee.load], false, "load", plot_properties, color_pal[5], (0,.2), left_margin, right_margin, true, true, true, false)


plot_syn_nf1 = make_subplot(df_syn_nodes.surv[idx_syn.normalform1], false, "NF1", plot_properties, color_pal[13], (0,.2), left_margin, right_margin, true, true, true, true)
plot_syn_nf2 = make_subplot(df_syn_nodes.surv[idx_syn.normalform2], false, "NF2", plot_properties, color_pal[17], (0,.41), 0mm, right_margin, true, true, true, false)
plot_syn_nf3 = make_subplot(df_syn_nodes.surv[idx_syn.normalform3], false, "NF3", plot_properties, color_pal[1], (0,.3), left_margin, right_margin, true, true, true, false)
plot_syn_load = make_subplot(df_syn_nodes.surv[idx_syn.load], false, "load", plot_properties, color_pal[5], (0,.2), left_margin, right_margin, true, true, true, false)





plot_fig = Plots.plot(
    plot_ieee_nf1,  plot_ieee_nf2,  plot_ieee_nf3,  plot_ieee_load, 
    plot_syn_nf1, plot_syn_nf2, plot_syn_nf3, plot_syn_load,
   layout = Plots.grid(2, 4),
    size=(750, 500),
#     margin=1Plots.mm,
#     lw=3,  
)
Plots.savefig("./../pics/ieee_sn_hist_surv_2x4.pdf")
#plt.show()


