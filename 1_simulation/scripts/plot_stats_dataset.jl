using Revise

using Pkg
package_dir = joinpath(@__DIR__, "../gnn_bs_new_dataset")
Pkg.activate(package_dir)


using HDF5
using Plots
using Plots.PlotMeasures
using PowerDynamics
using DataFrames
using GNN_BS_new_dataset

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

thresholds_1 = all_thresholds(
    -2. * 2*pi, #threshold_Ï‰_surv_low
     2. * 2*pi, # threshold_Ï‰_surv_high
    .1,
    .1
);

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

function make_subplot(plot_data, title, plot_properties, color, y_lims,left_margin, right_margin = 0mm, xaxis = false, yaxis=false, xlabel=false, ylabel=false)
    nBins = plot_properties["nBins"]
    x_label = plot_properties["xlabel"]
    y_label = plot_properties["ylabel"]
    fsGuide = plot_properties["fontSizeGuide"]
    fsTitle = plot_properties["fontSizeTitle"]
    fsTick = plot_properties["fontSizeTick"]
    x_lims = plot_properties["xlims"]
    bottom_margin=0mm
    fig = histogram(plot_data, normalize=:probability, color=color, xlims = x_lims, y_lims = y_lims, bins=nBins, xaxis = xaxis, yaxis=yaxis, grid =false, bottom_margin = bottom_margin, right_margin = right_margin, top_margin = -.2mm, left_margin = left_margin, fg_legend = :false, label=title)    
    plot!(xtickfontsize=7,ytickfontsize=7, xlabelfontsize=9, ylabelfontsize=9)
    if xlabel == true
        xlabel!("survivability")
    end
    if ylabel == true
        ylabel!("normalized density")
    end
    return fig
end

grid_ieee = get_statistical_results("../ieee/data/",1,1, thresholds_1);
println("grid_ieee_surv_mean: ", mean(grid_ieee.statistics["surv"]))
grids70_80 = get_statistical_results("../70-80nodes/data/",1,1000, thresholds_1);
println("grids70_80_mean: ", mean(grids70_80.statistics["surv"]))


right_margin = -0mm
left_margin = 2mm
color_pal = palette(:tab20b)


result_setup = grid_ieee
stats = result_setup.statistics
idx = result_setup.indices_node_type
plot_ieee_nf1 = make_subplot(stats["surv"][idx.normalform1], false, plot_properties, color_pal[13], (0,.2), left_margin, right_margin, true, true,false, true)
plot_ieee_nf2 = make_subplot(stats["surv"][idx.normalform2], false, plot_properties, color_pal[17], (0,.41), 0mm, right_margin, true, true, false, true)
plot_ieee_nf3 = make_subplot(stats["surv"][idx.normalform3], false, plot_properties, color_pal[1], (0,.3), left_margin, right_margin, true, true, false, true)
plot_ieee_load = make_subplot(stats["surv"][idx.load], false, plot_properties, color_pal[5], (0,.2), left_margin, right_margin, true, true, true, true)

result_setup = grids70_80
stats = result_setup.statistics
idx = result_setup.indices_node_type
plot_sn_nf1 = make_subplot(stats["surv"][idx.normalform1], "NF1", plot_properties, color_pal[13], (0,.2), left_margin, 0mm, true, true, false, false)
plot_sn_nf2 = make_subplot(stats["surv"][idx.normalform2], "NF2", plot_properties, color_pal[17], (0,.41), 0mm, 0mm, true, true, false, false)
plot_sn_nf3 = make_subplot(stats["surv"][idx.normalform3], "NF3", plot_properties, color_pal[1], (0,.3), left_margin, 0mm, true, true, false, false)
plot_sn_load = make_subplot(stats["surv"][idx.load], "load", plot_properties, color_pal[5], (0,.2), left_margin, 0mm, true, true, true, false)


plot_fig = Plots.plot(
    plot_ieee_nf1, plot_sn_nf1,
    plot_ieee_nf2, plot_sn_nf2,
    plot_ieee_nf3, plot_sn_nf3,
    plot_ieee_load, plot_sn_load,
    layout = Plots.grid(4, 2),
    size=(500, 750),
#     margin=1Plots.mm,
#     lw=3,  
)

Plots.savefig("pics/ieee_sn_hist_surv_4x2.pdf")
#plt.show()
