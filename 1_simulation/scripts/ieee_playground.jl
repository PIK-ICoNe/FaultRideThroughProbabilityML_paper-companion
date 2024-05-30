using Pkg
package_dir = joinpath(@__DIR__, "../")
Pkg.activate(package_dir)
using Revise
using GNN_BS_new_dataset
using PowerDynamics
using SyntheticPowerGrids
import SyntheticPowerGrids.validate_power_flow_on_lines
using Plots
using Colors
default(grid = false, foreground_color_legend = nothing, bar_edges = false, lw = 3, framestyle =:box, msc = :auto, dpi=300, legendfontsize = 11, labelfontsize = 12, tickfontsize = 10)

##
ieee96_adapted, df = get_ieee_96(line_paras = :Dena_380kV, virtual_inertia = :SyntheticPowerGrids, reactive_power = :perfect_voltage_mag) # Linearly Stable??
op_adapted = find_operationpoint(ieee96_adapted)

p1 = histogram(op_adapted[:, :v], xaxis = "V [p.u.]", lw = 0.0, c = colorant"coral", label = "Perfect V", bins = 20, xlims = [0.975, 1.06], ylims = [0, 19])

ieee96_adapted, df = get_ieee_96(line_paras = :Dena_380kV, virtual_inertia = :SyntheticPowerGrids, reactive_power = :Ieee96) # Linearly Stable??
op_adapted = find_operationpoint(ieee96_adapted)

p2 = histogram(op_adapted[:, :v], xaxis = "V [p.u.]", lw = 0.0, c = colorant"teal", label = "Ieee 96", bins = 20, xlims = [0.975, 1.06], ylims = [0, 19])

plt = Plots.plot(p1, p2; layout = (2,1), size = (500, 500))

savefig(plt, "voltage_magnitudes_ieee_Q_planning.png")
