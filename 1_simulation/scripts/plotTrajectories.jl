using Pkg
package_dir = joinpath(@__DIR__, "../")
Pkg.activate(package_dir)
using Revise
using Plots
# using Distributions
using OrdinaryDiffEq
using ModelingToolkit
using AmbientForcing
using LinearSolve
using LaTeXStrings
using GNN_BS_new_dataset
using PowerDynamics
using JSON: parsefile
# using Random
using Sobol
using HDF5
using OrderedCollections

## import functions
import GNN_BS_new_dataset: get_indices_maps, get_symbolic_problem, ambient_forcing_problem, get_initialized_return_values, get_slack_indices_maps, analyze_sol

const num_grids = 10000
const pert_per_node = 100

#     const grid_index_start = 1
#     const grid_index_end = 100 

const end_simulation_time = 120

const seed_summand = 0
const method = "All"
const plot_probability = 1.0

const ω_pert_size_low = convert(Float32, -2.0 * pi)
const ω_pert_size_high = convert(Float32, 2.0 * pi)
const u_pert_size = convert(Float32, 1.0)
const threshold_ω_out_of_bounds = 2.0 * 2.0 * pi

const sim_prop = GNN_BS_new_dataset.simulation_properties(end_simulation_time, num_grids, pert_per_node, method,
    seed_summand, ω_pert_size_low, ω_pert_size_high, u_pert_size, threshold_ω_out_of_bounds)

path_storage = joinpath(@__DIR__, "../data")




r = 1
name = "00001"
grid_directory = joinpath(path_storage, "grids/grid_")
grid_name = string(grid_directory, name, ".json")
file_state_name = string(path_storage, "/grids/state_", name, ".json")
pg = read_powergrid(grid_name, Json)
state_vec = parsefile(file_state_name; dicttype=Dict, inttype=Int64)
state_vec = convert(Array{Float64,1}, state_vec)
pg_state = State(pg, state_vec)
N = length(pg.graph.fadjlist)
pics_directory = joinpath(path_storage, "pics")



pert_per_node = sim_prop.pert_per_node
max_angular_frequency_dev, min_angular_frequency_dev, surv_vol_condition, infeasible, ω_excitability, voltage_excitability, P_diff_global, Q_diff_global, P_diff_local, Q_diff_local, final_diff_v, mfd_final, final_ω_excitability, final_voltage_excitability, survival_time = get_initialized_return_values(N, pert_per_node)

rpg = rhs(pg)
op = pg_state

indices, ω_idx, ur_idx, ui_idx, ω_idxs_map, v_idxs_map = get_indices_maps(pg, rpg, sim_prop.method)

soboli_perturbations = get_soboli_perturbations(sim_prop.ω_pert_size_low, sim_prop.ω_pert_size_high, pert_per_node)
prob_sparsejac = get_symbolic_problem(rpg, op, sim_prop.end_simulation_time)
afoprob = ambient_forcing_problem(prob_sparsejac.f, op.vec, 1.0, zeros(length(op.vec))) # τ = 1.0

nodes_wo_slack, map_idx_back = get_slack_indices_maps(pg, N)

P_set = map(x -> getfield(pg.nodes[x], :P), nodes_wo_slack)
Q_set = map(x -> getfield(pg.nodes[x], :Q), nodes_wo_slack)

all_x_new = Dict()
node = 18#3#18
all_x_new[node] = Dict()
idx = indices[node]
i = 17#63#17
x_new = random_perturbation(afoprob, soboli_perturbations[i, :], idx)
all_x_new[node][i] = x_new[idx]
x_pert = State(pg, x_new)
S_rated = x_pert[nodes_wo_slack, :s]


ΔP = real(S_rated) .- P_set
ΔQ = imag(S_rated) .- Q_set

# preparation for storage
P_diff_global[node, i] = sum(abs, ΔP)
Q_diff_global[node, i] = sum(abs, ΔQ)
P_diff_local[node, i] = ΔP[map_idx_back[node]]
Q_diff_local[node, i] = ΔQ[map_idx_back[node]]

sol = solve(
    remake(prob_sparsejac, u0=x_new),
    Rodas5(linsolve=KLUFactorization()),
    reltol=1e-6,
)

infeasible[node,i], first_nodes_ω_out_of_bounds, first_nodes_v_out_of_bounds, survival_time[node,i], final_nodes_ω_out_of_bounds, final_nodes_v_out_of_bounds = analyze_sol(node, i, sol, sim_prop, ω_idx, ω_idxs_map, ur_idx, ui_idx, v_idxs_map)
ω_excitability[first_nodes_ω_out_of_bounds] .+= 1
voltage_excitability[first_nodes_v_out_of_bounds] .+= 1


# access the whole time series and get the max / min frequency recorded at ANY node
max_angular_frequency_dev[node, i] = maximum(vec(sol[ω_idx, :]))
min_angular_frequency_dev[node, i] = minimum(vec(sol[ω_idx, :]))


if !isempty(first_nodes_v_out_of_bounds)
    surv_vol_condition[node, i] = 0
end


# saving final states
final_state = sol.u[end]
mfd_final[node, i] = maximum(abs.(final_state[ω_idx]))
v_final = abs.(final_state[ur_idx] .+ 1im .* final_state[ui_idx])
final_diff_v[node, i] = maximum(abs.((op[:, :v] .- v_final) ./ op[:, :v]))


final_ω_excitability[final_nodes_ω_out_of_bounds] .+= 1
final_voltage_excitability[final_nodes_v_out_of_bounds] .+= 1

# plotting
image_text = prepare_text_for_plot(max_angular_frequency_dev[node, i], min_angular_frequency_dev[node, i], surv_vol_condition[node, i], first_nodes_ω_out_of_bounds, first_nodes_v_out_of_bounds, final_nodes_ω_out_of_bounds, final_nodes_v_out_of_bounds, P_diff_global[node, i], Q_diff_global[node, i], P_diff_local[node, i], Q_diff_local[node, i], mfd_final[node, i], final_diff_v[node, i], infeasible[node, i], all_x_new[node][i], survival_time[node, i])
title_name = string("grid: ", r, " node: ", node, " pert: ", i)
title_name = ""
image_text = false
plot_res_log(sol,  pg, node, title_name, image_text, sim_prop.threshold_ω_out_of_bounds, low_voltage_ride_through(sol.t), high_voltage_ride_through(sol.t); plot_angular_ω = false, axis_lims = false)
filename = joinpath(pics_directory, string("grid_", r, "_node_", node, "_idx_", i, ".png"))
savefig(filename)


# end
GC.gc() # helps with the memory errors on slurm