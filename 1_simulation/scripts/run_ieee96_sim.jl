using Pkg
package_dir = joinpath(@__DIR__, "../")
Pkg.activate(package_dir)
using GNN_BS_new_dataset

using LibGit2
r = LibGit2.GitRepo(package_dir)
println("Git branch: ",LibGit2.branch(r))
println("GitHash: ",LibGit2.GitHash(LibGit2.head(r)))
println("GIT local changes ", LibGit2.isdirty(r))

using PowerDynamics
using JSON: parsefile
using Random
using HDF5
using OrderedCollections

const num_grids = 10000
const pert_per_node = 3

const end_simulation_time = 120

const seed_summand = 0
const method = "All"
const plot_probability = 0.1

const ω_pert_size_low = convert(Float32, -2.0 * pi)
const ω_pert_size_high = convert(Float32, 2.0 * pi)
const u_pert_size = convert(Float32, 1.0)
const threshold_ω_out_of_bounds = 2.0 * 2.0 * pi

const sim_prop = GNN_BS_new_dataset.simulation_properties(end_simulation_time, num_grids, pert_per_node, method,
    seed_summand, ω_pert_size_low, ω_pert_size_high, u_pert_size, threshold_ω_out_of_bounds)

path_storage = joinpath(@__DIR__, "../data")

function simulate_grid(r, name)
    grid_directory = joinpath(path_storage, "grids/grid_")
    grid_name = string(grid_directory, name, ".json")
    file_state_name = string(path_storage, "/grids/state_", name, ".json")
    pg = read_powergrid(grid_name, Json)
    state_vec = parsefile(file_state_name; dicttype=Dict, inttype=Int64)
    state_vec = convert(Array{Float64,1}, state_vec)
    pg_state = State(pg, state_vec)
    N = length(pg.graph.fadjlist)
    pics_directory = joinpath(path_storage, "pics")
    computational_effort = @timed ds_result = dynamic_simulation(r, N, pg, pg_state, sim_prop, plot_probability, pics_directory) # t.time t.bytes timed? ersten 3 Eintraege, save elapsed time
    dynamics_directory = joinpath(path_storage, "dynamics/")
    store_dynamics(name, dynamics_directory, computational_effort, ds_result...)
end

@time simulate_grid(1, "00001")