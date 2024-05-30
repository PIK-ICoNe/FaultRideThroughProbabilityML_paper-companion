using Pkg
package_dir = joinpath(@__DIR__, "../")
Pkg.activate(package_dir)
using Revise
using GNN_BS_new_dataset

using LibGit2
r = LibGit2.GitRepo(package_dir)
println("Git branch: ",LibGit2.branch(r))
println("GitHash: ",LibGit2.GitHash(LibGit2.head(r)))
println("GIT local changes ", LibGit2.isdirty(r))

using PowerDynamics
using SyntheticPowerGrids


##
# parameters
const N_lower_bound = 70
const N_upper_bound = 80
const num_grids = 10000
const grid_index_start = 1
const grid_index_end = 3
const P0_offpeak = 1.3
const P0_onpeak = 3.18

nodal_parameters_a = Dict(:τ_Q => 8.0, :K_P => 5, :K_Q => 0.1, :τ_P => 5.0)
nodal_parameters_b = Dict(:τ_Q => 8.0, :K_P => 5, :K_Q => 0.1, :τ_P => 1.0)
nodal_parameters_c = Dict(:τ_Q => 8.0, :K_P => 5, :K_Q => 0.1, :τ_P => 0.5)

nodal_dynamics = [
    (1 / 6, get_DroopControlledInverterApprox, nodal_parameters_a),
    (1 / 6, get_DroopControlledInverterApprox, nodal_parameters_b),
    (1 / 6, get_DroopControlledInverterApprox, nodal_parameters_c),
    (0.5, get_PQ, nothing),
]

# get a list containing the grid size using uniform sampling
random_N = rand(N_lower_bound:N_upper_bound, num_grids)
random_P0 = rand(P0_offpeak:P0_onpeak, num_grids)

path_storage = joinpath(@__DIR__, "../data")

function generate_grid(r)
    c = SyntheticPowerGrids.PGGeneration(
        num_nodes=random_N[r],
        nodal_dynamics=nodal_dynamics,
        lines=:PiModelLine,
        slack=true,
        P0=random_P0[r],
    )
    grid_generation(r, c, num_grids, random_P0[r], path_storage)
    return r
end
@time map(x-> generate_grid(x), grid_index_start:grid_index_end)
