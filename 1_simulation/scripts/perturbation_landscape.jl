using Pkg
using GNN_BS_new_dataset
using PowerDynamics
using Statistics
using LinearAlgebra
using StatsBase
using Revise
## Read grid and state

datadir = joinpath(@__DIR__, "../data/")
pg, pg_state = read_grid_and_state("ieee", datadir)

slack_nodes = typeof.(pg.nodes) .== SlackAlgebraic
nf_nodes = typeof.(pg.nodes) .== NormalForm{1}
pq_nodes = typeof.(pg.nodes) .== PQAlgebraic

N = length(pg.nodes)
pg.nodes[42]

## Initial conditions

sobol_ic = get_sobol_ic(N)

afo_ic = load_afo_ic(joinpath(datadir, "dynamics_ieee.h5"), pg)

## Read simulation results

sim_results = load_sim_results(joinpath(datadir, "dynamics_ieee.h5"))

## Correlation stuff
u_r_pre, u_i_pre, ω_pre, v_pre, φ_pre = sobol_ic
u_r, u_i, ω, v, φ = afo_ic
survt, mafd, ΔP, ΔQ = sim_results

Δpre = sqrt.((u_r .- u_r_pre) .^ 2 .+
             (u_i .- u_i_pre) .^ 2)

[cor(mafd[i, :], Δpre[i, :]) for i = 1:N]
diag(cor(mafd, Δpre, dims=2))[.!slack_nodes]
diag(cor(mafd, Δpre, dims=2))

diag(cor(mafd, abs.(φ), dims=2))[.!slack_nodes] |> mean
diag(cor(mafd, abs.(v), dims=2))[.!slack_nodes] |> mean
diag(corspearman(mafd', abs.(v)'))[.!slack_nodes] |> mean
diag(cor(mafd, abs.(ω), dims=2))[nf_nodes] |> mean


## Plotting

# Check that the latest failing nodes do so before 10s

survt, mafd, ΔP, ΔQ = sim_results

@assert sort!([Set(survt)...])[end-1] < 10

if false
    save=false
    for i = 1:N
        plot_urui(i, sobol_ic, afo_ic, pg_state, pg; c=survt, save=save)
        plot_urui(i, sobol_ic, afo_ic, pg_state, pg; c=mafd, save=save, label="MAFD in Hz")
    end

    for i = 1:N
        plot_phiv(i, sobol_ic, afo_ic, pg_state, pg; c=survt, save=save)
        plot_phiv(i, sobol_ic, afo_ic, pg_state, pg; c=mafd, save=save, label="MAFD in Hz")
    end

    for i = 1:N
        plot_PQ(i, ΔP, ΔQ, pg; c=survt, save=save)
        plot_PQ(i, ΔP, ΔQ, pg; c=mafd, save=save, label="MAFD in Hz")
    end
end

paper_plot(ΔP, ΔQ, survt .== 120.0)
