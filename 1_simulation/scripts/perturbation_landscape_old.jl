using Pkg
using PyPlot
using HDF5
using JSON: parsefile
using Random
using Distributions
using PowerDynamics
using Sobol
using LaTeXStrings
using Statistics
using StatsBase
using LinearAlgebra
using GNN_BS_new_dataset: read_grid_and_state, get_soboli_perturbations

### Define utitility function

"""transform_ric(ric)
    Transforms random initial conditions to another coordinate system.
"""
function transform_ric(ric)
    u_r = ric[:, :, 1]
    u_i = ric[:, :, 2]
    ω = ric[:, :, 3] ./ 2π
    v = abs.(u_r .+ im .* u_i)
    φ = angle.(u_r .+ im .* u_i)
    return u_r, u_i, ω, v, φ
end

## Read grid and state

pg, pg_state = read_grid_and_state("00001", "/home/micha/git/SNBSdataset/data/")

slack_nodes = typeof.(pg.nodes) .== SlackAlgebraic
nf_nodes = typeof.(pg.nodes) .== NormalForm{1}
pq_nodes = typeof.(pg.nodes) .== PQAlgebraic


## Compute a vector of variable indices for every node
begin
    typeof(pg.nodes[1]) == NormalForm{1} ? idx_list = [1:3] : error()
    for i = 2:length(pg.nodes)
        if typeof(pg.nodes[i]) == NormalForm{1}
            push!(idx_list, range(idx_list[end][end] + 1, idx_list[end][end] + 3))
        elseif typeof(pg.nodes[i]) == PQAlgebraic
            push!(idx_list, range(idx_list[end][end] + 1, idx_list[end][end] + 2))
        end
    end
    insert!(idx_list, 23, 1:0)
end



## Sobol initial conditions
begin
    N = length(pg.nodes)
    pert_per_node = 1000
    const ω_pert_size_low = convert(Float32, -2.0 * pi)
    const ω_pert_size_high = convert(Float32, 2.0 * pi)
    const u_pert_size = convert(Float32, 1.0)

    soboli_perturbations = get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, pert_per_node; v_size_low=0.15^2, skipbool=false)

    ric_pre = repeat(reshape(soboli_perturbations, 1, pert_per_node, 3), 73)

    # Transform into different coordinate systems
    u_r_pre, u_i_pre, ω_pre, v_pre, φ_pre = transform_ric(ric_pre)
end




## Load initial conditions after ambient_forcing
h5open("data/dynamics_00001.h5", "r") do file1
    ric = ones(73, pert_per_node, 3) .* 42
    for i = keys(file1["x_new"]) .|> x -> parse(Int64, x)
        for j = 1:pert_per_node
            # if NF write three numbers
            # if PQ write two numbers
            # if slack write Nothing
            # default value 42 indicates processing errors
            ric[i, j, 1:length(idx_list[i])] .= read(
                file1["x_new"], "$(i)/$(j)")
        end
    end

    global u_r, u_i, ω, v, φ = transform_ric(ric)
end


## Read simulation results

h5open("data/dynamics_00001.h5", "r") do file
    svc = read(file, "surv_vol_condition")
    mafd = max.(
        abs.(read(file, "max_angular_frequency_dev")),
        abs.(read(file, "min_angular_frequency_dev"))) ./ 2π
    # Discard additional initial conditions (if any)
    global mafd = mafd[:, 1:pert_per_node]
end


## Correlation stuff

Δpre = sqrt.((u_r .- u_r_pre) .^ 2 .+
             (u_i .- u_i_pre) .^ 2)


[cor(mafd[i, :], Δpre[i, :]) for i = 1:73]
diag(cor(mafd, Δpre, dims=2))[.!slack_nodes]
diag(cor(mafd, Δpre, dims=2))

diag(cor(mafd, abs.(φ), dims=2))[.!slack_nodes]
diag(cor(mafd, abs.(v), dims=2))[.!slack_nodes]
diag(corspearman(mafd', abs.(v)'))[.!slack_nodes]
diag(cor(mafd, abs.(ω), dims=2))[nf_nodes]

## Plots


# Define label sizes
size1 = 14
size2 = 11
max_mafd = 2


for i = 1:10

    clf()
    scatter(φ_pre[i, :], v_pre[i, :], c="orange", label="initial perturbation", alpha=0.5, s=8)

    # Here we need to do the subtraction in the complex plane
    # complex_pert_pre = @. (u_r_pre[i, :] + im * u_i_pre[i, :]) - pg_state[i, :u]
    # scatter(angle.(complex_pert_pre), abs.(complex_pert_pre), c="red", label="Frand", alpha=0.5, s=8)

    scatter(φ[i, :], v[i, :], label="ambient forcing", c=mafd[i, :], vmin=0, vmax=max_mafd)

    scatter([pg_state[i, :φ]], [pg_state[i, :v]], color="tab:red", marker=:x, label="operation point")

    plt.xlabel(L"\varphi", fontsize=size1)
    plt.ylabel(L"v", fontsize=size1)
    plt.ylim([-0.1, 1.1])
    plt.clim(0, max_mafd)
    #plt.ylim([0.7,1.1])
    #plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    #plt.yticks([1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    cbar = plt.colorbar()
    tickrange = LinRange(0, max_mafd, 5)
    cbar.set_ticks(tickrange)
    cbar.set_ticklabels([tickrange[1:4]...,
        L"\geq %$(max_mafd)"])
    plt.title("Node $(i) $(typeof(pg.nodes[i]).name.name)")
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower left")
    mkpath("plots/phiv")
    filename = "plots/phiv/$(typeof(pg.nodes[i]).name.name)_$(i)_phi_v.png"
    # plt.savefig(filename, dpi=300)
    display(gcf())
end


for i = 1

    clf()
    scatter(u_r_pre[i, :], u_i_pre[i, :], c="orange", label="initial perturbation", alpha=0.5, s=8)

    # Here we need to do the subtraction in the complex plane
    # complex_pert_pre = @. (u_r_pre[i, :] + im * u_i_pre[i, :]) - pg_state[i, :u]
    # scatter(angle.(complex_pert_pre), abs.(complex_pert_pre), c="red", label="Frand", alpha=0.5, s=8)

    scatter(u_r[i, :], u_i[i, :], label="ambient forcing", c=mafd[i, :], vmin=0, vmax=max_mafd)

    scatter([pg_state[i, :u_r]], [pg_state[i, :u_i]], color="tab:red", marker=:x, label="operation point")

    plt.xlabel(L"real u", fontsize=size1)
    plt.ylabel(L"imag u", fontsize=size1)
    #plt.ylim([-0.1, 1.1])
    plt.clim(0, max_mafd)
    #plt.ylim([0.7,1.1])
    #plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    #plt.yticks([1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    cbar = plt.colorbar()
    tickrange = LinRange(0, max_mafd, 5)
    cbar.set_ticks(tickrange)
    cbar.set_ticklabels([tickrange[1:4]...,
        L"\geq %$(max_mafd)"])
    plt.title("Node $(i) $(typeof(pg.nodes[i]).name.name)")
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower left")
    mkpath("plots/urui")
    filename = "plots/urui/$(typeof(pg.nodes[i]).name.name)_$(i)_ur_ui.png"
    # plt.savefig(filename, dpi=300)
    display(gcf())
end



for i = 1:2
    clf()
    scatter(ω_pre[i, :], φ_pre[i, :], c="orange", label="initial perturbation", alpha=0.5, s=8)

    # Here we need to do the subtraction in the complex plane
    # complex_pert_pre = @. (u_r_pre[i, :] + im * u_i_pre[i, :]) - pg_state[i, :u]
    # scatter(angle.(complex_pert_pre), abs.(complex_pert_pre), c="red", label="Frand", alpha=0.5, s=8)

    scatter(ω[i, :], φ[i, :], label="ambient forcing", c=mafd[i, :], vmin=0, vmax=max_mafd)

    #scatter([pg_state[i, :φ]], [pg_state[i, :v]], color="tab:red", marker=:x, label="operation point")

    plt.xlabel(L"\omega", fontsize=size1)
    plt.ylabel(L"φ", fontsize=size1)
    #plt.ylim([-0.1, 1.1])
    plt.clim(0, max_mafd)
    #plt.ylim([0.7,1.1])
    #plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    #plt.yticks([1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    cbar = plt.colorbar()
    #cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    #cbar.set_ticklabels(["0", "0.5", "1.0", "1.5",        L"\geq 2.0"])
    plt.title("Node $(i) $(typeof(pg.nodes[i]).name.name)")
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower left")
    filename = "$(typeof(pg.nodes[i]).name.name)_$(i)_phi_v.png"
    #plt.savefig(filename, dpi=300)
    display(gcf())
end



## Worst pert per node

worst_pert_idx = argmax(mafd, dims=2)
begin
    clf()
    hist(vec(ω))
    display(gcf())
end
mafd[worst_pert_idx]


begin
    clf()
    scatter(φ_pre[worst_pert_idx[pq_nodes]], v_pre[worst_pert_idx[pq_nodes]], marker="s", color="coral", label="PQ Bus")
    scatter(φ_pre[worst_pert_idx[nf_nodes]], v_pre[worst_pert_idx[nf_nodes]], c=ω[worst_pert_idx[nf_nodes]], label="NormalForm", marker="x", facecolors="none", vmax=1, vmin=-1)
    plt.xlabel("φ", fontsize=size1)
    plt.ylabel("v", fontsize=size1)
    plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    plt.yticks([1.25, 1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    plt.title("Worst initial perturbation for each node")
    cbar = plt.colorbar(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.legend(loc="lower right")
    cbar.set_label(label="ω in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    #plt.savefig("worst_pert_phi_v.png", dpi=400)
    display(gcf())
end

# After AF

begin
    clf()
    scatter(φ[worst_pert_idx[pq_nodes]], v[worst_pert_idx[pq_nodes]], marker="s", color="coral", label="PQ Bus")
    scatter(φ[worst_pert_idx[nf_nodes]], v[worst_pert_idx[nf_nodes]], c=ω[worst_pert_idx[nf_nodes]], label="NormalForm", marker="x", facecolors="none", vmax=1, vmin=-1)
    plt.xlabel("φ", fontsize=size1)
    plt.ylabel("v", fontsize=size1)
    plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    plt.yticks([1.25, 1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    plt.title("Worst ambient forcing for each node")
    cbar = plt.colorbar(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.legend(loc="lower right")
    cbar.set_label(label="ω in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    #plt.savefig("worst_pert_phi_v.png", dpi=400)
    display(gcf())
end



begin
    clf()
    scatter(u_r_pre[worst_pert_idx[pq_nodes]], u_i_pre[worst_pert_idx[pq_nodes]], marker="s", color="coral", label="PQ Bus")
    scatter(u_r_pre[worst_pert_idx[nf_nodes]], u_i_pre[worst_pert_idx[nf_nodes]], c=ω[worst_pert_idx[nf_nodes]], label="NormalForm", marker="x", facecolors="none", vmax=1, vmin=-1)
    scatter([real(pg_state[i, :u]) for i = 1:73], [imag(pg_state[i, :u]) for i = 1:73], color="tab:red", label="operation points", marker=:x)
    plt.clim([-1, 1])
    plt.xlabel("real u", fontsize=size1)
    plt.ylabel("imag u", fontsize=size1)
    plt.xticks([-1, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    plt.yticks([-1, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    plt.title("Worst initial perturbation for each node")
    cbar = plt.colorbar(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.legend()
    cbar.set_label(label="ω in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    plt.savefig("plots/worst_pert_phi_v_initial.png", dpi=400)
    display(gcf())
end

#After AF
ω[worst_pert_idx[nf_nodes]]
begin
    clf()
    scatter(u_r[worst_pert_idx[pq_nodes]], u_i[worst_pert_idx[pq_nodes]], marker="s", color="coral", label="PQ Bus")
    scatter(u_r[worst_pert_idx[nf_nodes]], u_i[worst_pert_idx[nf_nodes]], c=ω[worst_pert_idx[nf_nodes]], label="NormalForm", marker="x", facecolors="none", vmax=1, vmin=-1)
    scatter([real(pg_state[i, :u]) for i = 1:73], [imag(pg_state[i, :u]) for i = 1:73], color="tab:red", label="operation points", marker=:x)
    plt.xlabel("real u", fontsize=size1)
    plt.ylabel("imag u", fontsize=size1)
    plt.clim([-1, 1])

    plt.xticks([-1, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    plt.yticks([-1, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    plt.title("Worst ambient forcing for each node")
    cbar = plt.colorbar(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.legend()
    cbar.set_label(label="ω in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    plt.savefig("plots/worst_pert_phi_v_ambient_forcing.png", dpi=400)
    display(gcf())
end

## All perturbations with high MAFD are "worst"

worst_pert_idx = findall(x -> x > 15, mafd)
begin
    clf()
    scatter(φ_pre[worst_pert_idx], v_pre[worst_pert_idx], marker="s", color="coral", label="PQ Bus")
    scatter(φ_pre[worst_pert_idx], v_pre[worst_pert_idx], c=ω[worst_pert_idx], label="NormalForm", marker="x", facecolors="none", vmax=1, vmin=-1)
    plt.xlabel("φ", fontsize=size1)
    plt.ylabel("v", fontsize=size1)
    plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    plt.yticks([1.25, 1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    plt.title("Perturbations with MAFD > 15 Hz")
    cbar = plt.colorbar(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.legend(loc="lower right")
    cbar.set_label(label="ω in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    #plt.savefig("worst_pert_phi_v.png", dpi=400)
    display(gcf())
end

begin
    clf()
    scatter(u_r_pre[worst_pert_idx], u_i_pre[worst_pert_idx], marker="s", color="coral", label="PQ Bus")
    scatter(u_r_pre[worst_pert_idx], u_i_pre[worst_pert_idx], c=ω[worst_pert_idx], label="NormalForm", marker="x", facecolors="none", vmax=1, vmin=-1)
    plt.xlabel("real u", fontsize=size1)
    plt.ylabel("imag u", fontsize=size1)
    plt.xticks([-1, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    plt.yticks([-1, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    plt.title("Perturbations with MAFD > 15 Hz")
    cbar = plt.colorbar(ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.legend(loc="lower right")
    cbar.set_label(label="ω in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    #plt.savefig("worst_pert_phi_v.png", dpi=400)
    display(gcf())
end





## Plots in the ΔP, ΔQ Plane are also possible

h5open("data/dynamics_00001.h5", "r") do file
    global ΔP = read(file, "P_diff_global") # local exists as well
    global ΔQ = read(file, "Q_diff_global")
end

maximum(ΔQ)

for i = 1:73
    clf()
    scatter(ΔP[i, :], ΔQ[i, :], label="ambient forcing", c=mafd[i, :], vmin=0, vmax=max_mafd)
    plt.clim(0, max_mafd)
    plt.xlim([-100, 3000])
    plt.ylim([-200, 6000])

    plt.xlabel(L"\Delta P", fontsize=size1)
    plt.ylabel(L"\Delta Q", fontsize=size1)

    cbar = plt.colorbar()
    tickrange = LinRange(0, max_mafd, 5)
    cbar.set_ticks(tickrange)
    cbar.set_ticklabels([tickrange[1:4]...,
        L"\geq %$(max_mafd)"])
    plt.title("Node $(i) $(typeof(pg.nodes[i]).name.name)")
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower right")
    mkpath("plots/DPDQ")
    filename = "plots/DPDQ/$(typeof(pg.nodes[i]).name.name)_$(i)_DP_DQ.png"
    plt.savefig(filename, dpi=300)
    display(gcf())
end


## Where does ambient forcing move the random initial conditions?
pert_for_arrows = 15

for i = 1:73
    clf()
    scatter(u_r_pre[i, 1:pert_for_arrows], u_i_pre[i, 1:pert_for_arrows], c="orange", label="initial perturbation", alpha=0.5, s=8)

    scatter(u_r[i, 1:pert_for_arrows], u_i[i, 1:pert_for_arrows], c=mafd[i, 1:pert_for_arrows], vmin=0, vmax=max_mafd, label="ambient forcing")

    scatter([real(pg_state[i, :u])], [imag(pg_state[i, :u])], color="tab:red", label="operation point", marker=:x)

    for j = 1:pert_for_arrows
        arrow(u_r_pre[i, j], u_i_pre[i, j], u_r[i, j] - u_r_pre[i, j], u_i[i, j] - u_i_pre[i, j],
            width=0.0001, length_includes_head=true,
            alpha=0.5, head_width=0.05)
    end

    plt.xlabel("real u", fontsize=size1)
    plt.ylabel("imag u", fontsize=size1)
    plt.clim(0, max_mafd)
    cbar = plt.colorbar()
    tickrange = LinRange(0, max_mafd, 5)
    cbar.set_ticks(tickrange)
    cbar.set_ticklabels([tickrange[1:4]...,
        L"\geq %$(max_mafd)"])
    plt.title("Node $(i) $(typeof(pg.nodes[i]).name.name)")
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower left")
    mkpath("plots/arrows")
    filename = "plots/arrows/$(typeof(pg.nodes[i]).name.name)_$(i)_ur_ui_arrows.png"
    plt.savefig(filename, dpi=300)
    display(gcf())
end

































## FROM HERE ON OLD PLOTTING CODE THAT WAS INTENDED FOR THE RANDOM INITAL CONDITION BEFORE  AmbientForcing AND WITH WRONG FRAND

## Reconstruct initial random samples

seed = MersenneTwister(1)
const ω_pert_size_low = convert(Float32, -2.0 * pi)
const ω_pert_size_high = convert(Float32, 2.0 * pi)
const u_pert_size = convert(Float32, 1.0)
# const threshold_ω_out_of_bounds = 2.0 * 2.0 * pi


begin # Random initial conditions
    N = 73
    pert_per_node = 1000

    rand_r_values = sqrt.(rand(seed, Uniform(0, u_pert_size), N, pert_per_node, 1))
    rand_φ_values = rand(seed, Uniform(0, 2π), N, pert_per_node, 1)
    rand_ur_values = rand_r_values .* cos.(rand_φ_values)
    rand_ui_values = rand_r_values .* sin.(rand_φ_values)
    rand_ω_values = rand(
        seed,
        Uniform(ω_pert_size_low, ω_pert_size_high),
        N,
        pert_per_node,
        1,
    )
    ## Transform into different coordinate systems
    # Some plotting code below assumes that the array of random initial conditions is called `ric`

    ric_pre = [rand_ur_values;;; rand_ui_values;;; rand_ω_values]
    #ric_pre = ric_pre[:, 1:100, :] # Discard for compatibility with new simulations
end


# Test that i am doing the coordinate transformation right
#sum(@.angle(u_r_pre[i, :] + im * u_i_pre[i, :]) .== φ_pre[i, :]) == 100
#sum(@.abs(u_r_pre[i, :] + im * u_i_pre[i, :]) .== v_pre[i, :]) == 100


# Could easily be adapted!

begin
    clf()
    # Old and incorrect Frand
    scatter(real(pg_state[i, :u]) .- u_r[i, :], imag(pg_state[i, :u]) .- u_i[i, :], c=mafd[i, :])
    scatter([0.0], [0.0], color="tab:red", label="operation point")
    plt.xlabel("u_r", fontsize=size1)
    plt.ylabel("u_i", fontsize=size1)
    #plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    cbar = plt.colorbar()
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower right")
    #plt.savefig("PQ_4_U.png", dpi=400)
    display(gcf())
end


begin
    clf()
    complex_pert = @. pg_state[i, :u] - (u_r[i, :] + im * u_i[i, :])


    scatter(angle.(complex_pert), abs.(complex_pert), c="tab:red")
    complex_pert = @. v[i, :] * exp(im * φ[i, :])
    scatter(angle.(complex_pert), abs.(complex_pert), c="tab:green")
    #scatter([0.0], [0.0], color="tab:red", label="operation point")
    #plt.xlabel("φ", fontsize=size1)
    #plt.ylabel("v", fontsize=size1)
    #plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    #plt.yticks([1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    cbar = plt.colorbar()
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="upper right")
    #plt.savefig("PQ_4_phi_v.png", dpi=400)
    display(gcf())
end


complex_pert = @. v[i, :] * exp(im * φ[i, :])
complex_state =
# nodes 1 and 2 are normal forms. we plot their response in the u_r-u_i, the u_r-omega and the u_i-omega planes.
    i = 2
begin
    clf()
    scatter(φ[i, :], v[i, :], c=mafd[i, :])
    scatter([pg_state[i, :φ]], [pg_state[i, :v]], color="tab:red", label="operation point")
    plt.xlabel("φ", fontsize=size1)
    plt.ylabel("v", fontsize=size1)
    plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    plt.yticks([1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    cbar = plt.colorbar()
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower right")
    #plt.savefig("NF_2_phi_v.png", dpi=400)
    display(gcf())
end

begin
    clf()
    scatter(φ[i, :], ω[i, :], c=mafd[i, :])
    scatter([pg_state[i, :φ]], [0.0], color="tab:red", label="operation point")
    plt.xlabel("φ", fontsize=size1)
    plt.ylabel("ω", fontsize=size1)
    plt.xticks([-3, -1.5, 0.0, 1.5, 3.0], fontsize=size2)
    plt.yticks([-1, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    cbar = plt.colorbar()
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower right")
    #plt.savefig("NF_2_phi_omega.png", dpi=400)
    display(gcf())
end

begin
    clf()
    scatter(ω[i, :], v[i, :], c=mafd[i, :])
    scatter([0.0], [pg_state[i, :v]], color="tab:red", label="operation point")
    plt.xlabel("ω", fontsize=size1)
    plt.ylabel("v", fontsize=size1)
    plt.yticks([1.0, 0.75, 0.50, 0.25, 0.0], fontsize=size2)
    plt.xticks([-1, -0.5, 0.0, 0.5, 1.0], fontsize=size2)
    cbar = plt.colorbar()
    cbar.set_label(label="MAFD in Hz", fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    leg = plt.legend(fontsize=size2, loc="lower right")
    #plt.savefig("NF_2_omega_phi.png", dpi=400)
    display(gcf())
end

# Plots in the ΔP, ΔQ Plane are also possible
ΔP = read(file, "P_diff")
ΔQ = read(file, "Q_diff")


# What do these sign mean?
scatter(ΔP[i, :], ΔQ[i, :], marker_z=mfd[i, :])

scatter(ΔP[i, :] .* sign.(ric[i, :, 2]), ΔQ[i, :], marker_z=mfd[i, :])
scatter(ΔP[i, :] .* sign.(ric[i, :, 2]), ΔQ[i, :] .* sign.(ric[i, :, 3]), marker_z=mfd[i, :])
