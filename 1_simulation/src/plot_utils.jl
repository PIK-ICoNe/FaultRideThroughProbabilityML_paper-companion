using PyPlot
using HDF5
using JSON: parsefile
using Random
using Distributions
using PowerDynamics
using Sobol
using LaTeXStrings
using StatsBase
using LinearAlgebra

"""transform_ic(ric)
    Transforms random initial conditions to other coordinate systems.
"""
function transform_ic(ric)
    u_r = ric[:, :, 1]
    u_i = ric[:, :, 2]
    ω = ric[:, :, 3] ./ 2π
    v = abs.(u_r .+ im .* u_i)
    φ = angle.(u_r .+ im .* u_i)
    return (;u_r, u_i, ω, v, φ)
end

function get_sobol_ic(
    N;
    pert_per_node=1000,
    ω_pert_size_low = convert(Float32, -2.0 * pi),
    ω_pert_size_high = convert(Float32, 2.0 * pi),
    u_pert_size = convert(Float32, 1.0),
    )
    _ic_sob = get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, pert_per_node)
    # It would be more efficient to do the transformation
    # before the repetition
    _ic_sob = repeat(reshape(_ic_sob, 1, pert_per_node, 3), N)
    return transform_ic(_ic_sob)
end

## Compute a vector of variable indices for every node
function get_idx_list(pg::PowerGrid)
    @assert Set(typeof.(pg.nodes)) == Set([SlackAlgebraic, NormalForm{1}, PQAlgebraic])
    _i = 1
    if typeof(pg.nodes[_i]) == NormalForm{1}
        idx_list = [1:3]
    elseif typeof(pg.nodes[_i]) == PQAlgebraic
        idx_list = [1:2]
    elseif typeof(pg.nodes[_i]) == SlackAlgebraic
        idx_list = [1:0]
        _i += 1
        if typeof(pg.nodes[_i]) == NormalForm{1}
            push!(idx_list, [1:3])
        elseif typeof(pg.nodes[_i]) == PQAlgebraic
            push!(idx_list, [1:2])
        end
    end
    _i += 1

    for i = _i:length(pg.nodes)
        if typeof(pg.nodes[i]) == NormalForm{1}
            push!(idx_list, range(idx_list[end][end] + 1, idx_list[end][end] + 3))
        elseif typeof(pg.nodes[i]) == PQAlgebraic
            push!(idx_list, range(idx_list[end][end] + 1, idx_list[end][end] + 2))
        end
    end
    slack_nodes = findall(typeof.(pg.nodes) .== SlackAlgebraic)
    length(slack_nodes) == 1 ? nothing : error("Too many slack nodes.")
    insert!(idx_list, slack_nodes[1], 1:0)
    return idx_list
end

function load_afo_ic(filepath, pg; pert_per_node=1000)
    N = length(pg.nodes)
    idx_list = get_idx_list(pg)
    ## Load initial conditions after ambient_forcing
    h5open(filepath, "r") do file1
        ic = ones(N, pert_per_node, 3) .* 42
        for i = keys(file1["x_new"]) .|> x -> parse(Int64, x)
            for j = 1:pert_per_node
                # if NF write three numbers
                # if PQ write two numbers
                # if slack write Nothing
                # default value 42 indicates processing errors
                ic[i, j, 1:length(idx_list[i])] .= read(
                    file1["x_new"], "$(i)/$(j)")
            end
        end
        return transform_ic(ic)
    end
end


function load_sim_results(filepath; pert_per_node=1000)
    h5open(filepath, "r") do file
        # Discard additional initial conditions (if any)
        survt = read(file, "survival_time")[:, 1:pert_per_node]
        mafd = max.(
            abs.(read(file, "max_angular_frequency_dev")),
            abs.(read(file, "min_angular_frequency_dev"))) ./ 2π

        mafd = mafd[:, 1:pert_per_node]
        ΔP = read(file, "P_diff_global")[:, 1:pert_per_node] # local exists as well
        ΔQ = read(file, "Q_diff_global")[:, 1:pert_per_node]

        return (; survt, mafd, ΔP, ΔQ)
    end
end
## Plotting functions
# There is some code duplication in the plotting funciton which might be improved by using a common entry point

function plot_phiv(
    i, sobol_ic, afo_ic, pg_state, pg;
    c, max_c=10, save=false, label="survival time in s", size1=14, size2=11
)
    u_r_pre, u_i_pre, ω_pre, v_pre, φ_pre = sobol_ic
    u_r, u_i, ω, v, φ = afo_ic

    if label == "survival time in s"
        _cmap = "cividis_r"
    else
        _cmap = "cividis"
    end

    clf()
    plt.scatter(φ_pre[i, :], v_pre[i, :], c="orange", label="initial perturbation", alpha=0.5, s=8)
    plt.scatter(φ[i, :], v[i, :], label="ambient forcing", c=c[i, :], cmap=_cmap, vmin=0, vmax=max_c)
    plt.scatter([pg_state[i, :φ]], [pg_state[i, :v]], color="tab:red", marker=:x, label="operation point")
    plt.xlabel(L"\varphi", fontsize=size1)
    plt.ylabel(L"v", fontsize=size1)
    plt.ylim([-0.1, 1.1])
    plt.clim(0, max_c)
    cbar = plt.colorbar()
    tickrange = LinRange(0, max_c, 5)
    cbar.set_ticks(tickrange)
    cbar.set_ticklabels([tickrange[1:4]...,
        L"\geq %$(max_c)"])
    plt.title("Node $(i) $(typeof(pg.nodes[i]).name.name)")
    cbar.set_label(label=label, fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    plt.legend(fontsize=size2, loc="lower left")

    if save
        savepath = "plots/phiv_$(label[1:4])"
        mkpath(savepath)
        filename = savepath * "/$(typeof(pg.nodes[i]).name.name)_$(i)_$(label[1:4])_phi_v.png"
        plt.savefig(filename, dpi=300)
    end
    display(gcf())
end

function plot_PQ(
    i, ΔP, ΔQ, pg;
    c, max_c=10, save=false, label="survival time in s", size1=14, size2=11
)
    if label == "survival time in s"
        _cmap = "cividis_r"
    else
        _cmap = "cividis"
    end

    clf()
    plt.scatter(ΔP[i, :], ΔQ[i, :], label="ambient forcing", c=c[i, :], cmap=_cmap, vmin=0, vmax=max_c)
    plt.clim(0, max_c)

    plt.xlabel(L"\Delta P", fontsize=size1)
    plt.ylabel(L"\Delta Q", fontsize=size1)

    cbar = plt.colorbar()
    tickrange = LinRange(0, max_c, 5)
    cbar.set_ticks(tickrange)
    cbar.set_ticklabels([tickrange[1:4]...,
        L"\geq %$(max_c)"])
    plt.title("Node $(i) $(typeof(pg.nodes[i]).name.name)")
    cbar.set_label(label=label, fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    plt.legend(fontsize=size2, loc="lower right")

    if save
        savepath = "plots/PQglobal_$(label[1:4])"
        mkpath(savepath)
        filename = savepath * "/$(typeof(pg.nodes[i]).name.name)_$(i)_DP_DQ.png"
        plt.savefig(filename, dpi=300)
    end
    display(gcf())
end


function plot_urui(
    i, sobol_ic, afo_ic, pg_state, pg;
    c, max_c=10, save=false, label="survival time in s", size1=14, size2=11
)
    u_r_pre, u_i_pre, ω_pre, v_pre, φ_pre = sobol_ic
    u_r, u_i, ω, v, φ = afo_ic

    if label == "survival time in s"
        _cmap = "cividis_r"
    else
        _cmap = "cividis"
    end

    clf()
    plt.scatter(u_r_pre[i, :], u_i_pre[i, :], c="orange", label="initial perturbation", alpha=0.5, s=8)

    plt.scatter(u_r[i, :], u_i[i, :], label="ambient forcing", c=c[i, :], cmap=_cmap, vmin=0, vmax=max_c)
    plt.scatter([pg_state[i, :u_r]], [pg_state[i, :u_i]], color="tab:red", marker=:x, label="operation point")
    plt.xlabel("real u", fontsize=size1)
    plt.ylabel("imag u", fontsize=size1)
    plt.clim(0, max_c)
    cbar = plt.colorbar()
    tickrange = LinRange(0, max_c, 5)
    cbar.set_ticks(tickrange)
    cbar.set_ticklabels([
        tickrange[1:4]..., L"\geq %$(max_c)",
        ]
    )
    plt.title("Node $(i) $(typeof(pg.nodes[i]).name.name)")
    cbar.set_label(label=label, fontsize=size1)
    cbar.ax.tick_params(labelsize=size2)
    plt.legend(fontsize=size2, loc="lower left")
    if save
        savepath = "plots/urui_$(label[1:4])"
        mkpath(savepath)
        filename = savepath * "/$(typeof(pg.nodes[i]).name.name)_$(i)_$(label[1:4])_ur_ui.png"
        plt.savefig(filename, dpi=300)
    end
    display(gcf())
end


function paper_subplot!(
    ax, ΔP, ΔQ, surved;
    i, label, size1=15, size2=12, col=1, s=50
)

    ax.scatter(
        ΔP[i, findall(.!surved[i, :])],
        ΔQ[i, findall(.!surved[i, :])],
        c=[get_cmap("tab20b")(col + 3)],
        #alpha = 0.25,
        s=s,
    )

    ax.scatter(
        ΔP[i, findall(surved[i, :])],
        ΔQ[i, findall(surved[i, :])],
        c=[get_cmap("tab20b")(col + 1)],
        s=s,
    )

    ax.set_xlabel(L"\Delta P", fontsize=size1)
    ax.set_ylabel(L"\Delta Q", fontsize=size1)

    ax.tick_params(axis="both", which="major", labelsize=size2)

    ax.text(
        0.98, 0.98, label,
        transform=ax.transAxes,
        fontsize=size1,
        horizontalalignment="right", verticalalignment="top",
        #bbox=Dict(:facecolor => "grey", :alpha=>0.15)
    )
    return nothing
end

function paper_plot(ΔP, ΔQ, surved; format::String=".pdf",save=true)
    clf()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        dpi=150,
        figsize=( 6.4, 4 * 4.8),
        #figsize=(9 * 6.4, 3 * 4.8),
    )

    #ax1.set_box_aspect(1)
    #ax2.set_box_aspect(1)
    #ax3.set_box_aspect(1)

    paper_subplot!(
        ax1, ΔP, ΔQ, surved,
        i=19,
        label=L"load",
        col=4
    )
    paper_subplot!(
        ax2, ΔP, ΔQ, surved,
        i=50,
        label=L"NF1",
        col=12,
    )
    paper_subplot!(
        ax3, ΔP, ΔQ, surved,
        i=42,
        label=L"NF3",
        col=0,
    )


    ax3.legend(
        ["unstable", "stable"],
        bbox_to_anchor=(0.5, -0.3),
        loc="lower center",
        ncol=2,
        fontsize=12
    )

    if save
        fig.savefig(
            "plots/pert_landscape_paper" * format,
            bbox_inches = "tight",
            pad_inches = 0,
        )
    end

    display(gcf())

    return nothing
end
