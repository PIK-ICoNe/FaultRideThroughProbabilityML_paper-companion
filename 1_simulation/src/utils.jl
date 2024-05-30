function get_voltage_mag(sol::ODESolution, syms, node::Int64)
    u_r_idx = findfirst(s -> occursin(Regex("u_r_$node\$"), String(s)), syms)
    u_i_idx = findfirst(s -> occursin(Regex("u_i_$node\$"), String(s)), syms)

    @assert u_r_idx !== nothing
    @assert u_i_idx !== nothing

    v = sqrt.(sol[u_r_idx, :] .^ 2 + sol[u_i_idx, :] .^ 2)
    return v
end

function get_voltage_mag(sol::ODESolution, syms, nodes::Vector{Int64})
    V = map(node -> get_voltage_mag(sol, syms, node), nodes)
    return V
end

function get_frequency(sol::ODESolution, syms, node::Int64, angular=false)
    ω_idx = findfirst(s -> occursin(Regex("x_1_$node\$"), String(s)), syms)

    @assert ω_idx !== nothing    
    f = sol[ω_idx, :]
    if !angular
        f = f ./ (2π)
    end
    return f
end

function get_frequency(sol::ODESolution, syms, nodes::Vector{Int64}, angular=false)
    f = map(node -> get_frequency(sol, syms, node, angular), nodes)
    return f
end

"""
    prepare_text_for_plot(max_angular_frequency_dev, min_angular_frequency_dev, surv_vol_condition, first_nodes_ω_out_of_bounds, first_nodes_v_out_of_bounds, final_nodes_ω_out_of_bounds, final_nodes_v_out_of_bounds,  P_diff_global, Q_diff_global, P_diff_local, Q_diff_local, mfd_final, final_diff_v, infeasible, all_x_new, survival_time, digits=3)

Preparing the text dictionary that is used for plotting
- `max_angular_frequency_dev`
- `min_angular_frequency_dev`
- `surv_vol_condition`
- `first_nodes_ω_out_of_bounds`
- `first_nodes_v_out_of_bounds`
- `final_nodes_ω_out_of_bounds`
- `final_nodes_v_out_of_bounds`
- `P_diff_global`
- `Q_diff_global`
- `P_diff_local`
- `Q_diff_local`
- `mfd_final`
- `final_diff_v`
- `infeasible`
- `all_x_new`
- `survival_time`
- `digits` number of digits for rounding, default: 3
"""
function prepare_text_for_plot(max_angular_frequency_dev, min_angular_frequency_dev, surv_vol_condition, first_nodes_ω_out_of_bounds, first_nodes_v_out_of_bounds, final_nodes_ω_out_of_bounds, final_nodes_v_out_of_bounds,  P_diff_global, Q_diff_global, P_diff_local, Q_diff_local, mfd_final, final_diff_v, infeasible, all_x_new, survival_time, digits=3)
    image_text = Dict(
        1 =>  "max_angular_frequency_dev: " * string(round(max_angular_frequency_dev, digits=digits)),
        2 => "min_angular_frequency_dev: " * string(round(min_angular_frequency_dev, digits=digits)),
        3 => "surv_vol_condition: " * string(round(surv_vol_condition, digits=digits)),
        4 => "first_nodes_ω_out_of_bounds: " * string(first_nodes_ω_out_of_bounds),
        5 => "first_nodes_v_out_of_bounds: " * string(first_nodes_v_out_of_bounds),
        6 => "final_nodes_ω_out_of_bounds: " * string(final_nodes_ω_out_of_bounds),
        7 => "final_nodes_v_out_of_bounds: " * string(final_nodes_v_out_of_bounds),
        8 => "P_diff_global: " * string(round(P_diff_global, digits=digits)),
        9 => "Q_diff_global: " * string(round(Q_diff_global, digits=digits)),
        10 => "P_diff_local: " * string(round(P_diff_local, digits=digits)),
        11 => "Q_diff_local: " * string(round(Q_diff_local, digits=digits)),
        12 => "mfd_final: " * string(round(mfd_final, digits=digits)),
        13 => "final_diff_v: " * string(round(final_diff_v, digits=digits)),
        14 => "infeasible: " * string(infeasible),
        15 => "all_x_new[node][i]: " * string(round.(all_x_new,digits=digits)),
        16 => "survival_time: " * string(round(survival_time, digits=digits)))
    return image_text
end
"""
    plot_res_log(sol::ODESolution, pg, perturbed_node, title_name, image_text, threshold_ω_out_of_bounds, low_voltage_ride_through, high_voltage_ride_through; plot_angular_ω, axis_lims)

Plots the result of a ODESolution using a logarithmic time scale.
- `sol::ODESolution`: ODESolution
- `pg`: power grid
- `perturbed_node`: node idx of perturbed node
- `title_name`: plot title
- `image_text`: text that shows simulation results below the images, if false no text is shown
- `threshold_ω_out_of_bounds`: threshold_ω_out_of_bounds
- `low_voltage_ride_through`: low_voltage_ride_through for plotting limits
- `high_voltage_ride_through`: high_voltage_ride_through for plotting limits
- `plot_angular_ω`: Bool, if true, ω [rad/s] is shown, if false, f in Hz
- `axis_lims`: Bool, if true: fixed ylims on axis are used
"""
function plot_res_log(sol::ODESolution, pg, perturbed_node, title_name, image_text, threshold_ω_out_of_bounds, low_voltage_ride_through, high_voltage_ride_through; plot_angular_ω, axis_lims)
    n = length(pg.nodes)
    t = sol.t .+ 1.0 # for logarithmic axis
    syms = vcat(map(node -> string.(symbolsof(pg.nodes[node])) .* "_$node", 1:n)...)

    nf_nodes = findall(typeof.(pg.nodes) .== NormalForm{1})
    perturbed_node_nf_idx = findall(x -> x == perturbed_node, nf_nodes)
    deleteat!(nf_nodes, perturbed_node_nf_idx)

    V = get_voltage_mag(sol, syms, collect(1:n))
    V_pert = V[perturbed_node]
    V_wo_pert = deleteat!(V, perturbed_node)
    freq_wo_pert = get_frequency(sol, syms, nf_nodes, plot_angular_ω)

    CList = reshape(range(colorant"coral1", stop=colorant"steelblue", length=n), 1, n)
    
    # Decides frequency should be plotted in [Hz] or as the angular frequency in [rad/s] 
    if plot_angular_ω
        y_label_freq = L"\Delta ω [rad/s]"
        threshold_freq_plot = threshold_ω_out_of_bounds
    else
        y_label_freq = L"\Delta f [Hz]"
        threshold_freq_plot = threshold_ω_out_of_bounds /(2π)
    end
    # font sizes
    fsGuide = 13
    fsTick = 10
    
    # Plot Frequency transients
    Plots.plot([1, sol.t[end]], [threshold_freq_plot, threshold_freq_plot], linestyle=:dash, linecolor=colorant"gray", xlabelfontsize=fsGuide, ylabelfontsize=fsGuide, xtickfontsize=fsTick,ytickfontsize=fsTick,) # Plot upper bound frequency
    Plots.plot!([1, sol.t[end]], [-threshold_freq_plot, -threshold_freq_plot], linestyle=:dash, linecolor=colorant"gray", xlabelfontsize=fsGuide, ylabelfontsize=fsGuide, xtickfontsize=fsTick,ytickfontsize=fsTick,) # Plot lower bound frequency
    pl_freq = Plots.plot!(t, freq_wo_pert, legend=false, linecolor=CList, xaxis=:log, ylabel=y_label_freq, title=title_name, xlabelfontsize=fsGuide, ylabelfontsize=fsGuide, xtickfontsize=fsTick,ytickfontsize=fsTick,)
    if size(perturbed_node_nf_idx, 1) > 0
        f_pert = get_frequency(sol, syms, perturbed_node, plot_angular_ω)
        pl_freq = Plots.plot!(t, f_pert, legend=false, linecolor="black", xaxis=:log, ylabel=y_label_freq, xlabelfontsize=fsGuide, ylabelfontsize=fsGuide, xtickfontsize=fsTick,ytickfontsize=fsTick,)
    end
    if axis_lims
        ylims_ω = [-1.25 * threshold_freq_plot, 1.25 * threshold_freq_plot]
        pl_freq = Plots.plot!(ylims=ylims_ω, xlabelfontsize=fsGuide, ylabelfontsize=fsGuide)
    end
    Plots.plot!(grid=false)

    # Plot Voltage transients
    Plots.plot(t, high_voltage_ride_through, linecolor=colorant"gray", linestyle=:dash, xlabelfontsize=fsGuide, ylabelfontsize=fsGuide, xtickfontsize=fsTick,ytickfontsize=fsTick,) # Plot high voltage ride through curve
    Plots.plot!(t, low_voltage_ride_through, linecolor=colorant"gray", linestyle=:dash, xlabelfontsize=fsGuide, ylabelfontsize=fsGuide, xtickfontsize=fsTick,ytickfontsize=fsTick,) # Plot low voltage ride through curve
    Plots.plot!(t, V_wo_pert, legend=false, linecolor=CList, xaxis=:log, ylabel=L"V [p.u.]", xlabelfontsize=fsGuide, ylabelfontsize=fsGuide, xtickfontsize=fsTick,ytickfontsize=fsTick,) 
    Plots.plot!(t, V_pert, legend=false, linecolor="black", xaxis=:log, ylabel=L"V [p.u.]", xlabelfontsize=fsGuide, ylabelfontsize=fsGuide, xtickfontsize=fsTick,ytickfontsize=fsTick,)
    pl_v = Plots.plot!(xaxis=(L"t[s]"))
    Plots.plot!(grid=false)

    if axis_lims
        pl_v = Plots.plot!(ylims=[0.05, 1.16])
    end    
    if image_text != false
        # Plot simulation observables
        pl_text = Plots.plot(border=:none)
        x_pos = [.0, .5, .0, .5, .0, .0, .0, .0, .5, .0, .5, .0, .33, .66, .0, .66]
        y_pos = [1., 1., .875, .875, .75,  .6125, .5, .375, .375, .25, .25, .125, .125, .125, .0, 0]
        for (key,value) in image_text
            txt = (value, :left, 7)
            annotate!(pl_text, x_pos[key], y_pos[key], txt)
        end

        plt = Plots.plot(
            pl_freq, pl_v, pl_text;
            layout = Plots.grid(3, 1, heights=[0.4 ,0.4, 0.2]),
            size=(500, 750),
            margin=3Plots.mm,
            lw=3,  
        )
    else
        plt = Plots.plot(
            pl_freq, pl_v;
            layout=(2, 1),
            size=(500, 500),
            margin=3Plots.mm,
            lw=3,  
        )
    end
    return plt
end


"""
    function read_grid_and_state(id, grid_directory)

Reads grid and operation state of grid with `id` from `grid_directory`.
"""
function read_grid_and_state(id, grid_directory)
    pg = read_grid(id, grid_directory)
    file_state_name = string(grid_directory, "state_", id, ".json")
    state_vec = parsefile(file_state_name; dicttype=Dict, inttype=Int64)
    state_vec = convert(Array{Float64,1}, state_vec)
    pg_state = State(pg, state_vec)
    return pg, pg_state
end


"""
    function read_grid(id, grid_directory)
Read grid
"""
function read_grid(id, grid_directory)
    grid_name = string(grid_directory, "grid_", id, ".json")
    pg = read_powergrid(grid_name, Json)
    return pg
end


"""
    write_P0(P0,  file_name)

Stores `P0` 

- `P0`: P0
- `file_name`: File name
"""
function write_P0(P0, file_name)
    fid = h5open(file_name, "w")
    fid["P0"] = P0
    close(fid)
end


"""
    write_vertexpos(nodes_vertexpos,  file_name)

Stores `node_vertexpos` 

- `nodes_vertexpos`: Node coordinates
- `file_name`: File name
"""
function write_vertexpos(nodes_vertexpos, file_name)
    vertexpos_matrix = zeros(size(nodes_vertexpos, 1), 2)
    for i in 1:length(nodes_vertexpos)
        vertexpos_matrix[i, 1] = nodes_vertexpos[i][1]
        vertexpos_matrix[i, 2] = nodes_vertexpos[i][2]
    end
    fid = h5open(file_name, "w")
    fid["vertexpos"] = vertexpos_matrix
    close(fid)
end

"""
    read_vertexpos(file_name)
Reads positions of vertices of power grid
"""
function read_vertexpos(file_name)
    fid = h5open(file_name, "r")
    vertexpos = read(fid)["vertexpos"]
    close(fid)
    positions = Vector{Vector{Float64}}(undef, size(vertexpos, 1))
    for i in 1:size(vertexpos, 1)
        positions[i] = [vertexpos[i, 1], vertexpos[i, 2]]
    end
    return positions
end

"""
    restore_embedded_graph(id, grid_directory)

Load graph information and node coordinates to generate embedded graph.
"""
function restore_embedded_graph(id, grid_directory)
    pg = read_grid(id, grid_directory)
    positions = read_vertexpos(joinpath(grid_directory, string("grid_", id, "_vertexpos.json")))
    g = EmbeddedGraph(pg.graph, positions)
    return g
end

"""
    function store_dynamics(id, dynamics_directory, computational_effort, ds_result...)

Stores simulation results of grid with `id` in `dynamics_directory`.
"""
function store_dynamics(id, dynamics_directory, computational_effort, max_angular_frequency_dev, min_angular_frequency_dev, surv_vol_condition, infeasible, ω_excitability, voltage_excitability, P_diff_global, Q_diff_global, P_diff_local, Q_diff_local, mfd_final, final_diff_v, all_x_new, final_ω_excitability, final_voltage_excitability, survival_time, first_node_ω_out_of_bounds, first_node_voltage_out_of_bounds)
    file_name = string(dynamics_directory, "dynamics_", id, ".h5")
    fid = h5open(file_name, "w")
    fid["sim_time"] = computational_effort.time
    fid["bytes"] = computational_effort.bytes
    fid["max_angular_frequency_dev"] = max_angular_frequency_dev
    fid["min_angular_frequency_dev"] = min_angular_frequency_dev
    fid["surv_vol_condition"] = surv_vol_condition
    fid["infeasible"] = infeasible
    fid["ω_excitability"] = ω_excitability
    fid["voltage_excitability"] = voltage_excitability
    fid["P_diff_global"] = P_diff_global
    fid["Q_diff_global"] = Q_diff_global
    fid["P_diff_local"] = P_diff_local
    fid["Q_diff_local"] = Q_diff_local
    fid["mfd_final"] = mfd_final
    fid["final_diff_v"] = final_diff_v
    fid["final_ω_excitability"] = final_ω_excitability
    fid["final_voltage_excitability"] = final_voltage_excitability
    fid["survival_time"] = survival_time
    #fid["first_node_ω_out_of_bounds"] = first_node_ω_out_of_bounds
    #fid["first_node_voltage_out_of_bounds"] = first_node_voltage_out_of_bounds

    group_x_new = create_group(fid, "x_new")
    for (key, _) in all_x_new
        g = create_group(group_x_new, string(key))
        for (key2, val2) in all_x_new[key]
            g[string(key2)] = val2
        end
    end

    group_first_omega_oob = create_group(fid, "first_node_ω_out_of_bounds")
    for (key, _) in first_node_ω_out_of_bounds
        g = create_group(group_first_omega_oob, string(key))
        for (key2, val2) in first_node_ω_out_of_bounds[key]
            g[string(key2)] = val2
        end
    end 

    group_first_v_oob = create_group(fid, "first_node_voltage_out_of_bounds")
    for (key, _) in first_node_voltage_out_of_bounds
        g = create_group(group_first_v_oob, string(key))
        for (key2, val2) in first_node_voltage_out_of_bounds[key]
            g[string(key2)] = val2
        end
    end

    close(fid)
end