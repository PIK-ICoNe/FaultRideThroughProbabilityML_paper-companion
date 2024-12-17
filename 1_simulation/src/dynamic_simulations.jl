"""
    random_perturbation(afoprob, ambient_pert_val, idx)

Return a random perturbation using the given AmbientForcingProblem and perturbations vector.
"""
function random_perturbation(afoprob, ambient_pert_val, idx)
    op = afoprob.u0
    Frand = zeros(length(op))
    # For all nodes, perturb the voltage
    @views Frand[idx[1:2]] .= ambient_pert_val[1:2] .- op[idx[1:2]]
    # For inverters, perturb the frequency
    length(idx) == 3 && (Frand[idx[3]] = ambient_pert_val[3])
    z_new = ambient_forcing(afoprob, Frand)
    return z_new
end

"""
    pd_node_idx(pg::PowerGrid, node::Int, method::String)

Returns index vector of variables at node. Can be used in Random Force
"""
function pd_node_idx(pg::PowerGrid, rpg, node_num::Int, method::String)
    if method == "Voltage"
        str_vec = ["u_r_" * string(node_num), "u_i_" * string(node_num)]
    elseif method == "All"
        if typeof(pg.nodes) == OrderedDict{String,Any}
            node = collect(keys(pg.nodes))[node_num] # we have to give the key to find the node in the dict, but the key can not be used for the numbering
        else
            node = node_num # in Arrays it does not matter
        end

        if length(symbolsof(pg.nodes[node])) == 2
            str_vec = ["u_r_" * string(node_num), "u_i_" * string(node_num)]
        elseif :x_1 ∈ symbolsof(pg.nodes[node])
            if :θ ∈ symbolsof(pg.nodes[node])
                str_vec = [
                    "u_r_" * string(node_num),
                    "u_i_" * string(node_num),
                    "x_1_" * string(node_num),
                    "θ_" * string(node_num),
                ]
            else
                str_vec = [
                    "u_r_" * string(node_num),
                    "u_i_" * string(node_num),
                    "x_1_" * string(node_num),
                ]
            end
        else
            error("Please construct the vector manually")
        end
    else
        error("Please use a valid method.")
    end
    return idx_exclusive(rpg, str_vec)
end

"""
    struct simulation_properties
    
    Container for the properties of the simulation
"""
struct simulation_properties
    end_simulation_time::Int
    num_grids::Int
    pert_per_node::Int
    method::String
    seed_summand::Int
    ω_pert_size_low::Float32
    ω_pert_size_high::Float32
    u_pert_size::Float32
    threshold_ω_out_of_bounds::Float32
end

"""
    find_first_ω_out_of_bounds(x, t, ω_threshold, ω_idx, ω_idxs_map)

Checks if all node frequencies are below the threshold `ω_threshold`. 
If a nodes frequency is above the limit, it is checked when the threshold is exceeded for the first time. Then, the indices of all nodes that are above the limit at that timestep, and the timestep, are returned. 
If no limit is violated an empty `Vector{Int64}` will be returned.
`x` solution object of ODE solve
`t` time of solution object of ODE solve
`ω_threshold` threshold for ω
`ω_idx` are the indices of the frequency variables in the solution object
`ω_idxs_map` maps from `ω_idx` to the global node indices.
"""
function find_first_ω_out_of_bounds(x, t, ω_threshold, ω_idx, ω_idxs_map)
    @assert size(x, 2) == length(t)
    ω_abs = abs.(x[ω_idx,:])
    out_of_bounds = ω_abs .> ω_threshold
    return first_out_of_bounds(out_of_bounds, t, ω_idxs_map)
end

"""
    find_first_voltage_out_of_bounds(x, t, ur_idx, ui_idx, v_idxs_map)

Checks if all node voltages are above or below the limiting curves for a fault ride through for under- and over-voltages respectively. If a nodes voltage is out of bounds, it is checked when any threshold is exceeded for the first time. Then, the indices of all nodes that are below the limit at that timestep, and the timestep, are returned.
If no limit is violated an empty `Vector{Int64}` will be returned.
`x` solution object of ODE solve
`t` time of solution object of ODE solve
`ur_idx` are the indices of the real part of the complex voltage variables in the solution object
`ui_idx` are the indices of the complex part of the voltage
`v_idxs_map[ur_idx]` maps from `ur_idx` (and `ui_idx`) to the global node indices.
"""
function find_first_voltage_out_of_bounds(x, t, ur_idx, ui_idx, v_idxs_map)
    @assert size(x, 2) == length(t)
    v = abs.(x[ur_idx, :] .+ im .* x[ui_idx, :])
    # transpose() converts to matrix for row-wise comparison
    low_voltage_ride_through_curve = transpose(low_voltage_ride_through(t))
    high_voltage_ride_through_curve = transpose(high_voltage_ride_through(t))
    out_of_bounds = .!(low_voltage_ride_through_curve .< v .< high_voltage_ride_through_curve)
    return first_out_of_bounds(out_of_bounds, t, v_idxs_map)
end

"""
    first_out_of_bounds(out_of_bounds, t, idxs_map)

Small convenience function that returns the first nodes that are out of bounds and the time that the bounds were exceeded for the first time.
"""
function first_out_of_bounds(out_of_bounds, t, idxs_map)
    if any(out_of_bounds)
        # Compute first column with an element out of bounds
        first_col_oob = findfirst(out_of_bounds)[2]
        first_nodes_oob = findall(out_of_bounds[:, first_col_oob])
        return idxs_map[first_nodes_oob], t[first_col_oob]
    else
        return Int64[], t[end]
    end
end

"""
    get_initialized_return_values(N, pert_per_node)

Generates the initial empty vectors and matrices for the return values.

`N` Number of nodes
`pert_per_node` Number of perturbations per node.
"""
function get_initialized_return_values(N, pert_per_node)
    # init return variables
    max_angular_frequency_dev = zeros(Float32, N, pert_per_node)
    min_angular_frequency_dev = zeros(Float32, N, pert_per_node)

    surv_vol_condition = ones(N, pert_per_node)
    infeasible = zeros(Int8, N, pert_per_node)

    survival_time = zeros(Float32, N, pert_per_node)
    # excitability of nodes, i.e. bounds are violated at node i after perturbation at j
    ω_excitability = zeros(Int, N)
    voltage_excitability = zeros(Int, N)

    final_ω_excitability = zeros(Int, N)
    final_voltage_excitability = zeros(Int, N)

    # store energy per perturbation
    P_diff_global = zeros(Float32, N, pert_per_node)
    Q_diff_global = zeros(Float32, N, pert_per_node)
    P_diff_local = zeros(Float32, N, pert_per_node)
    Q_diff_local = zeros(Float32, N, pert_per_node)

    # final voltage
    final_diff_v = zeros(Float32, N, pert_per_node)
    # store mfd_final
    mfd_final = zeros(Float32, N, pert_per_node)

    first_node_ω_out_of_bounds = Dict() #zeros(Int32, N, pert_per_node)
    first_node_voltage_out_of_bounds = Dict() #zeros(Int32, N, pert_per_node)

    all_x_new = Dict()

    return max_angular_frequency_dev, min_angular_frequency_dev, surv_vol_condition, infeasible, ω_excitability, voltage_excitability, P_diff_global, Q_diff_global, P_diff_local, Q_diff_local, final_diff_v, mfd_final, final_ω_excitability, final_voltage_excitability, survival_time, first_node_ω_out_of_bounds, first_node_voltage_out_of_bounds, all_x_new
end

"""
    get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, pert_per_node)

Generates the sobol samples and transforms them to the real and imaginary parts for the voltage.

`ω_pert_size_low` Lower bound for the frequency perturbations in [rad/s]
`ω_pert_size_high` Upper bound for the frequency perturbations in [rad/s]
`pert_per_node` Number of perturbations = number of sobol samples
"""
function get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, pert_per_node; v_size_low=0.0, skipbool=true)
    _s = SobolSeq([v_size_low, -1.0, ω_pert_size_low], [1.0, 1.0, ω_pert_size_high])
    # The Sobol.jl manual suggests to skip the first part of the sequence to achieve better uniformity
    if skipbool
        s = skip(_s, pert_per_node)
    else
        s = _s
    end
    soboli_values = next!(s)
    for _ in 2:pert_per_node
        x = next!(s)
        soboli_values = hcat(soboli_values, x)
    end
    sob_ur_values = sqrt.(soboli_values[1, :]) .* cos.(pi * soboli_values[2, :])
    sob_ui_values = sqrt.(soboli_values[1, :]) .* sin.(pi * soboli_values[2, :])
    sob_ω_values = soboli_values[3, :]
    soboli_perturbations = hcat(sob_ur_values, sob_ui_values, sob_ω_values)

    return soboli_perturbations
end

"""
    get_symbolic_problem(rpg, op, end_simulation_time)

Generates a symbolic version of the ODE Problem that uses the symbolic jacobian and sparsity pattern. Results in a significant speed increase when running multiple simulations.

`rpg` Right hand side function of the powergrid
`op` operationpoint
`end_simulation_time` End time for the dynamic simulation in [s]
"""
function get_symbolic_problem(rpg, op, end_simulation_time)
    ### MTK stuff
    prob = ODEProblem(rpg, op.vec, (0.0, end_simulation_time)) # start at operation point
    sys = modelingtoolkitize(prob)
    prob_sparsejac = ODEProblem(sys, [], prob.tspan, jac=true, sparse=true)

    return prob_sparsejac
end

"""
    get_indices_maps(pg, rpg, method)

Returns a map for the variable indices of the nodes in the power grid. Also includes the maps for the real and imaginary part of the voltage and the frequency variable.

`pg` PowerGrid object from PowerDynamics.jl
`rpg` Right hand side function of the powergrid
`method` Decides if all variables ("All") should be perturbed or only the voltage ("Voltage")
"""
function get_indices_maps(pg, rpg, method)
    # Generate maps from variables indices to node indices
    indices = map(n -> pd_node_idx(pg, rpg, n, method), 1:length(pg.nodes))
    ω_idx = findall(map(variable -> occursin("x_1", variable), string.(rpg.syms))) # indexes of ω variables, needs to work for Arrays and Dicts
    ur_idx = findall(map(variable -> occursin("u_r", variable), string.(rpg.syms)))
    ui_idx = findall(map(variable -> occursin("u_i", variable), string.(rpg.syms)))

    # Find symbols of frequencies (ω), convert to String, discard the "ω_" part, convert to Int
    # Caveat: For some reason (Unicode??), the string's indices are [1,3,4,...]
    ω_idxs_map = map(x -> parse(Int, x[5:end]), String.(rpg.syms[ω_idx]))
    v_idxs_map = map(x -> parse(Int, x[5:end]), String.(rpg.syms[ur_idx]))
    
    return indices, ω_idx, ur_idx, ui_idx, ω_idxs_map, v_idxs_map
end

"""
    get_slack_indices_maps(pg, N)

Generates a list of nodes without the slack bus, as it cannot be perturbed and thus should not be simulated. Returns a map that gives the position in the vector also containing the slack.

`pg` PowerGrid object from PowerDynamics.jl
`N` Number of nodes
""" 
function get_slack_indices_maps(pg, N)
    slack_idx = findfirst(typeof.(pg.nodes) .== SlackAlgebraic)
    nodes = collect(1:N)
    nodes_wo_slack = deleteat!(nodes, findall(x -> x == slack_idx[1], nodes))
    map_idx_back = map(map_idx_back -> map_idx_back > slack_idx ? map_idx_back - 1 : map_idx_back, collect(1:N))
    map_idx_back[slack_idx] = 0

    return nodes_wo_slack, map_idx_back
end


"""
    analyze_sol(node, i, sol, sim_prop, ω_idx, ω_idxs_map, ur_idx, ui_idx, v_idxs_map)

Analyzes the sol object to compute storable values
- `node` node index
- `i` perturbation index
- `sol` sol object
- `sim_prop` simulation_properties
- `ω_idx` are the indices of the frequency variables in the solution object
- `ω_idxs_map` maps from `ω_idx` to the global node indices.
- `ur_idx` are the indices of the real part of the complex voltage variables in the solution object
- `ui_idx` are the indices of the complex part of the voltage
- `v_idxs_map[ur_idx]` maps from `ur_idx` (and `ui_idx`) to the global node indices.
"""
function analyze_sol(node, i, sol, sim_prop, ω_idx, ω_idxs_map, ur_idx, ui_idx, v_idxs_map)
    if sol.retcode != :Success && sol.retcode != :Terminated
        println("###### retcode is not success")
        println("retcode: ", sol.retcode)
        println("node: ", node)
        println("perturbation idx: ", i)
    end

    if sol.retcode == :Success
        infeasible = 0
    elseif sol.retcode == :Unstable
        infeasible = 1
    elseif sol.retcode == :DtLessThanMin
        infeasible = 2
    elseif sol.retcode == :MaxIters
        infeasible = 3
    elseif sol.retcode == :Terminated
        infeasible = 4
    else
        infeasible = 5
    end

    first_nodes_ω_out_of_bounds, t_ω_out_of_bounds = find_first_ω_out_of_bounds(
        sol,
        sol.t,
        sim_prop.threshold_ω_out_of_bounds,
        ω_idx,
        ω_idxs_map,
    )

    first_nodes_v_out_of_bounds, t_v_out_of_bounds = find_first_voltage_out_of_bounds(
        sol,
        sol.t,
        ur_idx,
        ui_idx,
        v_idxs_map,
    )


    ## final states
    final_nodes_ω_out_of_bounds, _ = find_first_ω_out_of_bounds(
        sol[end],
        sol.t[end],
        sim_prop.threshold_ω_out_of_bounds,
        ω_idx,
        ω_idxs_map,
    )

    final_nodes_v_out_of_bounds, _ = find_first_voltage_out_of_bounds(
        sol[end],
        sol.t[end],
        ur_idx,
        ui_idx,
        v_idxs_map,
    )

    survival_time = min(t_ω_out_of_bounds, t_v_out_of_bounds, sol.t[end])
    return infeasible, first_nodes_ω_out_of_bounds, first_nodes_v_out_of_bounds, survival_time, final_nodes_ω_out_of_bounds, final_nodes_v_out_of_bounds
end

"""
    dynamic_simulation(r, N, pg, pg_state, sim_prop, plot_probability, pics_directory)

Simulates `sim_prob.pert_per_node` perturbations per `N` nodes in the powergrid `pg` with operationpoint `pg_state`.
Further arguments are additional simulation properties such as a seed for the RNG.

- `r`: grid index
- `N`: number of nodes
- `pg`: power grid
- `pg_state`: state of power grid
- `sim_prop`: simulation properties
- `plot_probability`: probability to plot trajectories
- `pics_directory`: path to save plots
"""
function dynamic_simulation(r, N, pg, pg_state, sim_prop, plot_probability, pics_directory)
    pert_per_node = sim_prop.pert_per_node
    max_angular_frequency_dev, min_angular_frequency_dev, surv_vol_condition, infeasible, ω_excitability, voltage_excitability, P_diff_global, Q_diff_global, P_diff_local, Q_diff_local, final_diff_v, mfd_final, final_ω_excitability, final_voltage_excitability, survival_time, first_node_ω_out_of_bounds, first_node_voltage_out_of_bounds, all_x_new = get_initialized_return_values(N, pert_per_node)
    
    rpg = rhs(pg)
    op = pg_state

    indices, ω_idx, ur_idx, ui_idx, ω_idxs_map, v_idxs_map = get_indices_maps(pg, rpg, sim_prop.method)
    
    soboli_perturbations = get_soboli_perturbations(sim_prop.ω_pert_size_low, sim_prop.ω_pert_size_high, pert_per_node)
    prob_sparsejac = get_symbolic_problem(rpg, op, sim_prop.end_simulation_time)
    afoprob = ambient_forcing_problem(prob_sparsejac.f, op.vec, 1.0, zeros(length(op.vec))) # τ = 1.0

    nodes_wo_slack, map_idx_back = get_slack_indices_maps(pg, N)

    P_set = map(x -> getfield(pg.nodes[x], :P), nodes_wo_slack)
    Q_set = map(x -> getfield(pg.nodes[x], :Q), nodes_wo_slack)
    
    for node in nodes_wo_slack
        all_x_new[node] = Dict()
        first_node_voltage_out_of_bounds[node] = Dict()
        first_node_ω_out_of_bounds[node] = Dict()
        idx = indices[node]
        for i = 1:pert_per_node
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
                first_node_voltage_out_of_bounds[node][i] = first_nodes_v_out_of_bounds
            end

            if !isempty(first_nodes_ω_out_of_bounds)
                first_node_ω_out_of_bounds[node][i] = first_nodes_ω_out_of_bounds
            end

            # saving final states
            final_state = sol.u[end]
            mfd_final[node, i] = maximum(abs.(final_state[ω_idx]))
            v_final = abs.(final_state[ur_idx] .+ 1im .* final_state[ui_idx])
            final_diff_v[node, i] = maximum(abs.((op[:, :v] .- v_final) ./ op[:, :v]))

            final_ω_excitability[final_nodes_ω_out_of_bounds] .+= 1
            final_voltage_excitability[final_nodes_v_out_of_bounds] .+= 1

            # plotting
            if rand(1)[1] < plot_probability
                image_text = prepare_text_for_plot(max_angular_frequency_dev[node, i], min_angular_frequency_dev[node, i], surv_vol_condition[node, i], first_nodes_ω_out_of_bounds, first_nodes_v_out_of_bounds, final_nodes_ω_out_of_bounds, final_nodes_v_out_of_bounds, P_diff_global[node, i], Q_diff_global[node, i], P_diff_local[node, i], Q_diff_local[node, i], mfd_final[node, i], final_diff_v[node, i], infeasible[node, i], all_x_new[node][i], survival_time[node, i])
                title_name = string("grid: ", r, " node: ", node, " pert: ", i)
                plot_res_log(sol,  pg, node, title_name, image_text, sim_prop.threshold_ω_out_of_bounds, low_voltage_ride_through(sol.t), high_voltage_ride_through(sol.t); plot_angular_ω = true, axis_lims = false)
                filename = joinpath(pics_directory, string("grid_", r, "_node_", node, "_idx_", i, ".png"))
                Plots.savefig(filename)
            end
            GC.gc() # helps with the memory errors on slurm
        end
    end
    return max_angular_frequency_dev,
    min_angular_frequency_dev,
    surv_vol_condition,
    infeasible,
    ω_excitability,
    voltage_excitability,
    P_diff_global,
    Q_diff_global,
    P_diff_local,
    Q_diff_local,
    mfd_final,
    final_diff_v,
    all_x_new,
    final_ω_excitability,
    final_voltage_excitability,
    survival_time,
    first_node_ω_out_of_bounds,
    first_node_voltage_out_of_bounds
end

export dynamic_simulation, simulation_properties
