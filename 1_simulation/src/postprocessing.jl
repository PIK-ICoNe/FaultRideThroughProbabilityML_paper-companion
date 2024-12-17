struct all_thresholds
    threshold_ω_surv_low
    threshold_ω_surv_high
    threshold_ω_snbs
    threshold_u_snbs
end

function make_node_line_df(id, grid_dir, grid_type)
    pg, pg_state = read_grid_and_state(id, grid_dir)
    power_flow_P_dict, power_flow_Q_dict = get_power_flow_on_lines(pg_state)

    # node properties
    node_types = [string(typeof(node)) for node in pg.nodes]
    Bₓ_real = [
        try
            node.Bₓ[1]
        catch
            NaN
        end for node in pg.nodes
    ]
    P = [
        try
            node.P
        catch
            NaN
        end for node in pg.nodes
    ]
    Q = [
        try
            node.Q
        catch
            NaN
        end for node in pg.nodes
    ]
    num_nodes = length(pg.nodes)

    # line properties
    line_types = [string(typeof(line)) for line in pg.lines]
    from_node = [line.from for line in pg.lines]
    to_node = [line.to for line in pg.lines]
    power_flow_P_ij = [get(power_flow_P_dict, [line.from, line.to], missing) for line in pg.lines]
    power_flow_P_ji = [get(power_flow_P_dict, [line.to, line.from], missing) for line in pg.lines]
        
    power_flow_Q_ij = [get(power_flow_Q_dict, [line.from, line.to], missing) for line in pg.lines]
    power_flow_Q_ji = [get(power_flow_Q_dict, [line.to, line.from], missing) for line in pg.lines]
    y = [line.y for line in pg.lines]
    # y_shunt_km = [line.y_shunt_km for line in pg.lines]
    y_shunt_mk = [line.y_shunt_mk for line in pg.lines]

    # weights, flows, diffs
    weights_ij, flows_ij, diffs_ij, weights_ji, flows_ji, diffs_ji = compute_weights_flows_diffs(pg, pg_state)

    df_nodes = DataFrame(
        grid_type=grid_type,
        grid_index=id,
        node_id=collect(1:num_nodes),
        node_type=node_types,
        num_nodes=num_nodes,
        grid_dir=grid_dir,
        B_x_real=Bₓ_real,
        P=P,
        Q=Q,
    )
    df_lines = DataFrame(
        grid_type=grid_type,
        grid_index=id,
        line_type=line_types,
        source=from_node,
        destination=to_node,
        power_flow_P_ij=power_flow_P_ij,
        power_flow_P_ji=power_flow_P_ji,
        power_flow_Q_ij=power_flow_Q_ij,
        power_flow_Q_ji=power_flow_Q_ji,
        y_real=real.(y),
        y_imag=imag.(y),
        # y_shunt_km_real = real.(y_shunt_km),
        # y_shunt_km_imag = imag.(y_shunt_km),
        # y_shunt_mk_real = real.(y_shunt_mk),
        y_shunt_mk_imag=imag.(y_shunt_mk),
        flows_ij = flows_ij,
        flows_ji = flows_ji,
        diffs_ij = diffs_ij,
        diffs_ji = diffs_ji,
        weights_ij = weights_ij,
        weights_ji = weights_ji,
    )
    # compute the sums of edge features to use them as node features
    df_nodes.sum_y_real = [sum(df_lines[df_lines.source .== node_id, :y_real]) + sum(df_lines[df_lines.destination .== node_id, :y_real]) for node_id in df_nodes.node_id]
    df_nodes.sum_y_imag = [sum(df_lines[df_lines.source .== node_id, :y_imag]) + sum(df_lines[df_lines.destination .== node_id, :y_imag]) for node_id in df_nodes.node_id]
    df_nodes.sum_y_shunt_mk_imag = [sum(df_lines[df_lines.source .== node_id, :y_shunt_mk_imag]) + sum(df_lines[df_lines.destination .== node_id, :y_shunt_mk_imag]) for node_id in df_nodes.node_id]
    return df_nodes, df_lines
end


function compute_targets(results, thresholds_targets, consider_infeasible)

    threshold_ω_surv_low = thresholds_targets.threshold_ω_surv_low
    threshold_ω_surv_high = thresholds_targets.threshold_ω_surv_high
    threshold_ω_snbs = thresholds_targets.threshold_ω_snbs
    threshold_u_snbs = thresholds_targets.threshold_u_snbs

    infeasible = results["infeasible"] .> 0
    max_angular_frequency_dev = results["max_angular_frequency_dev"]
    min_angular_frequency_dev = results["min_angular_frequency_dev"]
    surv_vol_condition = results["surv_vol_condition"]
    ω_excitability = results["ω_excitability"]
    voltage_excitability = results["voltage_excitability"]
    mfd_final = results["mfd_final"]
    final_diff_v = results["final_diff_v"]

    surv_ω_bit_matrix = (min_angular_frequency_dev .> threshold_ω_surv_low) .&& (max_angular_frequency_dev .< threshold_ω_surv_high)
    surv_ω_bit_matrix .= consider_infeasible ? surv_ω_bit_matrix .* .!infeasible : surv_ω_bit_matrix
    surv_ω = vec(sum(surv_ω_bit_matrix, dims=2) ./ size(surv_ω_bit_matrix, 2))

    surv_u_bit_matrix = surv_vol_condition
    surv_u_bit_matrix .= consider_infeasible ? surv_u_bit_matrix .* .!infeasible : surv_u_bit_matrix
    surv_u = vec(sum(surv_u_bit_matrix, dims=2) ./ size(surv_u_bit_matrix, 2))

    surv_bit_matrix = surv_ω_bit_matrix .&& surv_u_bit_matrix
    surv = vec(sum(surv_bit_matrix, dims=2) ./ size(surv_bit_matrix, 2))

    snbs_ω_bit_matrix = mfd_final .< threshold_ω_snbs
    snbs_ω_bit_matrix .= consider_infeasible ? snbs_ω_bit_matrix .* .!infeasible : snbs_ω_bit_matrix
    snbs_ω = vec(sum(snbs_ω_bit_matrix, dims=2) ./ size(snbs_ω_bit_matrix, 2))

    snbs_u_bit_matrix = final_diff_v .< threshold_u_snbs
    snbs_u_bit_matrix .= consider_infeasible ? snbs_u_bit_matrix .* .!infeasible : snbs_u_bit_matrix
    snbs_u = vec(sum(snbs_u_bit_matrix, dims=2) ./ size(snbs_u_bit_matrix, 2))

    snbs_bit_matrix = snbs_ω_bit_matrix .&& snbs_u_bit_matrix
    snbs = vec(sum(snbs_bit_matrix, dims=2) ./ size(snbs_u_bit_matrix, 2))

    num_nodes, pert_per_node = pert_per_node = size(infeasible)
    excitability_ω = ω_excitability / (pert_per_node * num_nodes)
    excitability_u = voltage_excitability / (pert_per_node * num_nodes)
    return surv_ω, surv_u, surv, snbs_ω, snbs_u, snbs, excitability_ω, excitability_u
end

function collect_grid_sim_data!(df_all_nodes, df_all_lines, id, data_dir, grid_type, thresholds_targets, consider_infeasible)
    grid_dir = joinpath(data_dir, "grids/")
    df_nodes, df_lines = make_node_line_df(id, grid_dir, grid_type)
    dynamics_file_name = joinpath(data_dir, "dynamics", "dynamics_$id.h5")
    fid = h5open(dynamics_file_name, "r")
    results = read(fid)
    surv_ω, surv_u, surv, snbs_ω, snbs_u, snbs, excitability_ω, excitability_u = compute_targets(results, thresholds_targets, consider_infeasible)
    close(fid)
    df_nodes.surv_omega = surv_ω
    df_nodes.surv_u = surv_u
    df_nodes.surv = surv
    df_nodes.snbs_omega = snbs_ω
    df_nodes.snbs_u = snbs_u
    df_nodes.snbs = snbs
    df_nodes.excitability_omega = excitability_ω
    df_nodes.excitability_u = excitability_u
    append!(df_all_nodes, df_nodes)
    append!(df_all_lines, df_lines)
end

function compute_weights_flows_diffs(pg, pg_state)
    flows_ij = Float64[]
    weights_ij = Float64[]
    diffs_ij = Float64[]
    flows_ji = Float64[]
    weights_ji = Float64[]
    diffs_ji = Float64[]

    for line in pg.lines
        v_i = pg_state[line.to, :u]
        v_j = pg_state[line.from, :u]
        y_ij = line.y

        # For direction i -> j
        s_ij = v_i * conj(y_ij * (v_i - v_j))
        phi_ij = angle(v_i * conj(v_j))
        push!(flows_ij, real(s_ij) / abs(y_ij))
        push!(diffs_ij, phi_ij)
        push!(weights_ij, cos(phi_ij) * abs(y_ij))

        # For direction j -> i
        s_ji = v_j * conj(y_ij * (v_j - v_i))
        phi_ji = angle(v_j * conj(v_i))
        push!(flows_ji, real(s_ji) / abs(y_ij))
        push!(diffs_ji, phi_ji)
        push!(weights_ji, cos(phi_ji) * abs(y_ij))
    end

    return weights_ij, flows_ij, diffs_ij, weights_ji, flows_ji, diffs_ji
end