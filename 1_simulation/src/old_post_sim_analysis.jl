using DataFrames
using Combinatorics

mutable struct statistics_per_variable
    min::Float32
    max::Float32
    mean::Float32
    std::Float32
end


begin
    function get_nodal_properties_df(input_nodes)
        num_nodes = length(input_nodes)
        P = zeros(num_nodes)
        Q = zeros(num_nodes)
        Bᵤ, Cᵤ, Gᵤ, Hᵤ, Bₓ, Cₓ, Gₓ, Hₓ, Y_n = (zeros(ComplexF64, num_nodes) for _ = 1:9)
        node_types = String[]
        node_types_int = Int[]
        node_type_load = Int[]
        node_type_slack = Int[]
        for i in 1:num_nodes
            if typeof(input_nodes[i]) == SlackAlgebraic
                push!(node_types, "SlackAlgebraic")
                push!(node_types_int, 4)
                push!(node_type_slack, 1)
                push!(node_type_load, 0)
            else
                push!(node_type_slack, 0)
                P[i] = input_nodes[i].P
                Q[i] = input_nodes[i].Q
                if typeof(input_nodes[i]) == NormalForm{1}
                    push!(node_types, "normalForm")
                    bx_real = input_nodes[i].Bₓ[1]
                    if bx_real == -0.2
                        push!(node_types_int, 1)
                    elseif bx_real == -1.0
                        push!(node_types_int, 2)
                    elseif bx_real == -2.0
                        push!(node_types_int, 3)
                    end
                    Bᵤ[i] = input_nodes[i].Bᵤ[1]
                    Cᵤ[i] = input_nodes[i].Cᵤ[1]
                    Gᵤ[i] = input_nodes[i].Gᵤ[1]
                    Hᵤ[i] = input_nodes[i].Hᵤ[1]
                    Bₓ[i] = input_nodes[i].Bₓ[1]
                    Cₓ[i] = input_nodes[i].Cₓ[1]
                    Gₓ[i] = input_nodes[i].Gₓ[1]
                    Hₓ[i] = input_nodes[i].Hₓ[1]
                    Y_n[i] = input_nodes[i].Y_n[1]
                    push!(node_type_load, 0)
                else
                    push!(node_types, "load")
                    push!(node_types_int, 0)
                    push!(node_type_load, 1)
                end
            end
        end

        df = DataFrame()
        df.P = P
        df.Q = Q
        df.Bᵤ_real = real(Bᵤ)
        df.Bᵤ_imag = imag(Bᵤ)
        df.Cᵤ_real = real(Cᵤ)
        df.Cᵤ_imag = imag(Cᵤ)
        df.Gᵤ_real = real(Gᵤ)
        df.Gᵤ_imag = imag(Gᵤ)
        df.Hᵤ_real = real(Hᵤ)
        df.Hᵤ_imag = imag(Hᵤ)
        df.Bₓ_real = real(Bₓ)
        df.Bₓ_imag = imag(Bₓ)
        df.Cₓ_real = real(Cₓ)
        df.Cₓ_imag = imag(Cₓ)
        df.Gₓ_real = real(Gₓ)
        df.Gₓ_imag = imag(Gₓ)
        df.Hₓ_real = real(Hₓ)
        df.Hₓ_imag = imag(Hₓ)
        df.Y_n_real = real(Y_n)
        df.Y_n_imag = imag(Y_n)

        df.node_type = node_types
        df.node_type_int = node_types_int
        df.node_type_load = node_type_load
        df.node_type_slack = node_type_slack
        return df
    end

    function get_idx_of_nodes(all_node_types)
        idx_loads = findall(all_node_types .== 0)
        idx_normalform1 = findall(all_node_types .== 1)
        idx_normalform2 = findall(all_node_types .== 2)
        idx_normalform3 = findall(all_node_types .== 3)
        idx_slack = findall(all_node_types .== 4)
        return idx_loads, idx_normalform1, idx_normalform2, idx_normalform3, idx_slack
    end

    function combine_data_one_df(data)
        keys_data = collect(keys(data))
        df = data[keys_data[1]]
        for i in 2:length(data)
            df = vcat(df, data[keys_data[i]])
        end
        return df
    end

    function read_grids_data(data_dir, start_index, end_index)
        grid_data_nodes = Dict()
        grid_data_lines = Dict()
        for grid_index in start_index:end_index
            id = lpad(grid_index, length(digits(10000)), '0')
            grid_directory = joinpath(data_dir, "grids/")
            pg, op = read_grid_and_state(id, grid_directory)
            power_flow_P, power_flow_Q = get_power_flow_on_lines(op)
            df_nodes = get_nodal_properties_df(pg.nodes)
            grid_data_nodes[grid_index] = df_nodes
            df_edges = get_line_properties_df(pg.lines)
            grid_data_lines[grid_index] = df_edges
            add_power_flow!(df_edges, power_flow_P, power_flow_Q)
            add_nodewise_sum_shunts_admittances!(df_nodes, df_edges)
        end
        return grid_data_nodes, grid_data_lines
    end

    function get_min_max_variable(df, column_name, indices=false)
        if indices != false
            min_value = minimum(df[!, column_name][indices])
            max_value = maximum(df[!, column_name][indices])
        else
            min_value = minimum(df[!, column_name])
            max_value = maximum(df[!, column_name])
        end
        return [min_value, max_value]
    end

    function get_min_max_df(df_input, column_names, indices=false)
        df = DataFrame()
        for i in 1:size(column_names, 1)
            column_name = column_names[i]
            if indices != false
                df[!, column_name] = get_min_max_variable(df_input, column_name, indices[column_name])
            else
                df[!, column_name] = get_min_max_variable(df_input, column_name)
            end
        end
        return df
    end

    function get_statistics_features_df(df_input, column_names, indices=false)
        # df = DataFrame()
        stats = Dict()
        for i in 1:size(column_names, 1)
            column_name = column_names[i]
            if indices != false
                stats[column_name] = get_min_max_mean_std_variable(df_input, column_name, indices[column_name])
            else
                stats[column_name] = get_min_max_mean_std_variable(df_input, column_name)
            end
        end
        return stats
    end

    function get_min_max_mean_std_variable(df, column_name, indices=false)
        if indices == false
            indices = 1:size(df[!, column_name], 1)
        end
        min_value = minimum(df[!, column_name][indices])
        max_value = maximum(df[!, column_name][indices])
        mean_value = mean(df[!, column_name][indices])
        std_value = std(df[!, column_name][indices])
        return statistics_per_variable(min_value, max_value, mean_value, std_value)
    end

    function get_line_properties_df(input_edges)
        num_edges = length(input_edges)
        from, to = (zeros(Int32, num_edges) for _ in 1:2)
        y, y_shunt_km, y_shunt_mk = (zeros(ComplexF64, num_edges) for _ = 1:3)
        for i in 1:num_edges
            if typeof(input_edges[i]) == PiModelLine
                from[i] = input_edges[i].from
                to[i] = input_edges[i].to
                y[i] = input_edges[i].y
                y_shunt_km[i] = input_edges[i].y_shunt_km
                y_shunt_mk[i] = input_edges[i].y_shunt_mk
            end
        end
        df = DataFrame()
        df.from = from
        df.to = to
        df.y_real = real(y)
        df.y_imag = imag(y)
        df.y_shunt_mk_real = real(y_shunt_mk)
        df.y_shunt_mk_imag = imag(y_shunt_mk)
        df.y_shunt_km_real = real(y_shunt_km)
        df.y_shunt_km_imag = imag(y_shunt_km)
        return df
    end

    function normalize_array(value, stats_variable)
        if (stats_variable.min < 0) & (stats_variable.max > 0)
            a = -1.0
            b = 1.0
        elseif (stats_variable.min <= 0) & (stats_variable.max <= 0)
            a = 1.0
            b = 0.0
        elseif (stats_variable.min >= 0) & (stats_variable.max >= 0)
            a = 0.0
            b = 1.0
        end
        normalized = (b - a) * (value .- stats_variable.min) / (stats_variable.max - stats_variable.min) .+ a
        return normalized
    end

    function standardize_array(value, stats_variable)
        return (value .- stats_variable.mean) / stats_variable.std
    end

    ## structure with dynamical results
    struct sim_results
        max_angular_frequency_dev
        min_angular_frequency_dev
        surv_vol_condition
        infeasible
        ω_excitability
        voltage_excitability
        P_diff_global
        Q_diff_global
        P_diff_local
        Q_diff_local
        mfd_final
        final_diff_v
        all_x_new
        final_ω_excitability
        final_voltage_excitability
        survival_time
        first_node_ω_out_of_bounds
        first_node_voltage_out_of_bounds
        sim_time
        bytes
    end

    struct all_thresholds
        threshold_ω_surv_low
        threshold_ω_surv_high
        threshold_ω_snbs
        threshold_u_snbs
    end

    function read_simulation_results_for_one_grid(data_dir, grid_index)
        id = lpad(grid_index, length(digits(10000)), '0')
        file_name = string("dynamics/dynamics_", id, ".h5")
        sim_file_name = joinpath(data_dir, file_name)
        fid = h5open(sim_file_name, "r")
        results = read(fid)
        max_angular_frequency_dev = results["max_angular_frequency_dev"]
        min_angular_frequency_dev = results["min_angular_frequency_dev"]
        surv_vol_condition = results["surv_vol_condition"]
        infeasible = results["infeasible"]
        ω_excitability = results["ω_excitability"]
        voltage_excitability = results["voltage_excitability"]
        P_diff_global = results["P_diff_global"]
        Q_diff_global = results["Q_diff_global"]
        P_diff_local = results["P_diff_local"]
        Q_diff_local = results["Q_diff_local"]
        mfd_final = results["mfd_final"]
        final_diff_v = results["final_diff_v"]
        x_new = results["x_new"]
        final_ω_excitability = results["final_ω_excitability"]
        final_voltage_excitability = results["final_voltage_excitability"]
        survival_time = results["survival_time"]
        first_node_ω_out_of_bounds = results["first_node_ω_out_of_bounds"]
        first_node_voltage_out_of_bounds = results["first_node_voltage_out_of_bounds"]
        sim_time = results["sim_time"]
        bytes = results["bytes"]
        close(fid)
        return sim_results(max_angular_frequency_dev, min_angular_frequency_dev, surv_vol_condition, infeasible, ω_excitability, voltage_excitability, P_diff_global, Q_diff_global, P_diff_local, Q_diff_local, mfd_final, final_diff_v, x_new, final_ω_excitability, final_voltage_excitability, survival_time, first_node_ω_out_of_bounds, first_node_voltage_out_of_bounds, sim_time, bytes)
    end

    function obtain_results(data_dir, grid_index_start, grid_index_end)
        results_grid_one = read_simulation_results_for_one_grid(data_dir, grid_index_start)
        pert_per_node = size(results_grid_one.max_angular_frequency_dev, 2)
        # init arrays
        max_angular_frequency_dev = zeros(0, pert_per_node)
        min_angular_frequency_dev  = zeros(0, pert_per_node)
        surv_vol_condition = zeros(0, pert_per_node)
        infeasible = zeros(Int32, 0, pert_per_node)
        ω_excitability = zeros(0)
        voltage_excitability = zeros(0)
        P_diff_global = zeros(0, pert_per_node)
        Q_diff_global = zeros(0, pert_per_node)
        P_diff_local = zeros(0, pert_per_node)
        Q_diff_local = zeros(0, pert_per_node)
        mfd_final  = zeros(0, pert_per_node)
        final_diff_v  = zeros(0, pert_per_node)
        all_x_new = Dict()
        final_ω_excitability = zeros(0)
        final_voltage_excitability = zeros(0)
        survival_time = zeros(0, pert_per_node)
        sim_time = zeros(0)
        bytes = zeros(0)
        first_node_ω_out_of_bounds = Dict()
        first_node_voltage_out_of_bounds = Dict()
        for grid_index in grid_index_start:grid_index_end
            results = read_simulation_results_for_one_grid(data_dir, grid_index)
            max_angular_frequency_dev = vcat(max_angular_frequency_dev, results.max_angular_frequency_dev)
            min_angular_frequency_dev = vcat(min_angular_frequency_dev, results.min_angular_frequency_dev)
            surv_vol_condition = vcat(surv_vol_condition, results.surv_vol_condition)            
            infeasible = vcat(infeasible, results.infeasible)
            append!(ω_excitability, results.ω_excitability)
            append!(voltage_excitability, results.voltage_excitability)
            P_diff_global = vcat(P_diff_global, results.P_diff_global)
            Q_diff_global = vcat(Q_diff_global, results.Q_diff_global)
            P_diff_local = vcat(P_diff_local, results.P_diff_local)
            Q_diff_local = vcat(Q_diff_local, results.Q_diff_local)
            mfd_final = vcat(mfd_final, results.mfd_final)
            final_diff_v = vcat(final_diff_v, results.final_diff_v)
            append!(final_ω_excitability, results.final_ω_excitability)
            append!(final_voltage_excitability, results.final_voltage_excitability)
            survival_time = vcat(survival_time, results.survival_time)
            append!(sim_time, results.sim_time)
            append!(bytes, results.bytes)
            first_node_ω_out_of_bounds[grid_index] = results.first_node_ω_out_of_bounds
            first_node_voltage_out_of_bounds[grid_index] = results.first_node_voltage_out_of_bounds
            all_x_new[grid_index] = results.all_x_new
        end
        return sim_results(max_angular_frequency_dev, min_angular_frequency_dev, surv_vol_condition, infeasible, ω_excitability, voltage_excitability, P_diff_global, Q_diff_global, P_diff_local, Q_diff_local, mfd_final, final_diff_v, all_x_new, final_ω_excitability, final_voltage_excitability, survival_time, first_node_ω_out_of_bounds, first_node_voltage_out_of_bounds, sim_time, bytes)
    end

    function count_converged_from_bitMatrix(input_matrix)
        N, num_pert_per_node = size(input_matrix)
        output = zeros(N)
        for n in 1:N
            output[n] = size(findall(input_matrix[n, :] .== 1), 1)
        end
        output ./ num_pert_per_node
    end

    function compute_statistics(results, thresholds, consider_infeasible)
        infeasible = results.infeasible .> 0
        max_angular_frequency_dev = results.max_angular_frequency_dev
        min_angular_frequency_dev = results.min_angular_frequency_dev
        surv_vol_condition = results.surv_vol_condition
        ω_excitability = results.ω_excitability
        voltage_excitability = results.voltage_excitability
        mfd_final = results.mfd_final
        final_diff_v = results.final_diff_v

        num_nodes, pert_per_node = pert_per_node = size(results.max_angular_frequency_dev)

        threshold_ω_surv_low = thresholds.threshold_ω_surv_low
        threshold_ω_surv_high = thresholds.threshold_ω_surv_high
        threshold_ω_snbs = thresholds.threshold_ω_snbs
        threshold_u_snbs = thresholds.threshold_u_snbs

        statistics = Dict()

        # surv_ω
        surv_ω_bit_matrix = (min_angular_frequency_dev .> threshold_ω_surv_low) .&& (max_angular_frequency_dev .< threshold_ω_surv_high)
        if consider_infeasible == true
            surv_ω_bit_matrix .*= .!infeasible
        end
        surv_ω = count_converged_from_bitMatrix(surv_ω_bit_matrix)
        statistics["surv_ω"] = surv_ω

        # surv_u
        surv_u_bit_matrix = surv_vol_condition
        if consider_infeasible == true
            surv_u_bit_matrix .*= .!infeasible
        end
        surv_u = count_converged_from_bitMatrix(surv_u_bit_matrix)
        statistics["surv_u"] = surv_u

        # surv
        surv_bit_matrix = surv_ω_bit_matrix .&& surv_u_bit_matrix
        surv = count_converged_from_bitMatrix(surv_bit_matrix)
        statistics["surv"] = surv

        # snbs_ω
        snbs_ω_bit_matrix = mfd_final .< threshold_ω_snbs
        if consider_infeasible == true
            snbs_ω_bit_matrix .*= .!infeasible
        end
        snbs_ω = count_converged_from_bitMatrix(snbs_ω_bit_matrix)
        statistics["snbs_ω"] = snbs_ω
        
        # snbs_u
        snbs_u_bit_matrix = final_diff_v .< threshold_u_snbs
        if consider_infeasible == true
            snbs_u_bit_matrix .*= .!infeasible
        end
        snbs_u = count_converged_from_bitMatrix(snbs_u_bit_matrix)
        statistics["snbs_u"] = snbs_u
        
        # snbs 
        snbs_bit_matrix = snbs_ω_bit_matrix .&& snbs_u_bit_matrix
        snbs = count_converged_from_bitMatrix(snbs_bit_matrix)
        statistics["snbs"] = snbs

        # snbs = count_converged_from_bitMatrix(mfd_final .< 0.1)
        # if consider_infeasible == true
        #     snbs .*= .!infeasible
        # end
        # statistics["snbs"] = snbs

        # excitability
        statistics["excitability_ω"] = ω_excitability / (pert_per_node * num_nodes)
        statistics["excitability_u"] = voltage_excitability / (pert_per_node * num_nodes)



        return statistics
    end

    function get_correct_edge_combination(all_combinations, first_node_type, second_node_type)
        num_combs = length(all_combinations)
        for i in 1:num_combs
            if (all_combinations[i]["first_node_type"] == first_node_type) & (all_combinations[i]["second_node_type"] == second_node_type)
                return i
            elseif (all_combinations[i]["second_node_type"] == first_node_type) & (all_combinations[i]["second_node_type"] == second_node_type)
                return i
            end
        end

    end
end

function get_array_of_desired_columns(desired_variables)
    if typeof(desired_variables) == Vector{String}
        return unique(desired_variables)
    else
        all_names = Vector{String}()
        for (key, values) in desired_variables
            all_names = vcat(all_names, values)
        end
        return unique(all_names)
    end

end

function feature_scaling_df(df, desired_variables, scaling)
    name_columns = get_array_of_desired_columns(desired_variables)
    new_df = copy(df)
    for i in 1:length(name_columns)
        column = name_columns[i]
        if scaling["mode"] == "normalize"
            # new_df[!, column] = normalize_array(df[!, column], scaling[!, column])
            new_df[!, column] = normalize_array(df[!, column], scaling["variables"][column])
        elseif scaling["mode"] == "standardize"
            new_df[!, column] = standardize_array(df[!, column], scaling["variables"][column])
        end
    end
    return new_df
end


function scale_node_line_data(grid_data_nodes, grid_data_lines, desired_variables_per_node_type, desired_variables_per_line, scaling)
    if scaling["nodes"]["mode"] != false || scaling["lines"]["mode"] != false
        scaled_nodes = feature_scaling_df(grid_data_nodes, desired_variables_per_node_type, scaling["nodes"])
        scaled_lines = feature_scaling_df(grid_data_lines, desired_variables_per_line, scaling["lines"])
        return scaled_nodes, scaled_lines
    else
        return grid_data_nodes, grid_data_lines
    end
end


function get_all_possible_combinations(node_types)
    possible_edges_combinations_same = collect(combinations(node_types, 1))
    possible_edges_combinations_different = collect(combinations(node_types, 2))
    all_combs = [[possible_edges_combinations_same[1][1], possible_edges_combinations_same[1][1]]]
    for i in 2:length(possible_edges_combinations_same)
        all_combs = vcat(all_combs, [[possible_edges_combinations_same[i][1], possible_edges_combinations_same[i][1]]])
    end
    for i in 1:length(possible_edges_combinations_different)
        all_combs = vcat(all_combs, [[possible_edges_combinations_different[i][1], possible_edges_combinations_different[i][2]]])
        all_combs = vcat(all_combs, [[possible_edges_combinations_different[i][2], possible_edges_combinations_different[i][1]]])
    end
    return all_combs

end


function apply_imputation!(df, imputation_info)
    for i in 1:length(imputation_info)
        imp_variable = imputation_info[i]["variable"]
        imp_node_type_int = imputation_info[i]["node_type_int"]
        imp_value = imputation_info[i]["value"]
        load_idx = df.node_type_int .== imp_node_type_int
        df[!, imp_variable][load_idx] .= imp_value
    end
end


function get_homo_nodes_lines_targets(statistics, grid_data_nodes, grid_data_lines, desired_variables_per_node_type, desired_variables_per_line, add_node_type_load, add_node_type_slack, scaling)
    scaled_df_nodes, scaled_df_lines = scale_node_line_data(grid_data_nodes, grid_data_lines, desired_variables_per_node_type, desired_variables_per_line, scaling)
    apply_imputation!(scaled_df_nodes, scaling["nodes"]["imputation"])
    desired_variables_node_array = get_array_of_desired_columns(desired_variables_per_node_type)
    if add_node_type_load == true
        push!(desired_variables_node_array, "node_type_load")
    end
    if add_node_type_slack == true
        push!(desired_variables_node_array, "node_type_slack")
    end
    features = scaled_df_nodes[!, desired_variables_node_array]
    node_features = Dict()
    node_features["features"] = features

    targets = Dict()
    targets["surv"] = statistics["surv"]
    # targets["snbs"] = statistics["snbs"]
    targets["excitability_ω"] = statistics["excitability_ω"]
    targets["excitability_u"] = statistics["excitability_u"]
    node_features["targets"] = targets


    num_line_features = size(desired_variables_per_line, 1)
    num_lines = size(scaled_df_lines, 1)
    adjacency_matrix = zeros(Int32, 2, 2 * num_lines)
    line_features = zeros(num_line_features, 2 * num_lines)

    from_array = scaled_df_lines.from
    to_array = scaled_df_lines.to
    for i in 1:num_lines
        column_index = (i - 1) * 2 + 1
        adjacency_matrix[1, column_index] = from_array[i]
        adjacency_matrix[2, column_index] = to_array[i]
        adjacency_matrix[1, column_index+1] = to_array[i]
        adjacency_matrix[2, column_index+1] = from_array[i]
        for j in 1:num_line_features
            variable = desired_variables_per_line[j]
            line_features[j, column_index] = scaled_df_lines[i, variable]
        end
    end
    line_properties = Dict()
    line_properties["edge_index"] = adjacency_matrix
    line_properties["edge_attr"] = line_features
    return node_features, line_properties

end

function get_hetero_nodes_lines_targets(statistics, grid_data_nodes, grid_data_lines, desired_variables_per_node_type, desired_variables_per_line, scaling)
    node_types_array = grid_data_nodes.node_type
    node_types = unique(node_types_array)
    num_node_types = size(node_types, 1)
    scaled_df_nodes, scaled_df_lines = scale_node_line_data(grid_data_nodes, grid_data_lines, desired_variables_per_node_type, desired_variables_per_line, scaling)
    node_features = Dict()
    for i in 1:num_node_types
        node_type = node_types[i]
        # if node_type != "SlackAlgebraic"
        idx_nodes = findall(node_types_array .== node_type)
        one_node_properties = Dict()
        one_node_properties["features"] = scaled_df_nodes[idx_nodes, desired_variables_per_node_type[node_type]]
        targets = Dict()
        targets["surv"] = statistics["surv"][idx_nodes]
        # targets["snbs"] = statistics["snbs"][idx_nodes]
        targets["excitability_ω"] = statistics["excitability_ω"][idx_nodes]
        targets["excitability_u"] = statistics["excitability_u"][idx_nodes]
        one_node_properties["targets"] = targets
        node_features[node_type] = one_node_properties
        # end
    end
    possible_edges_combinations = get_all_possible_combinations(node_types)
    num_possible_comb = size(possible_edges_combinations, 1)
    line_features = Dict()
    for i in 1:num_possible_comb
        one_comb = Dict()
        one_comb["first_node_type"] = possible_edges_combinations[i][1]
        if size(possible_edges_combinations[i], 1) == 1
            one_comb["second_node_type"] = possible_edges_combinations[i][1]
        else
            one_comb["second_node_type"] = possible_edges_combinations[i][2]
        end
        one_comb["edge_index"] = zeros(Int32, 2, 0)
        one_comb["edge_attr"] = zeros(0, 5)
        line_features[i] = one_comb
    end
    ## add new idx to node features
    scaled_df_nodes[!, :new_node_idx] .= -1
    for i in 1:length(node_types)
        counter_idx = 1
        for j in 1:length(scaled_df_nodes.node_type)
            if scaled_df_nodes.node_type[j] == node_types[i]
                scaled_df_nodes[j, "new_node_idx"] = counter_idx
                counter_idx += 1
            end
        end
    end

    num_edges = size(scaled_df_lines, 1)
    for i in 1:num_edges
        first_node_type = scaled_df_nodes[scaled_df_lines[i, "from"], "node_type"]
        second_node_type = scaled_df_nodes[scaled_df_lines[i, "to"], "node_type"]
        index_comb_1 = get_correct_edge_combination(line_features, first_node_type, second_node_type)
        line_features[index_comb_1]["edge_index"] = hcat(line_features[index_comb_1]["edge_index"], [scaled_df_nodes[scaled_df_lines[i, "from"], "new_node_idx"], scaled_df_nodes[scaled_df_lines[i, "to"], "new_node_idx"]])
        line_attr = Array(scaled_df_lines[i, desired_variables_per_line])
        line_features[index_comb_1]["edge_attr"] = vcat(line_features[index_comb_1]["edge_attr"], transpose(line_attr))
        index_comb_2 = get_correct_edge_combination(line_features, second_node_type, first_node_type)
        line_features[index_comb_2]["edge_index"] = hcat(line_features[index_comb_2]["edge_index"], [scaled_df_nodes[scaled_df_lines[i, "to"], "new_node_idx"], scaled_df_nodes[scaled_df_lines[i, "from"], "new_node_idx"]])
        line_attr = Array(scaled_df_lines[i, desired_variables_per_line])
        line_features[index_comb_2]["edge_attr"] = vcat(line_features[index_comb_2]["edge_attr"], transpose(line_attr))
    end
    return node_features, line_features
end

function prepare_grid_data_hetero(data_dir, filename_ml, grid_data_lines, grid_data_nodes, desired_variables_per_node_type, desired_variables_per_line, thresholds, scaling, consider_infeasible=true)
    fid_grids = h5open(filename_ml, "w")
    grids_group = create_group(fid_grids, "grids")

    for grid_id in 1:length(grid_data_nodes)
        grid_id_group = create_group(grids_group, string(grid_id))
        results = read_simulation_results_for_one_grid(data_dir, grid_id)
        statistics = compute_statistics(results, thresholds, consider_infeasible)
        node_features, line_properties = get_hetero_nodes_lines_targets(statistics, grid_data_nodes[grid_id], grid_data_lines[grid_id], desired_variables_per_node_type, desired_variables_per_line, scaling)
        node_features_group = create_group(grid_id_group, "node_features_group")
        load_group = create_group(node_features_group, "load")
        load_group["features"] = Array(node_features["load"]["features"])
        load_group_targets = create_group(load_group, "targets")
        load_targets = node_features["load"]["targets"]
        for (key, value) in load_targets
            load_group_targets[key] = value
        end

        normalForm_group = create_group(node_features_group, "normalForm")
        normalForm_group["features"] = Array(node_features["normalForm"]["features"])
        normalForm_group_targets = create_group(normalForm_group, "targets")
        normalForm_targets = node_features["normalForm"]["targets"]
        for (key, value) in normalForm_targets
            normalForm_group_targets[key] = value
        end
        slackAlgebraic_group = create_group(node_features_group, "SlackAlgebraic")
        slackAlgebraic_group["features"] = 1

        line_features_group = create_group(grid_id_group, "line_features_group")
        for i in 1:length(line_properties)
            one_line_feature_group = create_group(line_features_group, string(i))
            one_line_feature_group["first_node_type"] = line_properties[i]["first_node_type"]
            one_line_feature_group["second_node_type"] = line_properties[i]["second_node_type"]
            one_line_feature_group["line_type"] = "PIModelLine"
            one_line_feature_group["edge_index"] = Array(line_properties[i]["edge_index"])
            one_line_feature_group["edge_attr"] = Array(line_properties[i]["edge_attr"])
        end
    end
    close(fid_grids)
end

function prepare_grid_data_homo(data_dir, filename_ml, grid_data_lines, grid_data_nodes, desired_variables_per_node_type, desired_variables_per_line, add_node_type_load, add_node_type_slack, thresholds, scaling, consider_infeasible=true)
    fid = h5open(filename_ml, "w")
    grids_group = create_group(fid, "grids")
    for grid_id in 1:length(grid_data_lines)
        grid_id_group = create_group(grids_group, string(grid_id))
        results = read_simulation_results_for_one_grid(data_dir, grid_id)
        statistics = compute_statistics(results, thresholds, consider_infeasible)
        node_features, line_properties = get_homo_nodes_lines_targets(statistics, grid_data_nodes[grid_id], grid_data_lines[grid_id], desired_variables_per_node_type, desired_variables_per_line, add_node_type_load, add_node_type_slack, scaling)
        grid_id_group["node_features"] = Array(node_features["features"])
        group_targets = create_group(grid_id_group, "targets")

        for (key, value) in node_features["targets"]
            group_targets[key] = value
        end
        grid_id_group["edge_attr"] = line_properties["edge_attr"]
        grid_id_group["edge_index"] = line_properties["edge_index"]
        mask = grid_data_nodes[grid_id][!, "node_type_int"] .!= 4
        grid_id_group["mask"] = Int.(mask)
    end
    close(fid)
end

function add_power_flow!(df_edges, power_flow_P, power_flow_Q)
    df_edges.power_flow_P .= 0.0
    df_edges.power_flow_Q .= 0.0
    for i in range(1, size(df_edges, 1))
        from = df_edges[i, "from"]
        to = df_edges[i, "to"]
        df_edges[i, "power_flow_P"] = power_flow_P[[from, to]]
        df_edges[i, "power_flow_Q"] = power_flow_Q[[from, to]]
    end
end

function add_nodewise_sum_shunts_admittances!(df_nodes, df_edges)
    df_nodes.sum_y_real .= 0.0
    df_nodes.sum_y_imag .= 0.0
    df_nodes.sum_y_shunt_mk_imag .= 0.0
    for node_idx in 1:size(df_nodes, 1)
        sum_y_real = 0
        sum_y_imag = 0
        sum_y_shunt_mk_imag = 0
        for j in 1:size(df_edges, 1)
            if df_edges[j, "from"] == node_idx || df_edges[j, "to"] == node_idx
                sum_y_real += df_edges[j, "y_real"]
                sum_y_imag += df_edges[j, "y_imag"]
                sum_y_shunt_mk_imag += df_edges[j, "y_shunt_mk_imag"]
            end
        end
        df_nodes[node_idx, "sum_y_real"] = sum_y_real
        df_nodes[node_idx, "sum_y_imag"] = sum_y_imag
        df_nodes[node_idx, "sum_y_shunt_mk_imag"] = sum_y_shunt_mk_imag
    end
end