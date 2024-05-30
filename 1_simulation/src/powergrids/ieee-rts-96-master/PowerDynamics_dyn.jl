function get_ieee_96(; line_paras::Symbol, virtual_inertia::Symbol, reactive_power::Symbol)
    @info "Import system RTS-96 (from prepared csv files)"
    ### Load Network Structure and Parameters from Data
    path = string(@__DIR__) * "/Data/"
    node_df, line_df = get_ieee_data_frame(path)
   
    N = nrow(node_df) # Number of node

    slack_idx = argmax(skipmissing(node_df.P_Gen)) # Slack bus is the largest generator
    
    lines = get_lines_ieee(line_df, line_paras) # Transmission lines

    op_cons = get_ancillary_op_ieee(node_df, slack_idx, lines, reactive_power, N) # Helper operation point, used for the reactive power

    τ_list = get_virtual_inertia(virtual_inertia, node_df, N) # List of time constant low pass filter measuring the active power

    nodes = get_nodes_ieee(node_df, slack_idx,  N, τ_list, op_cons)
    
    return PowerGrid(nodes, lines), node_df
end

"""
    get_ieee_data_frame(path)

Loads the data from the Ieee 96 into data DataFrames
    -`path`: Path to the original data set
"""
function get_ieee_data_frame(path)
    println("Generating DataFrame...")

    BusData = CSV.read(joinpath(path, "Bus.csv"), DataFrame)
    line_df = CSV.read(joinpath(path, "Line.csv"), DataFrame)
    line_df[!, :source] = line_df.source .|> Int
    line_df[!, :dest] = line_df.dest .|> Int
    GeneratorData = CSV.read(joinpath(path, "Generator.csv"), DataFrame)
    LoadData = CSV.read(joinpath(path, "Load.csv"), DataFrame)
    FlowData = CSV.read(joinpath(path, "Flow.csv"), DataFrame)

    node_df = outerjoin(BusData, GeneratorData, LoadData, FlowData, on=:ID, makeunique=true)
    return node_df, line_df
end

"""
    get_ancillary_op_ieee(node_df, slack_idx, lines, reactive_power, N)

Generates a helper operation point. Aids in the load flow calculation of the full power grid. 
    -`node_df`: DataFrame holding the nodal properties
    -`slack_idx`: Index of the slack Bus
    -`lines`: Vector containing the PowerDynamics lines
    -`reactive_power`: Decides if the reactive power of the original data set is used or if the reactive power is calculated for perfect voltage magnitudes.
    -`N`: Number of nodes
"""
function get_ancillary_op_ieee(node_df, slack_idx, lines, reactive_power, N)
    println("Generating helper operation point...")

    node_types = fill(:PVAlgebraic, N)
    gen_idx = findall(ismissing.(node_df.P_Gen) .== false)

    if reactive_power == :Ieee96 # For the loads use the reactive power from the ieee dataset
        load_idx = findall(ismissing.(node_df.P_Gen)) # All nodes without generation are loads
        node_types[load_idx] .= :PQAlgebraic  
        Q_vec = node_df.Q_Load
        V_vec = node_df.Vm
        V_vec[slack_idx] = node_df.Vm[slack_idx] * exp(1im * node_df.Va[slack_idx])

    elseif reactive_power == :perfect_voltage_mag
        Q_vec = fill(nothing, N) # The reactive power set-points are not used
        V_vec = ones(N)
    else
        error("This option is not implemented.")
    end
    
    P_vec = -node_df.P_Load
    P_vec[gen_idx] = P_vec[gen_idx] .+ node_df.P_Gen[gen_idx]

    op_cons = get_ancillary_operationpoint(P_vec, Q_vec, V_vec, node_types, slack_idx, lines)

    return op_cons
end

"""
    get_lines_ieee(line_df, line_paras)

Generates the list of PowerDynamics lines for the Ieee96 power grid.
    -`line_df`: DataFrame holding the line properties
    -`line_paras`: Decides if the admittances of the lines are the same as in the ieee96 data set or as in the SyntheticPowerGrids package.
"""
function get_lines_ieee(line_df, line_paras)
    println("Generating lines...")

    lines = []
    if line_paras == :Ieee96
        for line in eachrow(line_df)
            push!(lines, PiModelLine(; from=line.source, to=line.dest, y=inv(complex(line.r, line.x)), y_shunt_km=-0.5im * line.b, y_shunt_mk=-0.5im * line.b))
        end
    elseif line_paras == :Dena_380kV # Use Dena Standard Parameters 
        Z_base = (380 * 10^3)^2 / (100 * 10^6) # Base impedance
        Y_base = 1 / Z_base
        for line in eachrow(line_df)
            if line.length == 0.0 # Set minimal line lengths
                Z, Y_shunt = SyntheticPowerGrids.line_properties_380kV(0.06, 3)
            else
                Z, Y_shunt = SyntheticPowerGrids.line_properties_380kV(line.length, 3)
            end

            Z_pu = Z / Z_base # Per Unit conversion
            Y_shunt_pu = Y_shunt / Y_base

            push!(lines, PiModelLine(; from=line.source, to=line.dest, y=inv(Z_pu), y_shunt_km=0.5 * Y_shunt_pu, y_shunt_mk=0.5 * Y_shunt_pu))
        end
    else
        error("This option is not implemented.")
    end

    return lines
end

"""
    get_virtual_inertia(virtual_inertia, node_df, N)

Generates a list of inertia constants/ time constants for the low pass filter in a droop controlled inverter.
    -`virtual_inertia`: Decides if the inertia constants of the Ieee96 are used or if they are randomly sampled.
    -`node_df`: DataFrame holding the nodal properties
    -`N`: Number of nodes
"""
function get_virtual_inertia(virtual_inertia, node_df, N)
    if virtual_inertia == :Ieee96 # Inertia constants from the ieee power grid
        τ_list = node_df.Inertia
    elseif virtual_inertia == :SyntheticPowerGrids
        gen_nodes = findall(ismissing.(node_df.P_Gen) .== false) # Find all generator nodes
        τ_list = zeros(N) # Hold virtual inertia constants
        rng = MersenneTwister(42) # Reproducibly
        τ_s = [0.5, 1.0, 5.0] # Virtual Inertia Constants used in SyntheticPowerGrids

        for i in eachindex(τ_s) # Go over all options
            s = sample(rng, gen_nodes, 11, replace=false) # Sample virtual inertia
            symdiff!(gen_nodes, s) # Remove generators which were already sampled

            τ_list[s] .= τ_s[i]
        end
    else
        error("This option is not implemented.")
    end
    return τ_list
end

"""
    get_nodes_ieee(node_df, slack_idx, N, τ_list, op_cons)

Generates the list of PowerDynamics nodes for the Ieee96 power grid.
    -`node_df`: DataFrame holding the nodal properties
    -`slack_idx`: Index of the slack Bus
    -`N`: Number of nodes
    -`τ_list`: List of inertia constants / time constants
    -`op_cons`: Helper operation point. Used for the load flow
"""
function get_nodes_ieee(node_df, slack_idx, N, τ_list, op_cons)
    println("Generating nodes...")

    nodes = []
    for i in 1:N
        n = node_df[i, :]
        if n.Number == slack_idx
            push!(nodes, SlackAlgebraic(U = n.Vm * exp(1im * n.Va))) 
        else
            n = node_df[i, :]
            if n.P_Gen |> ismissing # Nodes without generation are loads
                push!(nodes, PQAlgebraic(P=op_cons[i, :p], Q=op_cons[i, :q]))
            else
                τ_P = τ_list[i] # Time constant low pass filter measuring the active power
                τ_Q = 8.0 # Time constant low pass filter measuring the reactive power
                K_P = 5.0 # Gain constant low pass filter measuring the active power
                K_Q = 0.1 # Gain constant low pass filter measuring the reactive power

                Bᵤ, Cᵤ, Gᵤ, Hᵤ, Bₓ, Cₓ, Gₓ, Hₓ, _ = parameter_DroopControlledInverterApprox(τ_Q=τ_Q, K_P=K_P, K_Q=K_Q, V_r=op_cons[i, :v], τ_P=τ_P, Y_n=0.0)
                nf = NormalForm(P=op_cons[i, :p], Q=op_cons[i, :q], V=op_cons[i, :v], Bᵤ=Bᵤ, Cᵤ=Cᵤ, Gᵤ=Gᵤ, Hᵤ=Hᵤ, Bₓ=Bₓ, Cₓ=Cₓ, Gₓ=Gₓ, Hₓ=Hₓ)
                push!(nodes, nf)
            end
        end
    end
    return nodes
end