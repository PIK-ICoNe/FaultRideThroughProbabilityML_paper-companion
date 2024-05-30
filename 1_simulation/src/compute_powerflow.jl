"""
    get_power_flow_on_lines(operation_point::State)

Calculates the power flow on the transmission lines of a grid. 
"""
function get_power_flow_on_lines(operation_point::State)
    lines = operation_point.grid.lines

    P_dict = Dict()
    Q_dict = Dict()
    for j in eachindex(lines)
        l = lines[j]
        m = l.from # Source 
        k = l.to # Destination
        
        V_m = operation_point[m, :u]  # Voltage node m 
        V_k = operation_point[k, :u]  # Voltage node k

        if typeof(l) == StaticLine
            Y_km = l.Y
            b_sh = 0.0

        elseif typeof(l) == PiModelLine
            Y_km = l.y

            b_sh = imag.(l.y_shunt_mk) # we don't have shunt conductances in our model
        end

        I_km = Y_km * (V_k - V_m) + 1im * b_sh * V_k
        I_mk = Y_km * (V_m - V_k) + 1im * b_sh * V_m

        S_km = V_k * conj(I_km)
        S_mk = V_m * conj(I_mk)

        P_dict[[k, m]] = real(S_km)
        P_dict[[m, k]] = real(S_mk)

        Q_dict[[k, m]] = imag(S_km)
        Q_dict[[m, k]] = imag(S_mk)
    end  
    return P_dict, Q_dict
end