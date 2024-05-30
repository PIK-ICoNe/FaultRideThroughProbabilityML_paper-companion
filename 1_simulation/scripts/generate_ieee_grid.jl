using Pkg
package_dir = joinpath(@__DIR__, "../")
Pkg.activate(package_dir)
using Revise
using GNN_BS_new_dataset
using PowerDynamics
using SyntheticPowerGrids
import SyntheticPowerGrids.line_properties_380kV
import SyntheticPowerGrids.validate_power_flow_on_lines


using LibGit2
r = LibGit2.GitRepo(package_dir)
println("Git branch: ",LibGit2.branch(r))
println("GitHash: ",LibGit2.GitHash(LibGit2.head(r)))
println("GIT local changes ", LibGit2.isdirty(r))


##
ieee96_adapted, df = get_ieee_96(line_paras = :Dena_380kV, virtual_inertia = :SyntheticPowerGrids, reactive_power = :Ieee96) # Linearly Stable??
op_adapted = find_operationpoint(ieee96_adapted)


validate_power_flow_on_lines(op_adapted, :PiModelLine)

function store_grid(input_name, pg, op)
    grid_name = joinpath(string("data/grids/grid_", input_name,".json"))
    write_powergrid(pg, grid_name, Json)
    file_state_name = joinpath(string("data/grids/state_", input_name, ".json"))
    write_state(op, file_state_name)
end


function write_state(state,file_name)
    open(file_name,"w") do f
        print(f, state.vec)
    end
end

store_grid("00001", ieee96_adapted, op_adapted)



## original grid 
#ieee96_original, df = get_ieee_96(line_paras = :Ieee96, virtual_inertia = :Ieee96)
# ieee96_original, df = get_ieee_96(line_paras = :Ieee96, virtual_inertia = :SyntheticPowerGrids)
# op_original = find_operationpoint(ieee96_original)
# validate_power_flow_on_lines(op_original, a)
# store_grid("original", ieee96_original, op_original)
