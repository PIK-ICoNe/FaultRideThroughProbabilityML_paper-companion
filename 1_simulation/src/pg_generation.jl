"""
    grid_generation(r,  pg_struct::PGGeneration, num_grids)

Generates `num_grids` different power grids with `N` nodes and stores them.

- `pg_struct::PGGeneration`: Struct containing all information's about the power grid.
- `r::Int`: Grid Index
- `num_grids::Int` total number of grids to sample in this run
"""
function grid_generation(
    r::Int,
    pg_struct::PGGeneration,
    num_grids::Int,
    P0,
    path_storage_data
)

    pg, op, grid_pg_struct, rejections = generate_powergrid_dynamics(pg_struct)
    id = lpad(r, length(digits(num_grids)), '0')
    grid_name = joinpath(path_storage_data, string("grids/grid_", id, ".json"))
    write_powergrid(pg, grid_name, Json)
    vertexpos = grid_pg_struct.embedded_graph.vertexpos
    vertexpos_file_name = joinpath(path_storage_data, string("grids/grid_", id, "_vertexpos.h5"))
    write_vertexpos(vertexpos, vertexpos_file_name)

    P0_file_name = joinpath(path_storage_data, string("grids/grid_", id, "_P0.h5"))
    write_P0(P0, P0_file_name)

    file_state_name = joinpath(path_storage_data, string("grids/state_", id, ".json"))
    write_state(op, file_state_name)

    println("For grid $r there were $rejections rejected power grids.", r, rejections.total)

    counter_name = joinpath(path_storage_data, string("rejection_info/grid_", id, ".txt"))
    open(counter_name, "w") do f
        println(f, rejections.total)
    end
end

function write_state(state, file_name)
    open(file_name, "w") do f
        print(f, state.vec)
    end
end