using Test

include(joinpath(@__DIR__,"../src/dynamic_simulations.jl"))


const ω_pert_size_low = convert(Float32, -2.0 * pi)
const ω_pert_size_high = convert(Float32, 2.0 * pi)


sob = [
    0.7150174823037546 0.0 0.0; 
    5.322724832672328e-17 -0.8692669325356855 -3.1415927410125732;
    3.1632593633698585e-17 0.5165994579942956 3.1415927410125732; 
    0.44105696910943376 -0.4410569691094337 1.5707963705062866; 
    -0.6625 0.6625000000000001 -4.71238911151886]

# Test against "ground truth"
@test sob == get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, 5; v_size_low=0.15^2, skipbool=false)
# Test start of series constant
@test sob == get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, 6; v_size_low=0.15^2, skipbool=false)[1:5,:]
# Test skip working as intended
@test get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, 6; v_size_low=0.15^2, skipbool=false)[4:6,:] == get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, 3; v_size_low=0.15^2, skipbool=true)
# Test third dimenison independet of v_size_low
@test get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, 5; v_size_low=0.15^2)[:,3] == get_soboli_perturbations(ω_pert_size_low, ω_pert_size_high, 5; v_size_low=0.0)[:,3]
