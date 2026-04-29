"""SI scaling: CharacteristicScalesSI, paper dimensionless groups, SimParams build, plot/thrust norms."""
module PaperScales

using ..PartCount

export CharacteristicScalesSI,
    e_C,
    THRUSTER_EFFECTIVE_AREA_M2,
    paper_dimensionless_from_si,
    k_I_from_beta0_si,
    sim_params_from_si_scales,
    alfven_reference_scales,
    si_plot_physical_scales,
    nu_ei_spitzer_hz,
    E0_dimless_from_discharge_voltage,
    v_pic0_dimless_from_ion_thermal,
    beta0_si_from_target_k_I,
    thrust_momentum_SI

# This file lives in `src/`; partials live in `src/paper_scales/`.
const _PS = joinpath(@__DIR__, "paper_scales")
include(joinpath(_PS, "constants.jl"))
include(joinpath(_PS, "collision.jl"))
include(joinpath(_PS, "dimensionless_from_si.jl"))
include(joinpath(_PS, "sim_from_si.jl"))
include(joinpath(_PS, "plot_scaling.jl"))

end
