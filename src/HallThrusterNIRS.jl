module HallThrusterNIRS

include("numerical.jl")
include("physics.jl")
include("dimensionless.jl")
include("solver.jl")

using .PartCount
using .NumericalFunctionsSPT
using .PlasmaDynamics
using .ParticleMovementSPT
using .MagneticField
using .Visualization
using .VisualizationDimensional
using .DiagnosticsMetrics
using .PaperScales
using .CoreSolver

module ConfigDefaults

using ..PartCount
using ..MagneticField

export build_default_params

function build_default_params()
    λ_e_λΣ_calc = 9.11e-31 / (2.18e-25 + 9.11e-31)
    return PartCount.SimParams(
        L = 1, M = 50,
        mi = 1.0, me = 4.14e-06,
        T_ion = 1.0, v_a = 1.0, n_a_left = 10.0, kI = 2.2, kR = 0.05,
        γ = 5 / 3, ε = 0.67, ν_m0 = 0.03,
        α = 0.00941, α0 = 120.05, ζ = 0.01, ε_dim = 1.0,
        λ_e_λΣ = λ_e_λΣ_calc,
        c_inv = 1.0,
        H0_func = z -> MagneticField.gaussian_Br(z, 1.0, 0.85, 0.18),
        N1 = 50,
        pic_charge_factor = 1.0,
    )
end

end

module Runner

using ..ConfigDefaults
using ..CoreSolver

export run_default_case

function run_default_case(;
    total_time = 20.0,
    save_times = [10.0, 15.0, 20.0],
    do_plot = true,
    plot_output_dir = joinpath(@__DIR__, "..", "output", "figures"),
)
    params = ConfigDefaults.build_default_params()
    return CoreSolver.run_simulation(params; total_time = total_time, save_times = save_times, do_plot = do_plot, plot_output_dir = plot_output_dir)
end

end

using .ConfigDefaults
using .Runner

export PartCount, NumericalFunctionsSPT, PlasmaDynamics, ParticleMovementSPT
export MagneticField, Visualization, VisualizationDimensional
export DiagnosticsMetrics, PaperScales, CoreSolver
export ConfigDefaults, Runner
export run_simulation, build_default_params, run_default_case
export CharacteristicScalesSI,
    THRUSTER_EFFECTIVE_AREA_M2,
    paper_dimensionless_from_si,
    alfven_reference_scales,
    si_plot_physical_scales,
    nu_ei_spitzer_hz,
    E0_dimless_from_discharge_voltage,
    v_pic0_dimless_from_ion_thermal,
    beta0_si_from_target_k_I,
    thrust_momentum_SI

const run_simulation = CoreSolver.run_simulation
const build_default_params = ConfigDefaults.build_default_params
const run_default_case = Runner.run_default_case

end
