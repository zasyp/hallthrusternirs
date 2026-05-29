
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "HallThrusterNIRS.jl"))
using .HallThrusterNIRS

const MI_AMU = 54.0
const MP_ME  = 1836.152672
const ME_OVER_MI = 1.0 / (MI_AMU * MP_ME)
const λ_i_λΣ = 1.0 / (1.0 + ME_OVER_MI)
const λ_e_λΣ = ME_OVER_MI / (1.0 + ME_OVER_MI)

const κ_paper  = 1.0
const ξ_paper  = 0.532
const ζ_paper  = 0.061
const ε_paper  = 1.0
const H0_B_max = 1.0
const H0_z0    = 0.75
const H0_σ     = 0.20

const λi_λe    = (1.0 + ME_OVER_MI) / ME_OVER_MI
const α_paper  = λ_i_λΣ * ξ_paper^2
const α0_paper = κ_paper * ξ_paper * λ_i_λΣ * sqrt(λi_λe)

params = PartCount.SimParams(;
    L = 1.0,
    M = 50,
    N1 = 50,
    mi = 1.0,
    me = ME_OVER_MI,
    T_ion = 1.0,
    v_a = 0.1,
    n_a_left = 10.0,
    kI = 1.0,
    kR = 0.0,
    γ = 5 / 3,
    ε = ε_paper,
    ν_m0 = 15.0,
    α = α_paper,
    α0 = 10.0,
    ζ = ζ_paper,
    ε_dim = 1.0,
    λ_e_λΣ = λ_e_λΣ,
    c_inv = 1.0,
    H0_func = z -> MagneticField.gaussian_Br(z, H0_B_max, H0_z0, H0_σ),
    v_pic0 = 2.0,
)

println("Case (II) — full EMHD, field averaging (44) clamped, ℓ=20:")
println("  κ=$κ_paper, ξ=$ξ_paper, ζ=$ζ_paper, ε=$ε_paper")

const T_END = 50.0
const FIGDIR = joinpath(ROOT, "output", "figures", "case2_eq44")

CoreSolver.run_simulation(
    params;
    mode = :case2,
    total_time = T_END,
    save_times = [40.0, 45.0, 50.0],
    do_plot = true,
    plot_output_dir = FIGDIR,
    plot_profiles_dimensionless = true,
    steklov_field_half_width = 5,
    steklov_field_passes = 5,
    steklov_field_boundary = :clamped,
)

println("Figures: ", FIGDIR)
