# Paper sec. 4 Case (II): figs. 8, 9 — field smoothing via (43), ℓ=20, five passes (p.44);
# figs. 10, 11 — modified smoothing (44); see `case2_average_eq44.jl`.
# Full hybrid EMHD: E_y from elliptic (Ohm), H_x_ind via Faraday, E_z from (38); j_y(0)=j_y(L)=0.
#
# Similarity (p.43): k_I=1, v_a=0.1, n_a=10, κ=1, ξ=0.532, ε=1, ζ=0.061, H_0=1
# Grid: M=100, N_1=100; ν_m0 = 15 (p.17, Spitzer).

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "HallThrusterNIRS.jl"))
using .HallThrusterNIRS

# Xe: m_i ≈ 54·m_p.
const MI_AMU = 54.0
const MP_ME  = 1836.152672
const ME_OVER_MI = 1.0 / (MI_AMU * MP_ME)
const λ_i_λΣ = 1.0 / (1.0 + ME_OVER_MI)
const λ_e_λΣ = ME_OVER_MI / (1.0 + ME_OVER_MI)

const κ_paper  = 1.0
const ξ_paper  = 0.532
const ζ_paper  = 0.061
const ε_paper  = 1.0
# H_0(z): Gaussian radial field (sec. 1, 4), peak near channel exit slice.
const H0_B_max = 1.0
const H0_z0    = 0.75
const H0_σ     = 0.20

# Closures (33), (38): α = (λ_i/λ_Σ)·ξ², α0 = κ·ξ·(λ_i/λ_Σ)·√(λ_i/λ_e).
const λi_λe    = (1.0 + ME_OVER_MI) / ME_OVER_MI
const α_paper  = λ_i_λΣ * ξ_paper^2
const α0_paper = κ_paper * ξ_paper * λ_i_λΣ * sqrt(λi_λe)

params = PartCount.SimParams(;
    L = 1.0,
    M = 100,
    N1 = 100,
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
    α0 = α0_paper,
    ζ = ζ_paper,
    ε_dim = 1.0,
    λ_e_λΣ = λ_e_λΣ,
    c_inv = 1.0,
    H0_func = z -> MagneticField.gaussian_Br(z, H0_B_max, H0_z0, H0_σ),
    v_pic0 = 58.0,              # (30), p.32
)

println("Case (II) — full EMHD:")
println("  κ=$κ_paper, ξ=$ξ_paper, ζ=$ζ_paper, ε=$ε_paper")
println("  H_0(z) = gaussian(B_max=$H0_B_max, z0=$H0_z0, σ=$H0_σ), L_B ≈ $(round(MagneticField.estimate_LB(H0_σ; model=:gaussian), sigdigits=3))")
println("  α  = (λ_i/λ_Σ)·ξ²                   = $(round(α_paper, sigdigits=5))")
println("  α0 = κ·ξ·(λ_i/λ_Σ)·√(λ_i/λ_e) = $(round(α0_paper, sigdigits=5))")
println("  λ_i/λ_Σ = $(round(λ_i_λΣ, sigdigits=6)), λ_e/λ_Σ = $(round(λ_e_λΣ, sigdigits=6))")

const T_END = 50.0
const FIGDIR = joinpath(ROOT, "output", "figures", "case2")

CoreSolver.run_simulation(
    params;
    mode = :case2,
    total_time = T_END,
    save_times = [10.0, 20.0, 30.0, 40.0, 50.0],
    do_plot = true,
    plot_output_dir = FIGDIR,
    plot_profiles_dimensionless = true,
    steklov_field_half_width = 20,
    steklov_field_passes = 5,
    steklov_field_boundary = :reflect,
)

println("Figures: ", FIGDIR)
