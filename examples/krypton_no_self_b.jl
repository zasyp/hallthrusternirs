const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "HallThrusterNIRS.jl"))
using .HallThrusterNIRS

const L_M = 0.010
const B_T = 0.012
const N_M3 = 5.0e17
const T_E_EV = 14.0
const M_I_KG = 1.39e-25
const M_E_KG = 9.1093837015e-31
const K_I_TARGET = 1.0

scales = PaperScales.alfven_reference_scales(;
    L_m = L_M, B_T = B_T, n_m3 = N_M3, T_e_eV = T_E_EV, m_i_kg = M_I_KG, m_e_kg = M_E_KG)
β0 = PaperScales.beta0_si_from_target_k_I(K_I_TARGET, scales)
g = PaperScales.paper_dimensionless_from_si(scales, β0)
me_ratio = scales.m_e_kg / scales.m_i_kg
λ_e_λΣ = scales.m_e_kg / (scales.m_i_kg + scales.m_e_kg)
λ_i_λΣ = scales.m_i_kg / (scales.m_i_kg + scales.m_e_kg)
λi_λe = scales.m_i_kg / scales.m_e_kg
α_ohm = λ_i_λΣ * g.ξ^2
α0_paper = g.κ * g.ξ * λ_i_λΣ * sqrt(λi_λe)
H0_func = z -> MagneticField.gaussian_Br(z, 1.0, 0.75, 0.20)

params = PartCount.SimParams(;
    L = 1.0, M = 80, mi = 1.0, me = me_ratio, T_ion = 1.0, v_a = 0.1, n_a_left = 10.0,
    kI = g.k_I, kR = 0.05, γ = 5 / 3, ε = g.ε, ν_m0 = g.ν_m0, α = α_ohm, α0 = α0_paper, ζ = g.ζ,
    ε_dim = 1.0, λ_e_λΣ = λ_e_λΣ, c_inv = 1.0, H0_func = H0_func, N1 = 80,
    v_pic0 = PaperScales.v_pic0_dimless_from_ion_thermal(scales, T_E_EV / 100),
    collision_model = :spitzer, alpha_B = 0.0, E_z0_dimless = 0.0, pic_charge_factor = 1.0,
    include_self_B = false)

CoreSolver.run_simulation(
    params;
    total_time = 25.0,
    save_times = [10.0, 20.0, 25.0],
    do_plot = true,
    plot_output_dir = joinpath(ROOT, "output", "figures", "krypton_no_self_b"),
    si_plot_scales = scales,
    plot_profiles_dimensionless = true,
)
