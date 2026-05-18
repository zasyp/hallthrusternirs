const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "HallThrusterNIRS.jl"))
using .HallThrusterNIRS

const M_KR_KG = 1.39e-25
const M_E_KG  = 9.1093837015e-31

const L_M   = 0.05
const B_MAX = 200.0e-4

const N_M3  = 5.0e17
const T_E_EV = 5.0
const T_ION_EV = 5.0
const U_DISCHARGE_V = 200.0
const BETA0_M3S = 8.0e-14

const H0_Z0    = 0.7
const H0_SIGMA = 0.30
const H0_OFFSET = 0.1

v_ion_ms = 12500.0 
scales_SI = PaperScales.CharacteristicScalesSI(;
    L_m    = L_M,
    v_ref_m_s = v_ion_ms,
    B_T    = B_MAX,
    n_m3   = N_M3,
    T_K    = T_E_EV * PaperScales.e_C / PaperScales.k_B_JK,
    m_i_kg = M_KR_KG,
    m_e_kg = M_E_KG,
)
k_I_from_beta = PaperScales.k_I_from_beta0_si(BETA0_M3S, scales_SI)

H0_func = z -> H0_OFFSET + MagneticField.gaussian_Br(z, 1.0, H0_Z0, H0_SIGMA)

"""Ballistic ceiling sqrt(2 e U_discharge / m_i); compare to ion v_max in channel (no sheath/split-voltage losses)."""
const HEXT_DIM_MAX = maximum(abs.(H0_func.(collect(range(0.0, 1.0; length = 400)))))
const E0_DIMLESS = PaperScales.E0_dimless_from_discharge_voltage(
    U_DISCHARGE_V, scales_SI, HEXT_DIM_MAX; L_m = L_M,
)

v_pic0_dimless = PaperScales.v_pic0_dimless_from_ion_thermal(scales_SI, T_ION_EV)

const ALPHA_B_ANOM = 0.002
const N_A_LEFT_DIMLESS = 66.0

const M_GRID = 30
const N1_MAC = 30

params, groups = PaperScales.sim_params_from_si_scales(
    scales_SI,
    BETA0_M3S;
    M = M_GRID,
    N1 = N1_MAC,
    H0_func = H0_func,
    v_a_dimless = 0.1,
    n_a_left    = N_A_LEFT_DIMLESS,
    kR          = 0.00,
    v_pic0      = v_pic0_dimless,
    collision_model = :spitzer,
    alpha_B         = ALPHA_B_ANOM,
    c_inv = 1.0,          
)
mutable_params = PartCount.SimParams(
    L = params.L,
    M = params.M,
    mi = params.mi,
    me = params.me,
    T_ion = params.T_ion,
    v_a = params.v_a,
    n_a_left = params.n_a_left,
    kI = params.kI,
    kR = params.kR,
    γ = params.γ,
    ε = params.ε,
    ν_m0 = params.ν_m0,
    α = params.α,
    α0 = 12.0,
    ζ = params.ζ,
    ε_dim = params.ε_dim,
    λ_e_λΣ = params.λ_e_λΣ,
    H0_func = params.H0_func,
    N1 = params.N1,
    v_pic0 = params.v_pic0,
    collision_model = params.collision_model,
    alpha_B = params.alpha_B,
    E_z0_dimless = params.E_z0_dimless,
    pic_charge_factor = params.pic_charge_factor,
    include_self_B = params.include_self_B,
)
println("SPT (Krypton) — SI → dimensionless (paper §2 pipeline):")
println("  β₀ (SI) = $(round(BETA0_M3S, sigdigits=4)) m³/s; k_I = β₀ n L/v_A = $(round(k_I_from_beta, sigdigits=5)) (matches `groups.k_I` = $(round(groups.k_I, sigdigits=5)))")
println("  SI: L=$(scales_SI.L_m) m, v_A=$(round(scales_SI.v_ref_m_s, sigdigits=5)) m/s, ",
        "B=$(round(scales_SI.B_T*1e3, sigdigits=4)) mT, n=$(scales_SI.n_m3) m⁻³, T_e=$T_E_EV eV")
# println("  Same β₀ with m_i=Xe — k_I = $(round(groups_Xe.k_I, sigdigits=4)) (differs from Kr: k_I ∝ L/v_A, v_A ∝ 1/√m_i), ε = $(round(groups_Xe.ε, sigdigits=4)), ν_m0 = $(round(groups_Xe.ν_m0, sigdigits=4))  |  Kr: ε, ν_m0 — see above")
ω_ce_exit = PaperScales.omega_ce_electron_rad_s(B_MAX, M_E_KG)
ν_e_spitzer = PaperScales.nu_ei_spitzer_hz(Float64(N_M3), Float64(T_E_EV))
println("  ω_ce(|B|)=$(round(ω_ce_exit, sigdigits=4)) rad/s; ν_ei(Spitzer, n,T)=",
        "$(round(ν_e_spitzer, sigdigits=4))")
println("  ν_m(z) = ν_m0/T_e^{3/2} (local), ν_m0 = $(round(params.ν_m0, sigdigits=5)) (sim)")
println("  v_pic0 = v_th,ion/[v] = $(round(v_pic0_dimless, sigdigits=5))  (T_ion = $T_ION_EV eV)")
println("  ε, κ, ζ, k_I (Kr) = $(round(groups.ε, sigdigits=4)), $(round(groups.κ, sigdigits=4)), $(round(groups.ζ, sigdigits=4)), $(round(groups.k_I, sigdigits=4))")
println("  α_B = $ALPHA_B_ANOM, α0 = $(round(params.α0, sigdigits=4))  |  n_a(0) = $N_A_LEFT_DIMLESS")
println(
    "  Cold-ion reference (full Uжд_discharge): sqrt(2 e U_discharge / m_i) = ",
    "$(round(v_ion_ms, digits=5)) m/s — compare to `v_iz` profile (sheath / partial Δφ not resolved).",
)

const T_END  = 50.0
const FIGDIR = joinpath(ROOT, "output", "figures", "spt70_krypton")

CoreSolver.run_simulation(
    mutable_params;
    mode = :case2,
    accumulate_induced_H = true,
    total_time = T_END,
    save_times = [T_END*0.9, T_END*0.95, T_END],
    do_plot = true,
    plot_output_dir = FIGDIR,
    si_plot_scales = scales_SI,
    plot_profiles_dimensionless = true,
    E0_dimless = E0_DIMLESS,
    steklov_field_half_width = 2,
    steklov_field_passes = 2,
    steklov_field_boundary = :reflect,
)

println("Figures: ", FIGDIR)
