const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "HallThrusterNIRS.jl"))
using .HallThrusterNIRS

const M_KR_KG = 1.39e-25
const M_XE_KG = 2.18e-25
const M_E_KG  = 9.1093837015e-31

const L_M   = 0.010
const B_MAX = 300.0e-4
const N_M3  = 5.0e17
const T_E_EV = 14.0
const BETA0_M3S = 8.0e-13
const T_ION_EV = 500.0 / 11600.0

const H0_Z0    = 0.8
const H0_SIGMA = 0.30
const H0_OFFSET = 0.2

scales_SI = PaperScales.alfven_reference_scales(;
    L_m    = L_M,
    B_T    = B_MAX,
    n_m3   = N_M3,
    T_e_eV = T_E_EV,
    m_i_kg = M_KR_KG,
    m_e_kg = M_E_KG,
)
scales_Xe = PaperScales.alfven_reference_scales(;
    L_m    = L_M,
    B_T    = B_MAX,
    n_m3   = N_M3,
    T_e_eV = T_E_EV,
    m_i_kg = M_XE_KG,
    m_e_kg = M_E_KG,
)

groups_Xe = PaperScales.paper_dimensionless_from_si(scales_Xe, BETA0_M3S)
k_I_from_beta = PaperScales.k_I_from_beta0_si(BETA0_M3S, scales_SI)

H0_func = z -> H0_OFFSET + MagneticField.gaussian_Br(z, 1.0, H0_Z0, H0_SIGMA)

const U_DISCHARGE_V = 250.0
"""Ballistic ceiling sqrt(2 e U_discharge / m_i); compare to ion v_max in channel (no sheath/split-voltage losses)."""
const V_ION_BEAM_MS = sqrt(2 * PaperScales.e_C * U_DISCHARGE_V / M_KR_KG)
const HEXT_DIM_MAX = maximum(abs.(H0_func.(collect(range(0.0, 1.0; length = 400)))))
const E0_DIMLESS = PaperScales.E0_dimless_from_discharge_voltage(
    U_DISCHARGE_V, scales_SI, HEXT_DIM_MAX; L_m = L_M,
)

v_pic0_dimless = PaperScales.v_pic0_dimless_from_ion_thermal(scales_SI, T_ION_EV)

const ALPHA_B_ANOM = 0.0
const N_A_LEFT_DIMLESS = 10.0

# Grid / PIC sizing for SPT-70 / Krypton at the SI scales above.
# - h_SI = L/M = 0.4 mm ≈ 8.5·λ_D (λ_D ≈ 47 μm at n=5e17, T_e=14 eV) — adequate for hybrid PIC.
# - N1 = 50 initial macroparticles per cell (5000 total): ionization grows the count by 10-50×
#   over one transit time, so additional initial seeding is wasted; moments are dominated by
#   ionized particles after t ≈ 1 t_char.
const M_GRID = 30
const N1_MAC = 50

params, groups = PaperScales.sim_params_from_si_scales(
    scales_SI,
    BETA0_M3S;
    M = M_GRID,
    N1 = N1_MAC,
    H0_func = H0_func,
    v_a_dimless = 0.01,
    n_a_left    = N_A_LEFT_DIMLESS,
    kR          = 0.01,
    v_pic0      = v_pic0_dimless,
    collision_model = :spitzer,
    alpha_B         = ALPHA_B_ANOM,
)

println("SPT (Krypton) — SI → dimensionless (paper §2 pipeline):")
println("  β₀ (SI) = $(round(BETA0_M3S, sigdigits=4)) m³/s; k_I = β₀ n L/v_A = $(round(k_I_from_beta, sigdigits=5)) (matches `groups.k_I` = $(round(groups.k_I, sigdigits=5)))")
println("  SI: L=$(scales_SI.L_m) m, v_A=$(round(scales_SI.v_ref_m_s, sigdigits=5)) m/s, ",
        "B=$(round(scales_SI.B_T*1e3, sigdigits=4)) mT, n=$(scales_SI.n_m3) m⁻³, T_e=$T_E_EV eV")
println("  Same β₀ with m_i=Xe — k_I = $(round(groups_Xe.k_I, sigdigits=4)) (differs from Kr: k_I ∝ L/v_A, v_A ∝ 1/√m_i), ε = $(round(groups_Xe.ε, sigdigits=4)), ν_m0 = $(round(groups_Xe.ν_m0, sigdigits=4))  |  Kr: ε, ν_m0 — see above")
ω_ce_exit = PaperScales.omega_ce_electron_rad_s(B_MAX, M_E_KG)
ν_e_spitzer = PaperScales.nu_ei_spitzer_hz(Float64(N_M3), Float64(T_E_EV))
β_e_spitzer = ω_ce_exit / max(ν_e_spitzer, eps())
println("  ω_ce(|B|)=$(round(ω_ce_exit, sigdigits=4)) rad/s; ν_ei(Spitzer, n,T)=",
        "$(round(ν_e_spitzer, sigdigits=4)) Hz  ⇒  β_e=ω_ce/ν_ei = $(round(β_e_spitzer, sigdigits=4))")
println("  ν_m(z) = ν_m0/T_e^{3/2} (local), ν_m0 = $(round(params.ν_m0, sigdigits=5)) (sim)")
println("  v_pic0 = v_th,ion/[v] = $(round(v_pic0_dimless, sigdigits=5))  (T_ion = $T_ION_EV eV)")
println("  ε, κ, ζ, k_I (Kr) = $(round(groups.ε, sigdigits=4)), $(round(groups.κ, sigdigits=4)), $(round(groups.ζ, sigdigits=4)), $(round(groups.k_I, sigdigits=4))")
println("  α_B = $ALPHA_B_ANOM, α0 = $(round(params.α0, sigdigits=4))  |  n_a(0) = $N_A_LEFT_DIMLESS")
println(
    "  Cold-ion reference (full U_discharge): sqrt(2 e U_discharge / m_i) = ",
    "$(round(V_ION_BEAM_MS, digits=5)) m/s — compare to `v_iz` profile (sheath / partial Δφ not resolved).",
)

# Total simulation time in dimensionless units `[t] = L/v_A`.
# - Ion transit time at U_discharge=250 V: t_transit ≈ L / sqrt(2eU/m_i) ≈ 1.7 μs ≈ 4.2 t_char.
# - Steady state typically reached after 5-8 transit times → T_END ≈ 30 is sufficient and ~3× cheaper
#   than T_END = 100 (no further physics is captured beyond the steady state).
const T_END  = 3.0
const FIGDIR = joinpath(ROOT, "output", "figures", "spt70_krypton")

CoreSolver.run_simulation(
    params;
    mode = :case2,
    accumulate_induced_H = false,
    total_time = T_END,
    # Snapshots from the last third of the run to inspect statistical steady state.
    save_times = [T_END*0.85, T_END*0.9, T_END],
    do_plot = true,
    plot_output_dir = FIGDIR,
    si_plot_scales = scales_SI,
    plot_profiles_dimensionless = true,
    E0_dimless = E0_DIMLESS,
    # Adaptive timestep: keep the four physical CFLs; relax the explicit Hall-whistler safety to 2.0
    # because the elliptic Ohm solve already treats the Hall part of the current semi-implicitly.
    tau_constraints = :full,
    tau_hall_safety = 2.0,
    log_tau_constraint = true,
)

println("Figures: ", FIGDIR)
