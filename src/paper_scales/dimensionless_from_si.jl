# Paper §2 dimensionless groups from SI CharacteristicScalesSI + beta0.

"""
Dimensionless parameters from SI (paper §2): epsilon, xi, zeta, kappa, k_I, nu_{m0}, plus SI helpers.
Uses Spitzer nu_ei -> nu_{m0} unless overridden elsewhere via sim_params_from_si_scales.
"""
function paper_dimensionless_from_si(s::CharacteristicScalesSI, β0_m3s::Float64)
    L = s.L_m
    v = max(s.v_ref_m_s, 1e-300)
    B = abs(s.B_T)
    n = max(s.n_m3, 1e-300)
    T = max(s.T_K, 1e-300)
    m_i = s.m_i_kg
    m_e = s.m_e_kg
    t_char = L / v
    ω_ci = e_C * B / m_i
    ε = t_char * ω_ci
    ρ = m_i * n
    v_A = B / sqrt(μ0_SI * max(ρ, 1e-300))
    κ = v_A / v
    ζ = k_B_JK * T / (m_i * v_A^2)
    ξ = xi_gaussian(L, n, m_i, m_e)
    k_I = β0_m3s * n * t_char
    ν_m0 = nu_m0_dimensionless(s)
    return (; ε, ξ, ζ, κ, k_I, ν_m0, ω_ci_SI = ω_ci, v_A_SI = v_A, t_char_SI = t_char, rho_SI = ρ)
end

function beta0_si_from_target_k_I(k_I::Float64, s::CharacteristicScalesSI)
    t_char = s.L_m / max(s.v_ref_m_s, 1e-300)
    return k_I / max(s.n_m3 * t_char, 1e-300)
end

"""k_I = beta0 [n] [t], [t] = L_m / v_ref."""
function k_I_from_beta0_si(β0_m3s::Float64, s::CharacteristicScalesSI)
    t_char = s.L_m / max(s.v_ref_m_s, 1e-300)
    return β0_m3s * s.n_m3 * t_char
end
