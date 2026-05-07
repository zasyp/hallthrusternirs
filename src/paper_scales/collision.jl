# Collision frequencies: Spitzer, dimensionless nu_m0, omega_ce diagnostics.

"""
    nu_ei_spitzer_hz(n_m3, T_eV; Z=1, lnL=15) -> Float64

Spitzer electron–ion collision frequency `ν_ei = 2.91·10⁻⁶ · n_cc · Z · ln Λ / T_eV^{1.5}` [Hz],
with `n_cc = n_m3 / 10⁶`. Default `lnL = 15` is the canonical value for low-density
laboratory plasmas; override for a specific Coulomb logarithm.
"""
function nu_ei_spitzer_hz(n_m3::Float64, T_eV::Float64; Z::Float64 = 1.0, lnL::Float64 = 15.0)
    n_cc = max(n_m3 * 1e-6, 0.0)
    Te = max(T_eV, 1e-9)
    return 2.91e-6 * n_cc * Z * lnL / Te^1.5
end

"""
    nu_m0_dimensionless(s) -> Float64

Spitzer `ν_ei` evaluated at the reference state of `s` and multiplied by `t_char = L/v_ref`
to produce the dimensionless `ν_m0` consumed by `local_nu_m`.
"""
function nu_m0_dimensionless(s::CharacteristicScalesSI)
    T_eV = k_B_JK * s.T_K / e_C
    ν_ei = nu_ei_spitzer_hz(s.n_m3, T_eV)
    t_char = s.L_m / max(s.v_ref_m_s, 1e-300)
    return ν_ei * t_char
end

"""omega_ce = e|B|/m_e [rad/s]. Use floatmin(me), not eps(), since m_e << eps(Float64)."""
function omega_ce_electron_rad_s(B_T::Float64, m_e_kg::Float64)
    return abs(e_C * B_T / max(m_e_kg, floatmin(Float64)))
end

"""nu_e [Hz] from beta_e = omega_ce/nu_e (inverse of solver Hall diagnostic)."""
function nu_e_hz_from_beta_e(B_T::Float64, m_e_kg::Float64, beta_e::Float64)
    return omega_ce_electron_rad_s(B_T, m_e_kg) / max(beta_e, eps())
end

"""Dimensionless nu_{m0} = nu_e * [t], [t] = L/v, for :constant collision model."""
function nu_m0_dimless_from_nu_e_hz(ν_e_hz::Float64, s::CharacteristicScalesSI)
    t_char = s.L_m / max(s.v_ref_m_s, 1e-300)
    return ν_e_hz * t_char
end
