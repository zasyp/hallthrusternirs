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

Paper §2 dimensionless magnetic viscosity prefactor

    ν_m0 = c² / (4π σ_0 [T]^{3/2} [L][v]),

with Spitzer `σ = σ_0 · T^{3/2}`, `[L] = L_m`, `[v] = v_ref`, `[T] = T_K`. Used by
`local_nu_m` as `ν_m(z,t) = ν_m0 / T_e^{3/2}` (paper Eq. for the magnetic-diffusion /
resistive-Ohm closure).

Identity used here (Gaussian or SI gives the same dimensionless value):

    c²/(4π σ) = (c/ω_pe)² · ν_ei,
    ξ          = (c/ω_pe) / [L]                       (electron skin depth / channel length),

so

    ν_m0 = ξ² · ν_ei([T]) · t_char.

`ν_ei([T])` is `nu_ei_spitzer_hz` at the reference state, `t_char = L_m/v_ref`.
This differs from a naive `ν_ei · t_char` by the factor `ξ²` that converts a collision
frequency into the magnetic diffusivity used in the paper closures.
"""
function nu_m0_dimensionless(s::CharacteristicScalesSI)
    T_eV = k_B_JK * s.T_K / e_C
    ν_ei = nu_ei_spitzer_hz(s.n_m3, T_eV)
    t_char = s.L_m / max(s.v_ref_m_s, 1e-300)
    ξ = xi_gaussian(s.L_m, s.n_m3, s.m_i_kg, s.m_e_kg)
    return ν_ei * t_char * ξ^2
end

"""omega_ce = e|B|/m_e [rad/s]. Use floatmin(me), not eps(), since m_e << eps(Float64)."""
function omega_ce_electron_rad_s(B_T::Float64, m_e_kg::Float64)
    return abs(e_C * B_T / max(m_e_kg, floatmin(Float64)))
end

"""nu_e [Hz] from beta_e = omega_ce/nu_e (inverse of solver Hall diagnostic)."""
function nu_e_hz_from_beta_e(B_T::Float64, m_e_kg::Float64, beta_e::Float64)
    return omega_ce_electron_rad_s(B_T, m_e_kg) / max(beta_e, eps())
end

"""
Dimensionless `ν_m0` from a user-supplied effective electron collision frequency
`ν_e` [Hz] (used by the `:constant` collision model and to bake in additional channels
such as electron–neutral or anomalous Bohm).

Paper §2 magnetic-viscosity convention (see `nu_m0_dimensionless`):

    ν_m0 = ξ² · ν_e · t_char,    ξ = (c/ω_pe) / [L],   t_char = L/v_ref.
"""
function nu_m0_dimless_from_nu_e_hz(ν_e_hz::Float64, s::CharacteristicScalesSI)
    t_char = s.L_m / max(s.v_ref_m_s, 1e-300)
    ξ = xi_gaussian(s.L_m, s.n_m3, s.m_i_kg, s.m_e_kg)
    return ν_e_hz * t_char * ξ^2
end
