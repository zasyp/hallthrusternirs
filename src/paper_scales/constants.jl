# SI / CGS constants and CharacteristicScalesSI (PaperScales internals).

const e_C = 1.602176634e-19
const k_B_JK = 1.380649e-23
const μ0_SI = 4π * 1e-7

"""Thruster channel effective cross-section area [m^2]; momentum thrust scaling for plots."""
const THRUSTER_EFFECTIVE_AREA_M2 = 0.0025

# Gaussian-xi: e in statC, c in cm/s, rho in g/cm^3, L in cm.
const e_stat = 4.80320425e-10
const c_cgs = 2.99792458e10

struct CharacteristicScalesSI
    L_m::Float64
    v_ref_m_s::Float64
    B_T::Float64
    n_m3::Float64
    T_K::Float64
    m_i_kg::Float64
    m_e_kg::Float64
end

function CharacteristicScalesSI(; L_m, v_ref_m_s, B_T, n_m3, T_K, m_i_kg, m_e_kg)
    return CharacteristicScalesSI(
        Float64(L_m), Float64(v_ref_m_s), Float64(B_T), Float64(n_m3), Float64(T_K),
        Float64(m_i_kg), Float64(m_e_kg),
    )
end

function xi_gaussian(L_m::Float64, n_m3::Float64, m_i_kg::Float64, m_e_kg::Float64)
    L_cm = L_m * 100.0
    n_cc = n_m3 * 1e-6
    m_i_g = m_i_kg * 1e3
    m_e_g = m_e_kg * 1e3
    rho = m_i_g * n_cc
    λ_i = m_i_g / e_stat
    λ_e = m_e_g / e_stat
    return c_cgs * sqrt(λ_i * λ_e) / (L_cm * sqrt(4 * pi * max(rho, 1e-100)))
end
