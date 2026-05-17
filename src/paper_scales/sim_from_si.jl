# Build SimParams and CharacteristicScalesSI from SI (paper pipeline).

"""
Build SimParams at dimensionless channel length L=1 from SI scales and paper closures.
Passes through epsilon, zeta, k_I, closure coefficients alpha/alpha0; nu_m0 from Spitzer
or optional fixed nu_e_total_hz [Hz].
"""
function sim_params_from_si_scales(
    s::CharacteristicScalesSI,
    β0_m3s::Float64;
    M::Int,
    N1::Int,
    H0_func,
    v_a_dimless::Float64 = 0.1,
    n_a_left::Float64 = 10.0,
    kR::Float64 = 0.05,
    γ::Float64 = 5 / 3,
    ε_dim::Float64 = 1.0,
    c_inv::Float64 = 2.15e-4,
    T_ion::Float64 = 1.0,
    ν_e_total_hz::Union{Nothing, Float64} = nothing,
    v_pic0::Float64 = 1.0,
    collision_model::Symbol = :spitzer,
    alpha_B::Float64 = 0.0,
    E_z0_dimless::Float64 = 0.0,
    pic_charge_factor::Float64 = 1.0,
    include_self_B::Bool = true,
)
    g = paper_dimensionless_from_si(s, β0_m3s)
    L_sim = 1.0
    me_ratio = s.m_e_kg / s.m_i_kg
    λ_e_λΣ = s.m_e_kg / (s.m_i_kg + s.m_e_kg)
    λ_i_λΣ = s.m_i_kg / (s.m_i_kg + s.m_e_kg)
    α_ohm = λ_i_λΣ * g.ξ^2
    λi_λe = s.m_i_kg / s.m_e_kg
    α0_paper = g.κ * g.ξ * λ_i_λΣ * sqrt(λi_λe)
    ν_m0 = ν_e_total_hz === nothing ? g.ν_m0 : nu_m0_dimless_from_nu_e_hz(ν_e_total_hz, s)
    return PartCount.SimParams(;
        L = L_sim,
        M = M,
        mi = 1.0,
        me = me_ratio,
        T_ion = T_ion,
        v_a = v_a_dimless,
        n_a_left = n_a_left,
        kI = g.k_I,
        kR = kR,
        γ = γ,
        ε = g.ε,
        ν_m0 = ν_m0,
        α = α_ohm,
        α0 = α0_paper,
        ζ = g.ζ,
        ε_dim = ε_dim,
        λ_e_λΣ = λ_e_λΣ,
        c_inv = c_inv,
        H0_func = H0_func,
        N1 = N1,
        v_pic0 = v_pic0,
        collision_model = collision_model,
        alpha_B = alpha_B,
        E_z0_dimless = E_z0_dimless,
        pic_charge_factor = pic_charge_factor,
        include_self_B = include_self_B,
    ), g
end

"""Alfvén reference with [v]=v_A (kappa = 1 when used as velocity scale); T_K from T_e [eV]."""
function alfven_reference_scales(;
    L_m::Float64,
    B_T::Float64,
    n_m3::Float64,
    T_e_eV::Float64 = 14.0,
    m_i_kg::Float64 = 2.18e-25,
    m_e_kg::Float64 = 9.1093837015e-31,
)
    B = abs(B_T)
    n = max(n_m3, 1e-300)
    ρ = m_i_kg * n
    v_A = B / sqrt(μ0_SI * max(ρ, 1e-300))
    T_K = max(T_e_eV, 1e-12) * e_C / k_B_JK
    return CharacteristicScalesSI(L_m, v_A, B_T, n_m3, T_K, m_i_kg, m_e_kg)
end
