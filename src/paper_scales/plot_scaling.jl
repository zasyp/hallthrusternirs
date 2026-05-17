# Dimensional factors for plotting, thrust normalization, Ez0 discharge shift, PIC v_pic0 scale.

"""Plot scales (SI). Caveat: solver variable `H_*` is the dimensionless **magnetic induction** (Alfvén-normalized,
`B_t = |B|/max(H_dimless)`); the `H` naming is historical. Plasma current scale uses `j₀ = e n v` so that the
dimensionless Ohm law and the Faraday-derived `j = ∂_z H` are consistent **iff** `L = c/ω_pi` (ion skin depth);
otherwise the `j` plotted in A/m^2 is `e n v ⋅ j_dimless` and a residual `B/(μ0 L) / (e n v)` factor measures
the model's Alfvén/skin-depth mismatch."""
function si_plot_physical_scales(s::CharacteristicScalesSI, Hext_dim_max::Float64;
    effective_area_m2::Float64 = THRUSTER_EFFECTIVE_AREA_M2)
    hm = max(Hext_dim_max, 1e-12)
    B_t = abs(s.B_T) / hm
    z_m = s.L_m
    v_ms = s.v_ref_m_s
    n_m3 = s.n_m3
    Te_eV = k_B_JK * s.T_K / e_C
    j_am2 = e_C * n_m3 * v_ms
    E_vm = v_ms * B_t
    nu_s = v_ms / z_m
    t_s = z_m / v_ms
    thrust_n = s.m_i_kg * effective_area_m2 * n_m3 * v_ms^2
    return (; z_m, n_m3, v_ms, t_s, Te_eV, E_vm, j_am2, B_t, nu_s, thrust_n)
end

"""
    thrust_momentum_SI(s, effective_area_m2=THRUSTER_EFFECTIVE_AREA_M2) -> Float64

Thrust scale `F₀ = m_i · n · A · v_ref²` [N]. Used to convert the dimensionless
momentum-flux diagnostic emitted by `move_particles` into Newtons.
"""
function thrust_momentum_SI(s::CharacteristicScalesSI, effective_area_m2::Float64 = THRUSTER_EFFECTIVE_AREA_M2)
    return s.m_i_kg * s.n_m3 * effective_area_m2 * (s.v_ref_m_s)^2
end

"""
    E0_dimless_from_discharge_voltage(U_volts, s, Hext_dim_max; L_m=s.L_m) -> Float64

Convert a discharge voltage `U` into a dimensionless uniform `E_z` bias using the same
plot scaling as the diagnostics (`E_vm = v_ms · B_t`). Used to set
`SimParams.E_z0_dimless` consistently with the experiment.
"""
function E0_dimless_from_discharge_voltage(U_volts::Float64, s::CharacteristicScalesSI, Hext_dim_max::Float64; L_m::Float64 = s.L_m)
    p = si_plot_physical_scales(s, Hext_dim_max)
    return (U_volts / max(L_m, eps())) / max(p.E_vm, eps())
end

"""
    v_pic0_dimless_from_ion_thermal(s, T_ion_eV) -> Float64

Initial PIC velocity scale: ion thermal speed `v_th = √(k_B T / m_i)` divided by the
reference velocity. Used as `SimParams.v_pic0` when initialising `Particle.vy/vz` from a
Maxwellian.
"""
function v_pic0_dimless_from_ion_thermal(s::CharacteristicScalesSI, T_ion_eV::Float64)
    T_K = max(T_ion_eV, 1e-18) * e_C / k_B_JK
    v_th = sqrt(k_B_JK * T_K / max(s.m_i_kg, 1e-300))
    return v_th / max(s.v_ref_m_s, 1e-300)
end
