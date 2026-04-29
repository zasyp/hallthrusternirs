# Dimensional factors for plotting, thrust normalization, Ez0 discharge shift, PIC v_pic0 scale.

function si_plot_physical_scales(s::CharacteristicScalesSI, Hext_dim_max::Float64;
    effective_area_m2::Float64 = THRUSTER_EFFECTIVE_AREA_M2)
    hm = max(Hext_dim_max, 1e-12)
    B_t = abs(s.B_T) / hm
    z_m = s.L_m
    v_ms = s.v_ref_m_s
    n_m3 = s.n_m3
    Te_eV = k_B_JK * s.T_K / e_C
    j_am2 = B_t / (μ0_SI * z_m)
    E_vm = v_ms * B_t
    nu_s = v_ms / z_m
    t_s = z_m / v_ms
    thrust_n = s.m_i_kg * effective_area_m2 * n_m3 * v_ms^2
    return (; z_m, n_m3, v_ms, t_s, Te_eV, E_vm, j_am2, B_t, nu_s, thrust_n)
end

function thrust_momentum_SI(s::CharacteristicScalesSI, effective_area_m2::Float64 = THRUSTER_EFFECTIVE_AREA_M2)
    return s.m_i_kg * s.n_m3 * effective_area_m2 * s.v_ref_m_s^2
end

function E0_dimless_from_discharge_voltage(U_volts::Float64, s::CharacteristicScalesSI, Hext_dim_max::Float64; L_m::Float64 = s.L_m)
    p = si_plot_physical_scales(s, Hext_dim_max)
    return (U_volts / max(L_m, eps())) / max(p.E_vm, eps())
end

function v_pic0_dimless_from_ion_thermal(s::CharacteristicScalesSI, T_ion_eV::Float64)
    T_K = max(T_ion_eV, 1e-18) * e_C / k_B_JK
    v_th = sqrt(k_B_JK * T_K / max(s.m_i_kg, 1e-300))
    return v_th / max(s.v_ref_m_s, 1e-300)
end
