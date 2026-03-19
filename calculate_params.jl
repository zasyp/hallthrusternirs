# compute_dimensionless.jl
using Printf

const e = 1.6e-19
const k_B = 1.38e-23
const ε0 = 8.854e-12
const c = 3e8
const μ0 = 4π * 1e-7

"""
    dimensionless_params(; L_phys, v_char, n_char, H_char, T_char, mi, me, β0, σ0_Spitzer=0.905e7)

Вычисляет безразмерные параметры для гибридной модели СПД.
"""
function dimensionless_params(;
    L_phys::Float64,
    v_char::Float64,
    n_char::Float64,
    H_char::Float64,
    T_char::Float64,
    mi::Float64,
    me::Float64,
    β0::Float64,
    σ0_Spitzer::Float64 = 0.905e7
)

    t_char = L_phys / v_char
    ωci = e * H_char / mi
    ε = t_char * ωci

    ρ = mi * n_char
    vA = H_char / sqrt(μ0 * ρ)
    κ = vA / v_char

    ζ = k_B * T_char / (mi * vA^2)

    ωpe = sqrt(e^2 * n_char / (ε0 * me))
    ξ = c / (L_phys * ωpe)
    α = ξ^2

    λi = mi / e
    λe = me / e
    λz = λi + λe

    α0 = κ * ξ * (λi / λz) * sqrt(λi / λe)
    kI = β0 * n_char * t_char

    # Масштаб силы: импульс / время = m * n * L * v^2
    force = mi * n_char * L_phys * v_char ^ 2

    return (ε=ε, κ=κ, ζ=ζ, ξ=ξ, α=α, α0=α0, kI=kI, t_char=t_char, vA=vA, force=force)
end

"""
    compute_nu_m0(H_char, mi, L_phys, v_char, σ0_Spitzer; calibration=1.0)

Вычисляет коэффициент ν_m0 (магнитная вязкость) из проводимости Спитцера.

ПРЕДУПРЕЖДЕНИЕ: текущая формула содержит неопределённость в масштабировании.
Использует калибровочный коэффициент от экспериментальных данных.

Формула: ν_m0 = calibration * c² / (4π σ0 L_phys v_char)
где σ0_Spitzer - проводимость Спитцера в СГС при T_e = 1 К

Типично: calibration ≈ 0.912 / (полученное значение)
для согласования с экспериментом СПД-70
"""
function compute_nu_m0(H_char, mi, L_phys, v_char, σ0_Spitzer = 0.905e7; calibration=1.0)
    # Формула магнитной вязкости в безразмерном виде
    # ν_m0 = c² / (4π σ0 L_phys v_char)
    # σ0_Spitzer в СГС (см⁻¹·с⁻¹·К^(3/2)·эрг^(-3/2))

    ν_m0 = calibration * (c ^ 2) / (4π * σ0_Spitzer * L_phys * v_char)

    return ν_m0
end

"""
    params_from_physics(; L_phys, v_char, n_char, H_char, T_char, mi, me, β0, σ0_Spitzer,
                          v_a_ion, n_a_left, kR, M=100, N1=100, ε_dim=1.0, H0_func=z->1.0,
                          ν_m0_override=nothing, kI_override=nothing)

Вычисляет полный набор безразмерных параметров для SimParams из физических параметров СПД-70.
Возвращает кортеж (sim_params, force_scale, L_phys, v_char) для преобразования размерностей.
"""
function params_from_physics(;
    L_phys::Float64,
    v_char::Float64,
    n_char::Float64,
    H_char::Float64,
    T_char::Float64,
    mi::Float64,
    me::Float64,
    β0::Float64,
    σ0_Spitzer::Float64 = 0.905e7,
    v_a_ion::Float64 = 0.0408 * 8461.7,
    n_a_left::Float64 = 10.0,
    kR::Float64 = 0.0,
    M::Int = 100,
    N1::Int = 100,
    ε_dim::Float64 = 1.0,
    H0_func::Function = z -> 1.0,
    ν_m0_override::Union{Float64, Nothing} = nothing,
    kI_override::Union{Float64, Nothing} = nothing
)
    dimens = dimensionless_params(;
        L_phys = L_phys,
        v_char = v_char,
        n_char = n_char,
        H_char = H_char,
        T_char = T_char,
        mi = mi,
        me = me,
        β0 = β0,
        σ0_Spitzer = σ0_Spitzer
    )

    if ν_m0_override === nothing
        ν_m0 = compute_nu_m0(H_char, mi, L_phys, v_char, σ0_Spitzer; calibration=1.0)
    else
        ν_m0 = ν_m0_override
    end

    mi_nondim = 1.0
    me_nondim = me / mi
    T_ion_nondim = 1.0
    v_a_nondim = v_a_ion / v_char
    kI = (kI_override !== nothing) ? kI_override : dimens.kI

    sim_params = PartCount.SimParams(
        L = 1.0,
        M = M,
        mi = mi_nondim,
        me = me_nondim,
        T_ion = T_ion_nondim,
        v_a = v_a_nondim,
        n_a_left = n_a_left,
        kI = kI,
        kR = kR,
        γ = 5.0 / 3.0,
        ε = dimens.ε,
        ν_m0 = ν_m0,
        α = dimens.α,
        α0 = dimens.α0,
        ζ = dimens.ζ,
        ε_dim = ε_dim,
        H0_func = H0_func,
        N1 = N1
    )

    force_scale = dimens.force

    return (sim_params, force_scale, L_phys, v_char)
end

# -------------------------------------------------------------------
# Пример расчёта для третьего режима СПД-70 (данные из from_cw.jl)
# -------------------------------------------------------------------
let
    L_phys = 0.04
    v_char = 8461.7
    n_char = 3.2837e17
    H_char = 0.01172
    T_char = 14.0 * 11604.5
    mi = 1.391e-25
    me = 9.11e-31
    β0 = 1e-14
    σ0_Spitzer = 0.905e7

    params_dimens = dimensionless_params(;
        L_phys = L_phys,
        v_char = v_char,
        n_char = n_char,
        H_char = H_char,
        T_char = T_char,
        mi = mi,
        me = me,
        β0 = β0,
        σ0_Spitzer = σ0_Spitzer
    )

    println("РЕЗУЛЬТАТЫ РАСЧЁТА БЕЗРАЗМЕРНЫХ ПАРАМЕТРОВ ДЛЯ СПД-70")
    println("==================================================")
    println("ε  = ", params_dimens.ε)
    println("κ  = ", params_dimens.κ)
    println("ζ  = ", params_dimens.ζ)
    println("α  = ", params_dimens.α)
    println("α0 = ", params_dimens.α0)
    println("kI = ", params_dimens.kI)

    ν_m0_calc = compute_nu_m0(H_char, mi, L_phys, v_char, σ0_Spitzer; calibration=1.0)
    println("\nν_m0 (вычислено, calibration=1.0) = ", ν_m0_calc)
    println("ν_m0 (в коде используется) = 0.912")

    v_a_default = 0.040780141843971635 * v_char
    (sim_params_with_calib, force_scale, _, _) = params_from_physics(;
        L_phys = L_phys,
        v_char = v_char,
        n_char = n_char,
        H_char = H_char,
        T_char = T_char,
        mi = mi,
        me = me,
        β0 = β0,
        σ0_Spitzer = σ0_Spitzer,
        v_a_ion = v_a_default,
        n_a_left = 10.0,
        kR = 0.0,
        ν_m0_override = 0.912,
        kI_override = 0.16
    )

    println("\nПараметры для симуляции:")
    println("L=1.0, M=100")
    @printf("mi=%.1f, me=%.3e\n", sim_params_with_calib.mi, sim_params_with_calib.me)
    @printf("ε=%.3e, ν_m0=%.3f, α=%.3e, α0=%.1f, ζ=%.3f\n",
            sim_params_with_calib.ε, sim_params_with_calib.ν_m0, sim_params_with_calib.α,
            sim_params_with_calib.α0, sim_params_with_calib.ζ)
    @printf("\nМасштаб силы: force_scale = %.3e Н\n", force_scale)
end