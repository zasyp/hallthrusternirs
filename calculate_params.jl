#= 
    calculate_params.jl
    Вычисление безразмерных параметров для гибридной модели СПД
    в соответствии с препринтом ИПМ №35 за 2021 г. (Гавриков, Таюрский)
    Используются масштабы: [L]=1 см, [n]=5e11 см⁻³, [T]=12.1 эВ, [H]=200 Гс.
    Результат выводится в формате, готовом для вставки в main.jl.
=#

using Printf
using DelimitedFiles

# Масштабы из статьи
const L_SCALE_CM = 1.0
const N_SCALE_CM3 = 5.0e11
const T_SCALE_EV = 12.1
const H_SCALE_GS = 200.0

"""
    compute_dimensionless_params(; ...)

Вычисляет безразмерные параметры для заданных физических характеристик.
Возвращает словарь и выводит код для вставки в let-блок main.jl.
"""
function compute_dimensionless_params(;
    L_m::Float64,               # длина канала, м
    v_a_m::Float64,              # скорость нейтралов на входе, м/с
    n_a_m3::Float64,             # концентрация нейтралов на входе, м⁻³
    H_max_T::Float64,            # максимальное магнитное поле, Тл
    T_eV::Float64,               # характерная температура электронов, эВ
    gas::String,                 # "Xe", "Kr", "Ar"
    β0_m3_s::Float64,             # коэффициент ионизации, м³/с
    ν_m0_guess::Float64 = 15.0,   # калибровочный параметр магнитной вязкости
    M::Int = 100,                 # число ячеек сетки
    N1::Int = 100,                # число макрочастиц на ячейку
    kR::Float64 = 0.1,            # коэффициент рекомбинации
    γ::Float64 = 5/3,             # показатель адиабаты
    ε_dim::Float64 = 1.0,         # масштабный множитель для поля (обычно 1)
    H0_ratio::Float64 = 1.0,      # отношение внешнего поля к H_scale (H0/H_scale)
    output_file::Union{String,Nothing} = nothing   # имя CSV-файла для сохранения
)
    # Физические константы в СГС
    c_cgs = 2.99792458e10        # см/с
    e_cgs = 4.8032044e-10        # ед. СГСЭ (статкулон)
    k_B_cgs = 1.380649e-16       # эрг/К
    m_e_g = 9.1093837e-28        # г
    π = 3.141592653589793

    # Массы ионов в граммах
    gas_masses_g = Dict(
        "Xe" => 131.293 * 1.66053906660e-24,
        "Kr" => 83.798 * 1.66053906660e-24,
        "Ar" => 39.948 * 1.66053906660e-24
    )
    if !haskey(gas_masses_g, gas)
        error("Неизвестный газ: $gas. Допустимые: Xe, Kr, Ar")
    end
    m_i_g = gas_masses_g[gas]

    # Перевод входных параметров в СГС
    L_cm = L_m * 100
    v_a_cms = v_a_m * 100
    n_a_cm3 = n_a_m3 * 1e-6
    H_max_Gs = H_max_T * 1e4
    T_K = T_eV * 11604.5          # Кельвины

    # Масштабы
    t_scale_s = L_cm / v_a_cms     # с (характерное время)
    ρ_scale_gcm3 = m_i_g * n_a_cm3 # г/см³ (плотность)

    # Альфвеновская скорость (см/с)
    v_A_cms = H_max_Gs / sqrt(4π * ρ_scale_gcm3)

    # Параметр инерции электронов ξ
    ξ = c_cgs * sqrt(m_i_g * m_e_g) / (e_cgs * L_cm * sqrt(4π * ρ_scale_gcm3))

    # κ (отношение альфвеновской скорости к скорости нейтралов)
    κ = v_A_cms / v_a_cms

    # ω_ci – циклотронная частота ионов (рад/с)
    ω_ci = e_cgs * H_max_Gs / (c_cgs * m_i_g)

    # ε – параметр замагниченности ионов
    ε = ω_ci * t_scale_s

    # Безразмерный коэффициент ионизации kI
    β0_cgs = β0_m3_s * 1e6          # перевод м³/с → см³/с (1 м³/с = 10⁶ см³/с)
    kI = β0_cgs * n_a_cm3 * t_scale_s

    # α (ξ²)
    α = ξ^2

    # α0 – составной параметр для E_z
    λ_i_λ_z = m_i_g / (m_i_g + m_e_g)   # ≈ 1
    α0 = κ * ξ * sqrt(m_i_g / m_e_g) * λ_i_λ_z

    # ζ – параметр плазменной беты
    ζ = k_B_cgs * T_K / (m_i_g * v_A_cms^2)

    # Безразмерные величины с использованием масштабов из статьи
    L_dim = L_cm / L_SCALE_CM            # безразмерная длина (в единицах 1 см)
    n_a_left_dim = n_a_cm3 / N_SCALE_CM3 # безразмерная концентрация на входе
    T_ion_dim = T_eV / T_SCALE_EV        # безразмерная температура ионизации
    v_a_dim = 1.0                         # скорость нейтралов принята за единицу
    mi_dim = 1.0                           # масса иона за единицу
    me_dim = m_e_g / m_i_g                 # отношение масс

    # Словарь с именами, соответствующими полям SimParams
    params_dict = Dict(
        "L"          => L_dim,
        "M"          => M,
        "mi"         => mi_dim,
        "me"         => me_dim,
        "T_ion"      => T_ion_dim,
        "v_a"        => v_a_dim,
        "n_a_left"   => n_a_left_dim,
        "kI"         => kI,
        "kR"         => kR,
        "γ"          => γ,
        "ε"          => ε,
        "ν_m0"       => ν_m0_guess,
        "α"          => α,
        "α0"         => α0,
        "ζ"          => ζ,
        "ε_dim"      => ε_dim,
        "H0_ratio"   => H0_ratio,   # для справки
        "N1"         => N1
    )

    # Вывод на экран в формате, готовом для вставки в main.jl
    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    println("Параметры для вставки в let-блок (газ: $gas)")
    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    println("params = PartCount.SimParams(")
    println("    L=$(L_dim), M=$(M),")
    println("    mi=$(mi_dim), me=$(me_dim),                     # отношение масс для $gas")
    println("    T_ion=$(T_ion_dim), v_a=$(v_a_dim), n_a_left=$(n_a_left_dim), kI=$(kI), kR=$(kR),")
    println("    γ=$(γ), ε=$(ε), ν_m0=$(ν_m0_guess),")
    println("    α=$(α), α0=$(α0), ζ=$(ζ), ε_dim=$(ε_dim),")
    println("    H0_func=z->$(H0_ratio),                         # внешнее магнитное поле (константа)")
    println("    N1=$(N1)")
    println(")")
    println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Сохранение в CSV файл, если указано
    if output_file !== nothing
        csv_dict = Dict{String,Any}()
        for (k,v) in params_dict
            csv_dict[k] = v
        end
        # Добавим исходные физические данные для справки
        csv_dict["gas"] = gas
        csv_dict["L_m"] = L_m
        csv_dict["v_a_m"] = v_a_m
        csv_dict["n_a_m3"] = n_a_m3
        csv_dict["H_max_T"] = H_max_T
        csv_dict["T_eV"] = T_eV
        csv_dict["β0_m3_s"] = β0_m3_s
        save_params_to_csv(csv_dict, output_file)
    end

    return params_dict
end

"""
    save_params_to_csv(params::Dict, filename::String)

Сохраняет словарь параметров в CSV файл.
"""
function save_params_to_csv(params::Dict, filename::String)
    keys_sorted = sort(collect(keys(params)))
    header = join(keys_sorted, ",")
    values = [params[k] isa Float64 ? @sprintf("%.6e", params[k]) : string(params[k]) for k in keys_sorted]
    data_row = join(values, ",")
    open(filename, "w") do io
        println(io, header)
        println(io, data_row)
    end
    println("Результаты сохранены в файл: ", filename)
end

# Пример для криптона (СПД-70)
function example_kr()
    params = compute_dimensionless_params(
        L_m       = 0.04,
        v_a_m     = 270.0,
        n_a_m3    = 3.0e19,
        H_max_T   = 0.02,
        T_eV      = 14.0,
        gas       = "Kr",
        β0_m3_s   = 1e-14,
        ν_m0_guess = 15.0,
        M         = 100,
        N1        = 100,
        kR        = 0.1,
        γ         = 5/3,
        ε_dim     = 1.0,
        H0_ratio  = 1.9,       # H0 = 1.9 * 200 Гс = 380 Гс
        output_file = "kr_params.csv"
    )
    return params
end

example_kr()