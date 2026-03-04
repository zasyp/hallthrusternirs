using LinearAlgebra
using Plots

# -------------------------------------------------------------------
# Константы
# -------------------------------------------------------------------
const N_FLOOR = 1e-8          # Минимальная концентрация
const T_FLOOR = 1e-6          # Минимальная температура
const VELOCITY_LIMIT = 1e6    # Максимальная скорость частицы
const MIN_PARTICLE_MASS = 1e-8 # Минимальная масса для добавления частицы
const SMOOTHING_PASSES = 3     # Количество проходов сглаживания

# -------------------------------------------------------------------
# Структуры данных
# -------------------------------------------------------------------

"""Счётчики для диагностики выполнения программы"""
mutable struct Counters
    nan::Int
    overspeed::Int
    exited_right::Int
    reflected_left::Int
end

"""Макрочастица плазмы"""
mutable struct Particle
    z::Float64      # координата
    vy::Float64     # скорость по y (азимутальная)
    vz::Float64     # скорость по z (продольная)
    T::Float64      # температура
    q::Float64      # заряд/масса
    active::Bool    # активна ли частица
end

# -------------------------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------------------------

"""
    solve_tridiagonal(a, b, c, d)

Решение трёхдиагональной системы методом прогонки.
a - нижняя диагональ (длина n-1)
b - главная диагональ (длина n)
c - верхняя диагональ (длина n-1)
d - правая часть (длина n)
Возвращает вектор x длины n.
"""
function solve_tridiagonal(a::Vector{Float64}, b::Vector{Float64},
                           c::Vector{Float64}, d::Vector{Float64})
    n = length(b)
    @assert length(a) == n-1 && length(c) == n-1 && length(d) == n
    
    cp = similar(c)
    dp = similar(d)
    
    cp[1] = c[1] / b[1]
    dp[1] = d[1] / b[1]
    
    for i in 2:n-1
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    end
    
    x = zeros(n)
    x[n] = (d[n] - a[n-1] * dp[n-1]) / (b[n] - a[n-1] * cp[n-1])
    
    for i in n-1:-1:1
        x[i] = dp[i] - cp[i] * x[i+1]
    end
    
    return x
end

"""
    interpolation_weights(x, x_grid)

Возвращает индексы двух ближайших узлов сетки и веса для линейной интерполяции.
Возвращает (k0, k1, w0, w1).
"""
function interpolation_weights(x, x_grid)
    M = length(x_grid) - 1
    h = x_grid[2] - x_grid[1]
    x_min = x_grid[1]
    x_max = x_grid[end]
    
    if x <= x_min
        return 1, 2, 1.0, 0.0
    elseif x >= x_max
        return M, M+1, 1.0, 0.0
    else
        k0 = floor(Int, (x - x_min) / h) + 1
        k0 = min(k0, M)
        w1 = (x - x_grid[k0]) / h
        w0 = 1.0 - w1
        return k0, k0+1, w0, w1
    end
end

"""
    smooth_field!(f, window)

Сглаживание поля f скользящим средним с полуюконом window.
"""
function smooth_field!(f, window)
    n = length(f)
    g = copy(f)
    for i in 1:n
        left = max(1, i-window)
        right = min(n, i+window)
        g[i] = sum(@view f[left:right]) / (right-left+1)
    end
    f .= g
    return f
end

"""
    stewart_smoothing!(f, ℓ, h, L)

Осреднение по Стеклову согласно PDF стр. 42.
"""
function stewart_smoothing!(f::Vector{Float64}, ℓ::Int, h::Float64, L::Float64)
    n = length(f)
    f_smooth = copy(f)
    
    for _ in 1:SMOOTHING_PASSES
        for i in 1:n
            z = (i-1) * h
            left = max(1, i-ℓ)
            right = min(n, i+ℓ)
            
            total_weight = 0.0
            sum_val = 0.0
            
            for j in left:right
                zj = (j-1) * h
                dist = abs(z - zj)
                if dist <= ℓ * h
                    weight = 1.0 - dist / (ℓ * h)
                    sum_val += f[j] * weight
                    total_weight += weight
                end
            end
            
            if total_weight > 0
                f_smooth[i] = sum_val / total_weight
            end
        end
        f .= f_smooth
    end
    return f
end

"""
    compute_current!(j, H_x, h)

Вычисление плотности тока j из магнитного поля H_x по закону Ампера.
"""
function compute_current!(j::Vector{Float64}, H_x::Vector{Float64}, h::Float64)
    M = length(H_x)
    @assert length(j) == M+1
    
    for k in 2:M
        j[k] = (H_x[k] - H_x[k-1]) / h
    end
    j[1] = j[M+1] = 0.0  # граничные условия j=0
    
    return j
end

# -------------------------------------------------------------------
# Гидродинамические блоки
# -------------------------------------------------------------------

"""
    update_neutrals!(n_a_new, n_a_old, n_ion, τ, v_a, kI, h, n_left_bc)

Обновление концентрации нейтралов по уравнению (31).
"""
function update_neutrals!(n_a_new::Vector{Float64},
                          n_a_old::Vector{Float64},
                          n_ion::Vector{Float64},
                          τ::Float64,
                          v_a::Float64,
                          kI::Float64,
                          h::Float64,
                          n_left_bc::Float64)

    M = length(n_a_old) - 1
    C = v_a * τ / h

    if C > 1.0
        @warn "CFL violated for neutrals: v_a * τ / h = $C > 1, reducing timestep recommended"
    end

    n_a_new[1] = n_left_bc  # фиксированное значение на левой границе

    for k in 2:M+1
        convection = -v_a * τ * (n_a_old[k] - n_a_old[k-1]) / h
        # Неявная схема для ионизационного члена
        n_a_new[k] = (n_a_old[k] + convection) / (1.0 + τ * kI * n_ion[k])
        n_a_new[k] = max(n_a_new[k], 0.0)
    end

    return n_a_new
end

"""
    intermediate_temperature_hybrid!(T_new, T_old, n, vz, j, n_a, τ, γ, mi_over_mΣ, ν_m0, kI, h)

Вычисление промежуточной температуры по уравнению (32) (без переноса).
"""
function intermediate_temperature_hybrid!(T_new::Vector{Float64}, T_old::Vector{Float64},
                                          n::Vector{Float64}, vz::Vector{Float64},
                                          j::Vector{Float64}, n_a::Vector{Float64},
                                          τ::Float64, γ::Float64, mi_over_mΣ::Float64,
                                          ν_m0::Float64, kI::Float64, h::Float64)
    M = length(T_old) - 1
    
    for k in 2:M
        n_safe = max(n[k], N_FLOOR)
        T_safe = max(T_old[k], T_FLOOR)
        
        ν_m = ν_m0 / T_safe^(3/2)
        
        # Центральная разность для dvz/dz
        dvz = (vz[k+1] - vz[k-1]) / (2h)
        
        # Источники: джоулев нагрев и ионизационный нагрев
        Q_coll = (γ-1) * mi_over_mΣ * ν_m * j[k]^2 / n_safe
        Q_ion = (γ-1) * kI * n_a[k] * n_safe
        
        T_new[k] = T_old[k] + τ * (Q_coll + Q_ion - (γ-1) * T_old[k] * dvz)
        T_new[k] = max(T_new[k], T_FLOOR)
    end
    
    # Экстраполяция на границы
    T_new[1] = T_new[2]
    T_new[M+1] = T_new[M]
    
    return T_new
end

# -------------------------------------------------------------------
# Электродинамические блоки
# -------------------------------------------------------------------

"""
    solve_electric_field_hybrid!(E_y, H_x_old, j_old, n, vz, T, τ, α, ν_m0, h, x_grid, H0_func, bc_type)

Решение для Ey по схеме (36) с граничными условиями (34).
"""
function solve_electric_field_hybrid!(E_y::Vector{Float64},
                                      H_x_old::Vector{Float64},
                                      j_old::Vector{Float64},
                                      n::Vector{Float64},
                                      vz::Vector{Float64},
                                      T::Vector{Float64},
                                      τ::Float64,
                                      α::Float64,
                                      ν_m0::Float64,
                                      h::Float64,
                                      x_grid::AbstractVector{Float64},
                                      H0_func,
                                      bc_type::Symbol)
    M = length(H_x_old)   # количество полуцелых узлов
    @assert length(E_y) == M+1
    @assert length(j_old) == M+1
    @assert length(n) == M+1
    @assert length(vz) == M+1
    @assert length(T) == M+1

    # Интерполяция H_x_old с полуцелых на целые узлы
    H_int = zeros(M+1)
    H_int[1] = H_x_old[1]
    for k in 2:M
        H_int[k] = 0.5 * (H_x_old[k-1] + H_x_old[k])
    end
    H_int[M+1] = H_x_old[M]

    H_star = H_int .+ H0_func.(x_grid)

    # Подготовка коэффициентов трёхдиагональной системы
    a = zeros(M+1)
    b = zeros(M+1)
    c = zeros(M+1)
    d = zeros(M+1)

    for k in 2:M
        n_safe = max(n[k], N_FLOOR)
        T_safe = max(T[k], T_FLOOR)
        
        A = α / (n_safe * h^2)
        B = (ν_m0 / T_safe^(3/2)) * (τ / h^2)
        
        dvz = (vz[k+1] - vz[k-1]) / (2h)
        C = vz[k] * τ / (4h)
        D = (α * τ / (n_safe * h^2)) * dvz

        a[k] = -A - (B + D) - C
        b[k] = 1.0 + 2A + 2(B + D)
        c[k] = -A - (B + D) + C

        dj = (j_old[k+1] - j_old[k-1]) / (2h)
        d[k] = (ν_m0 / T_safe^(3/2)) * j_old[k] - H_star[k] * vz[k] +
               (α / n_safe) * j_old[k] * dvz + (α / n_safe) * vz[k] * dj
    end

    # Граничные условия
    if bc_type == :j0
        # Левая граница
        if n[1] > N_FLOOR
            dj_left = j_old[2] / h
            E_y[1] = (-H_star[1] + (α / n[1]) * dj_left) * vz[1]
        else
            E_y[1] = 0.0
        end
        
        # Правая граница
        if n[M+1] > N_FLOOR
            dj_right = -j_old[M] / h
            E_y[M+1] = (-H_star[M+1] + (α / n[M+1]) * dj_right) * vz[M+1]
        else
            E_y[M+1] = 0.0
        end

        # Модификация внутренней системы
        if M-1 > 0
            d[2] -= a[2] * E_y[1]
            a[2] = 0.0
            d[M] -= c[M] * E_y[M+1]
            c[M] = 0.0
        end
    else
        error("Unsupported boundary condition: $bc_type")
    end

    # Решение для внутренних узлов
    if M-1 > 0
        N_inner = M-1
        a_inner = view(a, 3:M)   # a[3], a[4], ..., a[M]
        b_inner = view(b, 2:M)   # b[2], ..., b[M]
        c_inner = view(c, 2:M-1) # c[2], ..., c[M-1]
        d_inner = view(d, 2:M)   # d[2], ..., d[M]
        
        # Преобразование в полные векторы для solve_tridiagonal
        a_full = collect(a_inner)
        b_full = collect(b_inner)
        c_full = collect(c_inner)
        d_full = collect(d_inner)
        
        E_inner = solve_tridiagonal(a_full, b_full, c_full, d_full)
        for (idx, val) in enumerate(E_inner)
            E_y[idx+1] = val
        end
    end

    # Обновление H_x по закону Фарадея (37)
    H_x_new = similar(H_x_old)
    for k in 1:M
        H_x_new[k] = H_x_old[k] + τ * (E_y[k+1] - E_y[k]) / h
    end

    # Обновление j по закону Ампера
    j_new = zeros(M+1)
    compute_current!(j_new, H_x_new, h)

    return E_y, H_x_new, j_new
end

"""
    compute_Ez_hybrid!(Ez, H_x_old, H_x_new, j_old, j_new, n, T, vy, n_a, α0, ζ, kI, ε_dim, va, h, x_grid, H0_func; N_REG, νE)

Вычисление Ez по явной формуле (38) с полусуммами.
"""
function compute_Ez_hybrid!(Ez::Vector{Float64},
                            H_x_old::Vector{Float64}, H_x_new::Vector{Float64},
                            j_old::Vector{Float64}, j_new::Vector{Float64},
                            n::Vector{Float64}, T::Vector{Float64},
                            vy::Vector{Float64},
                            n_a::Vector{Float64},
                            α0::Float64, ζ::Float64, kI::Float64, ε_dim::Float64,
                            va::Float64, h::Float64, x_grid::AbstractVector{Float64},
                            H0_func;
                            N_REG::Float64 = N_FLOOR, νE::Float64 = 0.08)
    M = length(H_x_old)
    @assert length(Ez) == M+1

    # Полусуммы для H и j
    H_x_mid = 0.5 * (H_x_old + H_x_new)
    j_mid = 0.5 * (j_old + j_new)

    # Интерполяция H_x_mid на целые узлы
    H_int = zeros(M+1)
    H_int[1] = H_x_mid[1]
    for k in 2:M
        H_int[k] = 0.5 * (H_x_mid[k-1] + H_x_mid[k])
    end
    H_int[M+1] = H_x_mid[M]

    H_star = H_int .+ H0_func.(x_grid)

    # Сглаживание для производной
    n_s = copy(n)
    T_s = copy(T)
    smooth_field!(n_s, 2)
    smooth_field!(T_s, 2)

    # Производная d(nT)/dz
    d_nT = zeros(M+1)
    for k in 2:M
        d_nT[k] = (n_s[k+1]*T_s[k+1] - n_s[k-1]*T_s[k-1]) / (2h)
    end
    d_nT[1] = d_nT[2]
    d_nT[M+1] = d_nT[M]

    # Основной цикл
    for k in 1:M+1
        n_safe = max(n_s[k], N_REG)
        term1 = H_star[k] * vy[k]
        term2 = (α0 / n_safe) * H_star[k] * j_mid[k]
        term3 = (ζ * α0 / n_safe) * d_nT[k]
        term4 = (kI / ε_dim) * n_a[k] * va
        Ez[k] = term1 - term2 - term3 - term4
    end

    # Искусственная вязкость (сглаживание)
    for _ in 1:SMOOTHING_PASSES
        Ez_smooth = copy(Ez)
        for k in 2:M
            Ez_smooth[k] = Ez[k] + νE * (Ez[k+1] - 2Ez[k] + Ez[k-1])
        end
        Ez .= Ez_smooth
    end

    return Ez
end

# -------------------------------------------------------------------
# Функции работы с частицами
# -------------------------------------------------------------------

"""
    deposit_particles!(particles, x_grid, n, v_y, v_z, T, h, pj_weights)

Осаждение частиц на сетку: вычисление n, v_y, v_z, T в узлах по формулам (22).
"""
function deposit_particles!(particles, x_grid, n, v_y, v_z, T, h, pj_weights)
    M = length(x_grid) - 1
    fill!(n, 0.0)
    n_v_y = zeros(length(n))
    n_v_z = zeros(length(n))
    n_T = zeros(length(n))

    # Сбор моментов
    for p in particles
        p.active || continue
        
        k0, k1, w0, w1 = interpolation_weights(p.z, x_grid)
        
        n[k0] += p.q * w0
        n[k1] += p.q * w1
        
        n_v_y[k0] += p.q * w0 * p.vy
        n_v_y[k1] += p.q * w1 * p.vy
        
        n_v_z[k0] += p.q * w0 * p.vz
        n_v_z[k1] += p.q * w1 * p.vz
        
        n_T[k0] += p.q * w0 * p.T
        n_T[k1] += p.q * w1 * p.T
    end

    # Нормировка с учетом объема ячеек
    for k in eachindex(n)
        vol = (k == 1 || k == length(n)) ? h/2 : h
        
        if n[k] > N_FLOOR
            v_y[k] = n_v_y[k] / n[k]
            v_z[k] = n_v_z[k] / n[k]
            T[k] = n_T[k] / n[k]
        else
            v_y[k] = 0.0
            v_z[k] = 0.0
            T[k] = T_FLOOR
        end
        
        n[k] /= vol
    end
    
    return n, v_y, v_z, T
end

"""
    move_particles!(particles, E_y0, E_y1, E_z0, E_z1, H_x0, H_x1, j0, j1,
                    ν_m0_grid0, ν_m0_grid1, x_grid, x_half, τ, h, ε, ν_m0_const,
                    v_a, kI, n_a_new, H0_func, counters)

Движение частиц по схеме Эйлера с пересчётом (39) с интерполяцией полей.
"""
function move_particles!(particles,
                         E_y0, E_y1, E_z0, E_z1,
                         H_x0, H_x1, j0, j1,
                         ν_m0_grid0, ν_m0_grid1,
                         x_grid, x_half, τ, h, ε, ν_m0_const,
                         v_a, kI, n_a_new, H0_func,
                         counters::Counters)
    thrust_step = 0.0
    L = x_grid[end]
    
    for p in particles
        p.active || continue
        z = p.z
        vy = p.vy
        vz = p.vz
        v_abs = sqrt(vy^2 + vz^2)
        
        # Адаптивный подшаг для соблюдения условия Куранта
        N0 = max(1, ceil(Int, τ * v_abs / (0.25 * h)))
        τ0 = τ / N0

        for i in 1:N0
            # Коррекция положения при выходе за границы
            if z < x_grid[1]
                z = x_grid[1]
                vz = abs(vz)
            elseif z > x_grid[end]
                z = x_grid[end]
            end

            # Относительное время в середине подшага
            t_mid_rel = (i - 0.5) / N0
            
            # Интерполяция для целых узлов
            k0, k1, w0, w1 = interpolation_weights(z, x_grid)
            
            # Функция линейной интерполяции по времени и пространству
            function interp_field(F0, F1)
                val_now = w0 * F0[k0] + w1 * F0[k1]
                val_next = w0 * F1[k0] + w1 * F1[k1]
                return (1 - t_mid_rel) * val_now + t_mid_rel * val_next
            end
            
            E_y_mid = interp_field(E_y0, E_y1)
            E_z_mid = interp_field(E_z0, E_z1)
            j_mid = interp_field(j0, j1)
            ν_m_mid = interp_field(ν_m0_grid0, ν_m0_grid1)
            
            # Интерполяция для полуцелых узлов (H_x)
            if z <= x_half[1]
                kh, wh = 1, 1.0
            elseif z >= x_half[end]
                kh, wh = length(x_half), 1.0
            else
                kh = floor(Int, (z - x_half[1]) / h) + 1
                kh = clamp(kh, 1, length(x_half)-1)
                wh = (z - x_half[kh]) / h
            end
            
            H_now = (kh < length(x_half)) ? (1-wh)*H_x0[kh] + wh*H_x0[kh+1] : H_x0[kh]
            H_next = (kh < length(x_half)) ? (1-wh)*H_x1[kh] + wh*H_x1[kh+1] : H_x1[kh]
            H_mid = (1 - t_mid_rel) * H_now + t_mid_rel * H_next
            
            H_star_mid = H_mid + H0_func(z)

            # Схема Эйлера с пересчётом (предиктор-корректор)
            vy_pred = vy + 0.5 * τ0 * ε * (E_y_mid + H_star_mid * vz - ν_m_mid * j_mid)
            vz_pred = vz + 0.5 * τ0 * ε * (E_z_mid - H_star_mid * vy)

            vy_new = vy + τ0 * ε * (E_y_mid + H_star_mid * vz_pred - ν_m_mid * j_mid)
            vz_new = vz + τ0 * ε * (E_z_mid - H_star_mid * vy_pred)

            # Проверка на некорректные значения
            if !isfinite(vy_new) || !isfinite(vz_new)
                counters.nan += 1
                p.active = false
                break
            end
            
            if abs(vy_new) > VELOCITY_LIMIT || abs(vz_new) > VELOCITY_LIMIT
                counters.overspeed += 1
                p.active = false
                break
            end

            # Обновление координаты (среднее арифметическое скоростей)
            z_new = z + τ0 * (vz + vz_new) / 2
            
            p.z = z_new
            p.vy = vy_new
            p.vz = vz_new
            
            z, vy, vz = z_new, vy_new, vz_new
        end

        # Обработка граничных условий
        if p.z >= L
            p.active = false
            thrust_step += p.q * p.vz
            counters.exited_right += 1
        elseif p.z <= 0.0
            p.z = -p.z
            p.vz = -p.vz
            counters.reflected_left += 1
        end
    end
    
    return thrust_step
end

"""
    add_new_particles!(particles, n_a_new, n_ion, x_grid, τ, kI, v_a, T_ion, h)

Добавление новых частиц от ионизации согласно PDF стр. 41.
"""
function add_new_particles!(particles, n_a_new, n_ion, x_grid, τ, kI, v_a, T_ion, h)
    for k in eachindex(x_grid)
        q_new = kI * n_a_new[k] * n_ion[k] * τ * h
        q_new = min(q_new, n_a_new[k] * h)  # не больше, чем есть нейтралов
        
        if q_new < MIN_PARTICLE_MASS
            continue
        end
        
        # Случайное положение в ячейке
        z = x_grid[k] + h * (rand() - 0.5)
        z = clamp(z, x_grid[1], x_grid[end])
        
        push!(particles, Particle(z, 0.0, v_a, T_ion, q_new, true))
    end
end

"""
    remove_inactive_particles!(particles, L, τ, kR)

Удаление частиц, вышедших за границы или рекомбинировавших.
"""
function remove_inactive_particles!(particles, L, τ, kR)
    for p in particles
        if p.z < 0 || p.z > L
            p.active = false
            continue
        end
        # Вероятность рекомбинации
        P_rec = min(1.0, kR * p.q * τ)
        rand() < P_rec && (p.active = false)
    end
    filter!(p -> p.active, particles)
    return particles
end

# -------------------------------------------------------------------
# Постобработка и визуализация
# -------------------------------------------------------------------

"""
    plot_results(snapshots, thrust_time, thrust_values, save_times)

Построение графиков профилей и силы тяги.
"""
function plot_results(snapshots, thrust_time, thrust_values, save_times)
    if !isempty(snapshots)
        times = sort(collect(keys(snapshots)))
        p1 = plot(title="n_a", xlabel="z")
        p2 = plot(title="n_i", xlabel="z")
        p3 = plot(title="v_z", xlabel="z")
        p4 = plot(title="E_z", xlabel="z")
        colors = palette(:tab10, length(times))
        
        for (idx, t) in enumerate(times)
            z, n_a, n, v_z, E_z = snapshots[t]
            plot!(p1, z, n_a, label="t=$t", color=colors[idx])
            plot!(p2, z, n, label="t=$t", color=colors[idx])
            plot!(p3, z, v_z, label="t=$t", color=colors[idx])
            plot!(p4, z, E_z, label="t=$t", color=colors[idx])
        end
        
        plot(p1, p2, p3, p4, layout=4, size=(800,600))
        savefig("profiles.png")
        display(current())
    end
    
    if !isempty(thrust_time)
        p_thrust = plot(thrust_time, thrust_values, xlabel="t", ylabel="F_T", legend=false)
        savefig("thrust.png")
        display(p_thrust)
    end
end

# -------------------------------------------------------------------
# Основная функция моделирования
# -------------------------------------------------------------------

"""
    run_simulation(; L=1.0, M=100, N1=1000, T_ion=1.0, v_a=0.1, n_a_left=10.0,
                    kI=2.0, ε=1.0, ν_m0=15.0, γ=5/3, mi_over_mΣ=1.0,
                    H0_func=z->1.0, α=0.75^2, α0=1.0, ζ=0.001, ε_dim=1.0,
                    total_time=50.0, save_times=[10.0,20.0,30.0,40.0,50.0],
                    do_plot=true, kR=0.1)

Запуск гибридного моделирования СПД.
Возвращает словарь снимков и массивы времени и силы тяги.
"""
function run_simulation(;
    L=1.0, M=100, N1=1000, T_ion=1.0, v_a=0.1, n_a_left=10.0, kI=2.0,
    ε=1.0, ν_m0=15.0, γ=5/3, mi_over_mΣ=1.0, H0_func=z->1.0,
    α=0.75^2, α0=1.0, ζ=0.001, ε_dim=1.0,
    total_time=50.0, save_times=[10.0,20.0,30.0,40.0,50.0],
    do_plot=true, kR=0.1)

    # Инициализация сеток
    h = L / M
    x_grid = range(0, L, length=M+1)          # целые узлы
    x_half = range(h/2, L-h/2, length=M)      # полуцелые узлы

    # Инициализация макрочастиц (30)
    particles = Particle[]
    q0 = L / (N1 * M)
    for k in 1:M
        z0 = x_grid[k] + h/2   # центр ячейки
        for s in 1:N1
            φ = 2π*(s-1)/N1 + π/N1 + k*sqrt(2)
            vy = 5.8 * cos(φ)
            vz = 5.8 * sin(φ)
            push!(particles, Particle(z0, vy, vz, T_ion, q0, true))
        end
    end

    # Начальное распределение нейтралов
    n_a_old = n_a_left .* exp.(-5 .* x_grid / L)

    # Поля
    H_x_half = zeros(M)          # на полуцелых
    j = zeros(M+1)               # на целых
    E_y = zeros(M+1)
    E_z = zeros(M+1)
    T_e = fill(T_ion, M+1)

    # Гидродинамические величины
    n_ion = zeros(M+1)
    v_iy = zeros(M+1)
    v_iz = zeros(M+1)

    # Для хранения старых полей
    H_x_old = copy(H_x_half)
    j_old = copy(j)
    E_y_old = copy(E_y)
    E_z_old = copy(E_z)

    # Хранение результатов
    snapshots = Dict{Float64, Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}}()
    thrust_time = Float64[]
    thrust_values = Float64[]

    t = 0.0
    step = 0

    while t < total_time
        # Осаждение частиц
        deposit_particles!(particles, x_grid, n_ion, v_iy, v_iz, T_e, h, nothing)
        smooth_field!(n_ion, 4)
        smooth_field!(v_iy, 4)
        smooth_field!(v_iz, 4)
        smooth_field!(T_e, 4)

        # Определение шага по времени
        max_vz = max(maximum(abs.(v_iz)), 1e-12)
        ν_m_grid = ν_m0 ./ max.(T_e, T_FLOOR).^(3/2)
        ν_max = maximum(ν_m_grid)
        τ_coll = 0.5 / max(ν_max, 1e-8)
        τ = min(h / v_a, 0.4 * h / max_vz, τ_coll, total_time - t)

        # Вычисление тока из H
        compute_current!(j, H_x_half, h)

        # ---- Этап 1: промежуточная температура ----
        T_tilde = similar(T_e)
        intermediate_temperature_hybrid!(T_tilde, T_e, n_ion, v_iz, j, n_a_old,
                                         τ, γ, mi_over_mΣ, ν_m0, kI, h)

        # "Отемпературивание" частиц
        for p in particles
            p.active || continue
            k0, k1, w0, w1 = interpolation_weights(p.z, x_grid)
            p.T = w0 * T_tilde[k0] + w1 * T_tilde[k1]
        end

        # ---- Обновление нейтралов ----
        n_a_new = similar(n_a_old)
        update_neutrals!(n_a_new, n_a_old, n_ion, τ, v_a, kI, h, n_a_left)

        # Сохранение старых полей
        copyto!(H_x_old, H_x_half)
        copyto!(j_old, j)
        copyto!(E_y_old, E_y)
        copyto!(E_z_old, E_z)

        # ---- Решение для Ey ----
        E_y, H_x_half, j = solve_electric_field_hybrid!(E_y,
                                                         H_x_old, j_old,
                                                         n_ion, v_iz, T_e,
                                                         τ, α, ν_m0, h, x_grid,
                                                         H0_func, :j0)
        stewart_smoothing!(E_y, 15, h, L)   # ℓ=15, можно увеличить до 20
        stewart_smoothing!(H_x_half, 15, h, L)
        stewart_smoothing!(j, 15, h, L)

        # Сглаживание
        smooth_field!(E_y, 3)
        smooth_field!(H_x_half, 3)
        smooth_field!(j, 3)

        # ---- Вычисление Ez ----
        compute_Ez_hybrid!(E_z,
                           H_x_old, H_x_half,
                           j_old, j,
                           n_ion, T_e, v_iy,
                           n_a_new,
                           α0, ζ, kI, ε_dim, v_a, h, x_grid,
                           H0_func)
        stewart_smoothing!(E_z, 20, h, L)
        smooth_field!(E_z, 5)

        # ---- Движение частиц ----
        counters = Counters(0, 0, 0, 0)
        ν_m_grid = ν_m0 ./ max.(T_e, T_FLOOR).^(3/2)
        
        thrust_step = move_particles!(particles,
                                      E_y_old, E_y,
                                      E_z_old, E_z,
                                      H_x_old, H_x_half,
                                      j_old, j,
                                      ν_m_grid, ν_m_grid,
                                      x_grid, x_half, τ, h, ε, ν_m0,
                                      v_a, kI, n_a_new, H0_func,
                                      counters)

        push!(thrust_time, t+τ)
        push!(thrust_values, thrust_step / τ)

        # ---- Добавление новых частиц ----
        add_new_particles!(particles, n_a_new, n_ion, x_grid, τ, kI, v_a, T_ion, h)

        # ---- Удаление неактивных ----
        remove_inactive_particles!(particles, L, τ, kR)

        # ---- Сохранение снимков ----
        for st in save_times
            if abs(t+τ - st) < τ/2 && !haskey(snapshots, st)
                snapshots[st] = (copy(x_grid), copy(n_a_new), copy(n_ion), copy(v_iz), copy(E_z))
            end
        end

        # ---- Подготовка к следующему шагу ----
        n_a_old .= n_a_new
        t += τ
        step += 1

        # Вывод диагностики
        println("Step $step, t=$t, #particles=$(length(particles)), ",
                "min_n=$(minimum(n_ion)), max_Ez=$(maximum(abs.(E_z))), ",
                "nan=$(counters.nan), overspeed=$(counters.overspeed), ",
                "exited=$(counters.exited_right), reflected=$(counters.reflected_left)")
    end

    do_plot && plot_results(snapshots, thrust_time, thrust_values, save_times)
    
    return snapshots, thrust_time, thrust_values
end

# -------------------------------------------------------------------
# Пример запуска
# -------------------------------------------------------------------
let
    H0_func(z) = 1.0
    run_simulation(
    L=1.0, M=100, N1=100, T_ion=1.0, v_a=0.1, n_a_left=10.0, kI=1.0,
    ε=1.0, ν_m0=15.0, γ=5/3, mi_over_mΣ=1.0, H0_func=z->1.0,
    α=0.283,          # ξ² ≈ 0.532²
    α0=160.0,         # κξ √(mᵢ/mₑ) ≈ 1·0.532·300
    ζ=0.061,          # из PDF
    ε_dim=1.0,
    total_time=20.0,  # дать время на развитие
    save_times=[5.0,10.0,15.0,20.0],
    do_plot=true,
    kR=0.0
)
end