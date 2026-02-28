using LinearAlgebra
using Plots

# Структура макрочастицы с полем active
mutable struct Particle
    z::Float64
    vy::Float64
    vz::Float64
    T::Float64
    q::Float64
    active::Bool
end

# ------------------------------------------------------------
# Базовые функции (интерполяция, сглаживание, прогонка)
# ------------------------------------------------------------

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

Возвращает индексы k0, k0+1 и веса w0, w1 для линейной интерполяции
значения в точке x по узлам равномерной сетки x_grid.
Теперь с защитой от выхода за границы.
"""
function interpolation_weights(x, x_grid)
    M = length(x_grid) - 1
    h = x_grid[2] - x_grid[1]
    x_min = x_grid[1]
    x_max = x_grid[end]

    if x <= x_min
        return 1, 2, 1.0, 0.0          # целиком на левый узел
    elseif x >= x_max
        return M, M+1, 1.0, 0.0        # целиком на правый узел
    else
        k0 = floor(Int, (x - x_min) / h) + 1
        k0 = min(k0, M)                 # гарантия, что k0 ≤ M
        w1 = (x - x_grid[k0]) / h
        w0 = 1.0 - w1
        return k0, k0+1, w0, w1
    end
end

"""
    smooth_field!(f, ℓ)

Сглаживание поля f простым скользящим средним с окном 2ℓ+1.
"""
function smooth_field!(f, ℓ)
    n = length(f)
    g = copy(f)
    for i in 1:n
        left = max(1, i-ℓ)
        right = min(n, i+ℓ)
        g[i] = sum(f[left:right]) / (right-left+1)
    end
    f .= g
end

# ------------------------------------------------------------
# Уравнение переноса нейтралов (31)
# ------------------------------------------------------------
function update_neutrals(
    n_a_new::Vector{Float64}, n_a_old::Vector{Float64},
    n_ion::Vector{Float64}, τ::Float64,
    v_a::Float64, kI::Float64,
    h::Float64, na_const::Float64
)
    if τ > h / v_a
        error("Time step τ must satisfy Courant condition: τ <= h / v_a")
    end
    M = length(n_a_old) - 1
    n_a_new[1] = na_const
    for k in 2:M+1
        denom = 1.0 + τ * kI * n_ion[k]
        term1 = n_a_old[k] * (1.0 - v_a * τ / h) / denom
        term2 = n_a_old[k-1] * (v_a * τ / h) / denom
        n_a_new[k] = term1 + term2
        if n_a_new[k] < 0.0
            n_a_new[k] = 0.0
        end
    end
    return n_a_new
end

# ------------------------------------------------------------
# Промежуточная температура (без переноса) – формула (32)
# ------------------------------------------------------------
function compute_Ez!(
    Ez::Vector{Float64}, H_x::Vector{Float64}, H0::Float64,
    n::Vector{Float64}, T::Vector{Float64}, j::Vector{Float64},
    vy::Vector{Float64}, n_a::Vector{Float64}, α0::Float64,
    ζ::Float64, kI::Float64, ε_dim::Float64,
    va::Float64, h::Float64
)
    M = length(H_x)
    @assert length(Ez) == M+1 && length(n) == M+1 && length(T) == M+1
    @assert length(j) == M+1 && length(vy) == M+1 && length(n_a) == M+1

    H_int = zeros(M+1)
    H_int[1] = H_x[1]
    for k in 2:M
        H_int[k] = (H_x[k-1] + H_x[k]) / 2
    end
    H_int[M+1] = H_x[M]
    H_star = H_int .+ H0

    d_nT = zeros(M+1)
    for k in 2:M
        d_nT[k] = (n[k+1]*T[k+1] - n[k-1]*T[k-1]) / (2h)
    end
    d_nT[1] = d_nT[2]
    d_nT[M+1] = d_nT[M]

    for k in 1:M+1
        n_safe = max(n[k], 1)   # увеличено до 1e-4
        # Вычисляем Ez с ограничением
        term1 = H_star[k] * vy[k]
        term2 = (α0 / n_safe) * H_star[k] * j[k]
        term3 = (ζ * α0 / n_safe) * d_nT[k]
        term4 = (kI / ε_dim) * n_a[k] * va
        Ez[k] = term1 - term2 - term3 - term4
        if abs(Ez[k]) > 1e3
            Ez[k] = sign(Ez[k]) * 1e3
        end
    end
    return Ez
end

# ------------------------------------------------------------
# Ток j по магнитному полю H (закон Ампера)
# ------------------------------------------------------------
function compute_current!(j::Vector{Float64}, H_x::Vector{Float64}, h::Float64)
    M = length(H_x)
    @assert length(j) == M+1
    for k in 2:M
        j[k] = (H_x[k] - H_x[k-1]) / h
    end
    j[1] = 0.0
    j[M+1] = 0.0
    return j
end

# ------------------------------------------------------------
# Решение для Ey (обобщённый закон Ома + Фарадей) – формула (36)
# ------------------------------------------------------------
function solve_electric_field!(
    E_y::Vector{Float64}, H_x::Vector{Float64}, H0::Float64,
    n::Vector{Float64}, vz::Vector{Float64}, T::Vector{Float64},
    j_old::Vector{Float64}, τ::Float64, α::Float64,
    ν_m0::Float64, h::Float64, bc_type::Symbol
)
    M = length(H_x)
    @assert length(E_y) == M+1
    @assert length(n) == M+1 && length(vz) == M+1 && length(T) == M+1 && length(j_old) == M+1

    H_int = zeros(M+1)
    H_int[1] = H_x[1]
    for k in 2:M
        H_int[k] = (H_x[k-1] + H_x[k]) / 2
    end
    H_int[M+1] = H_x[M]

    a = zeros(M+1)
    b = zeros(M+1)
    c = zeros(M+1)
    d = zeros(M+1)

    for k in 2:M
        n_safe = max(n[k], 1e-12)
        T_safe = max(T[k], 1e-12)
        if n_safe > 0 && T_safe > 0
            coeff = α / (n_safe * h^2)
            a[k] = -coeff
            b[k] = 1.0 + 2.0 * coeff
            c[k] = -coeff
            H_star_k = H_int[k] + H0
            dvz = (vz[k+1] - vz[k-1]) / (2h)
            dj  = (j_old[k+1] - j_old[k-1]) / (2h)
            d[k] = (ν_m0 / T_safe^(3/2)) * j_old[k] - H_star_k * vz[k] +
                   (α * j_old[k] / n_safe) * dvz + (α * vz[k] / n_safe) * dj
        else
            b[k] = 1.0
            d[k] = 0.0
        end
    end

    if bc_type == :j0
        # Левая граница
        if n[1] > 1e-12
            H_star_1 = H_int[1] + H0
            dj_left = j_old[2] / h
            E_y[1] = -H_star_1 * vz[1] + (α / n[1]) * dj_left * vz[1]
        else
            E_y[1] = 0.0
        end
        # Правая граница
        if n[M+1] > 1e-12
            H_star_end = H_int[M+1] + H0
            dj_right = -j_old[M] / h
            E_y[M+1] = -H_star_end * vz[M+1] + (α / n[M+1]) * dj_right * vz[M+1]
        else
            E_y[M+1] = 0.0
        end
        # Исключаем граничные узлы
        d[2] -= a[2] * E_y[1]
        a[2] = 0.0
        d[M] -= c[M] * E_y[M+1]
        c[M] = 0.0
    else
        error("Unsupported boundary condition: $bc_type")
    end

    # Решение для внутренних узлов (2..M)
    N_inner = M-1
    if N_inner > 0
        a_inner = zeros(N_inner-1)
        b_inner = zeros(N_inner)
        c_inner = zeros(N_inner-1)
        d_inner = zeros(N_inner)

        for i in 1:N_inner
            k = i+1
            b_inner[i] = b[k]
            d_inner[i] = d[k]
            if i < N_inner
                c_inner[i] = c[k]
            end
            if i > 1
                a_inner[i-1] = a[k]
            end
        end

        E_inner = solve_tridiagonal(a_inner, b_inner, c_inner, d_inner)
        for (idx, val) in enumerate(E_inner)
            E_y[idx+1] = val
        end
    end

    # Обновление H_x
    H_x_new = similar(H_x)
    for k in 1:M
        H_x_new[k] = H_x[k] + τ * (E_y[k+1] - E_y[k]) / h
    end

    # Обновление j
    j_new = zeros(M+1)
    compute_current!(j_new, H_x_new, h)

    return E_y, H_x_new, j_new
end

# ------------------------------------------------------------
# Явная формула для Ez (38)
# ------------------------------------------------------------
function compute_Ez!(
    Ez::Vector{Float64}, H_x::Vector{Float64}, H0::Float64,
    n::Vector{Float64}, T::Vector{Float64}, j::Vector{Float64},
    vy::Vector{Float64}, n_a::Vector{Float64}, α0::Float64,
    ζ::Float64, kI::Float64, ε_dim::Float64,
    va::Float64, h::Float64
)
    M = length(H_x)
    @assert length(Ez) == M+1 && length(n) == M+1 && length(T) == M+1
    @assert length(j) == M+1 && length(vy) == M+1 && length(n_a) == M+1

    H_int = zeros(M+1)
    H_int[1] = H_x[1]
    for k in 2:M
        H_int[k] = (H_x[k-1] + H_x[k]) / 2
    end
    H_int[M+1] = H_x[M]
    H_star = H_int .+ H0

    d_nT = zeros(M+1)
    for k in 2:M
        d_nT[k] = (n[k+1]*T[k+1] - n[k-1]*T[k-1]) / (2h)
    end
    d_nT[1] = d_nT[2]
    d_nT[M+1] = d_nT[M]

    for k in 1:M+1
        n_safe = max(n[k], 1)   # увеличено до 1e-4
        # Вычисляем Ez с ограничением
        term1 = H_star[k] * vy[k]
        term2 = (α0 / n_safe) * H_star[k] * j[k]
        term3 = (ζ * α0 / n_safe) * d_nT[k]
        term4 = (kI / ε_dim) * n_a[k] * va
        Ez[k] = term1 - term2 - term3 - term4
        if abs(Ez[k]) > 1e3
            Ez[k] = sign(Ez[k]) * 1e3
        end
    end
    return Ez
end

# ------------------------------------------------------------
# Осреднение величин по частицам на сетку (формулы (17),(22))
# ------------------------------------------------------------
function compute_grid_quantities!(particles, x_grid, n, v_y, v_z, T, h)
    M = length(x_grid) - 1
    fill!(n, 0.0)
    n_v_y = zeros(length(n))
    n_v_z = zeros(length(n))
    n_T   = zeros(length(n))

    for p in particles
        p.active || continue
        k0, k1, w0, w1 = interpolation_weights(p.z, x_grid)
        n[k0] += p.q * w0
        n_v_y[k0] += p.q * w0 * p.vy
        n_v_z[k0] += p.q * w0 * p.vz
        n_T[k0]   += p.q * w0 * p.T

        n[k1] += p.q * w1
        n_v_y[k1] += p.q * w1 * p.vy
        n_v_z[k1] += p.q * w1 * p.vz
        n_T[k1]   += p.q * w1 * p.T
    end

    for k in eachindex(n)
        vol = (k == 1 || k == length(n)) ? h/2 : h
        if vol > 0
            n[k] /= vol
            if n[k] > 1e-12
                v_y[k] = n_v_y[k] / (n[k] * vol)
                v_z[k] = n_v_z[k] / (n[k] * vol)
                T[k]   = n_T[k]   / (n[k] * vol)
            else
                v_y[k] = 0.0
                v_z[k] = 0.0
                T[k]   = 1e-12
            end
        else
            n[k] = 1e-12
            v_y[k] = 0.0
            v_z[k] = 0.0
            T[k]   = 1e-12
        end
    end
    return
end

# ------------------------------------------------------------
# Движение макрочастиц (схема Эйлера с пересчётом, формулы (39)-(40))
# ------------------------------------------------------------
function move_particles!(particles,
                         E_y0, E_y1,
                         E_z0, E_z1,
                         H_x0, H_x1,
                         j0, j1,
                         ν_m0_grid, ν_m1_grid,
                         x_grid, x_half, τ, h,
                         ε, ν_m0_const, v_a, kI, n_a_new,
                         H0_func)
    thrust_step = 0.0
    L = x_grid[end]

    for p in particles
        p.active || continue
        z = p.z
        vy = p.vy
        vz = p.vz
        v_abs = sqrt(vy^2 + vz^2)
        N0 = max(1, ceil(Int, τ * v_abs / h))
        τ0 = τ / N0

        for i in 1:N0
            # Интерполяционные веса для целой сетки
            k0, k1, w0, w1 = interpolation_weights(z, x_grid)
            t_mid_rel = (i - 0.5) * τ0 / τ

            # Интерполяция на целой сетке
            function interp_whole(F0, F1)
                val0 = w0*F0[k0] + w1*F0[k1]
                val1 = w0*F1[k0] + w1*F1[k1]
                return (1-t_mid_rel)*val0 + t_mid_rel*val1
            end

            # Интерполяция на полуцелой сетке (H_x)
            if z <= x_half[1]
                kh0, kh1 = 1, 1
                wh0, wh1 = 1.0, 0.0
            elseif z >= x_half[end]
                kh0, kh1 = length(x_half), length(x_half)
                wh0, wh1 = 1.0, 0.0
            else
                kh0 = floor(Int, (z - x_half[1]) / h) + 1
                kh0 = clamp(kh0, 1, length(x_half)-1)
                wh1 = (z - x_half[kh0]) / h
                wh0 = 1.0 - wh1
                kh1 = kh0 + 1
            end

            function interp_half(F0, F1)
                val0 = wh0*F0[kh0] + wh1*F0[kh1]
                val1 = wh0*F1[kh0] + wh1*F1[kh1]
                return (1-t_mid_rel)*val0 + t_mid_rel*val1
            end

            E_y_mid = interp_whole(E_y0, E_y1)
            E_z_mid = interp_whole(E_z0, E_z1)
            H_mid   = interp_half(H_x0, H_x1)
            j_mid   = interp_whole(j0, j1)
            ν_m_mid = interp_whole(ν_m0_grid, ν_m1_grid)

            H_star_mid = H_mid + H0_func(z)

            # Предиктор-корректор
            vy_pred = vy + 0.5τ0 * (ε * (E_y_mid + H_star_mid*vz - ν_m_mid*j_mid))
            vz_pred = vz + 0.5τ0 * (ε * (E_z_mid - H_star_mid*vy))

            vy_new = vy + τ0 * (ε * (E_y_mid + H_star_mid*vz_pred - ν_m_mid*j_mid))
            vz_new = vz + τ0 * (ε * (E_z_mid - H_star_mid*vy_pred))

            # Защита от нечисловых значений
            if isnan(vy_new) || isnan(vz_new) || abs(vy_new) > 1e6 || abs(vz_new) > 1e6
                p.active = false
                break
            end

            z_new = z + τ0 * (vz + vz_new)/2
            z_new = clamp(z_new, x_grid[1], x_grid[end])   # удерживаем внутри области

            p.z = z_new
            p.vy = vy_new
            p.vz = vz_new
            z = z_new
            vy = vy_new
            vz = vz_new
        end

        # Граничные условия
        if p.z >= L
            p.active = false
            thrust_step += p.q * p.vz
        elseif p.z <= 0.0
            p.z = -p.z
            p.vz = -p.vz
        end
    end

    return thrust_step
end

# ------------------------------------------------------------
# Добавление новых частиц (ионизация)
# ------------------------------------------------------------
function add_new_particles!(particles, n_a_new, n_ion, x_grid, τ, kI, v_a, T_ion, h)
    for k in eachindex(x_grid)
        z = x_grid[k]
        q_new = kI * n_a_new[k] * n_ion[k] * τ * h
        if q_new > 0
            push!(particles, Particle(z, 0.0, v_a, T_ion, q_new, true))
        end
    end
end

# ------------------------------------------------------------
# Удаление неактивных частиц
# ------------------------------------------------------------
function remove_inactive_particles!(particles)
    filter!(p -> p.active, particles)
end

# ------------------------------------------------------------
# Построение графиков (аналогично рис. 8-11)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Основная функция расчёта
# ------------------------------------------------------------
function run_simulation(;
    L       = 1.0,
    M       = 100,
    N1      = 1000,
    T_ion   = 1.0,
    v_a     = 0.1,
    n_a_left= 10.0,
    kI      = 1.0,
    ε       = 1.0,
    ν_m0    = 15.0,
    γ       = 5/3,
    mi_over_mΣ = 1.0,
    H0_func = z -> 1.0,
    α       = 0.75^2,
    α0      = 367.0,
    ζ       = 0.061,
    ε_dim   = 1.0,
    total_time = 50.0,
    save_times = [10.0, 20.0, 30.0, 40.0, 50.0],
    do_plot = true
)


    h = L / M
    x_grid = range(0, L, length=M+1)
    x_half = range(h/2, L - h/2, length=M)

    # Инициализация макрочастиц
    particles = Particle[]
    q0 = L / (N1 * M)
    for k in 1:M
        z0 = x_grid[k] + h/2
        for s in 1:N1
            φ = 2π*(s-1)/N1 + π/N1 + k*sqrt(2)
            vy = 5.8 * cos(φ)
            vz = 5.8 * sin(φ)
            push!(particles, Particle(z0, vy, vz, T_ion, q0, true))
        end
    end

    n_a_old = [2*n_a_left / (2 + 1e5*z) for z in x_grid]
    H_x_half = zeros(M)
    j = zeros(M+1)
    E_y = zeros(M+1)
    E_z = zeros(M+1)
    T_e = fill(T_ion, M+1)

    n_ion = zeros(M+1)
    v_iy  = zeros(M+1)
    v_iz  = zeros(M+1)

    snapshots = Dict{Float64, Tuple{Vector{Float64},Vector{Float64},
                                    Vector{Float64},Vector{Float64},Vector{Float64}}}()
    thrust_time = Float64[]
    thrust_values = Float64[]

    t = 0.0
    step = 0

    while t < total_time
        compute_grid_quantities!(particles, x_grid, n_ion, v_iy, v_iz, T_e, h)
        smooth_field!(n_ion, 10)
        smooth_field!(v_iy, 10)
        smooth_field!(v_iz, 10)
        smooth_field!(T_e, 10)
        max_vz = max(maximum(abs.(v_iz)), 1e-12)
        τ = min(h / v_a, 0.3 * h / max_vz, total_time - t)

        compute_current!(j, H_x_half, h)

        T_tilde = similar(T_e)
        intermediate_temperature(T_tilde, T_e, n_ion, v_iz, j, n_a_old,
                                 τ, γ, mi_over_mΣ, ν_m0, kI, h)

        for p in particles
            if p.active
                k0, k1, w0, w1 = interpolation_weights(p.z, x_grid)
                p.T = w0 * T_tilde[k0] + w1 * T_tilde[k1]
            end
        end

        n_a_new = similar(n_a_old)
        update_neutrals(n_a_new, n_a_old, n_ion, τ, v_a, kI, h, n_a_left)

        # Внешнее поле считаем постоянным, поэтому H0_func(z) -> H0_val
        H0_val = H0_func(0.0)   # если поле не постоянно, нужно передавать массив
        E_y, H_x_half, j = solve_electric_field!(E_y, H_x_half, H0_val,
                                                  n_ion, v_iz, T_e, j,
                                                  τ, α, ν_m0, h, :j0)

        smooth_field!(E_y, 20)
        smooth_field!(H_x_half, 20)
        smooth_field!(j, 20)

        compute_Ez!(E_z, H_x_half, H0_val,
                    n_ion, T_e, j, v_iy, n_a_new,
                    α0, ζ, kI, ε_dim, v_a, h)
        
        ν_m_grid = ν_m0 ./ max.(T_e, 1e-12).^(3/2)
        thrust_step = move_particles!(particles,
                                      E_y, E_y,
                                      E_z, E_z,
                                      H_x_half, H_x_half,
                                      j, j,
                                      ν_m_grid, ν_m_grid,
                                      x_grid, x_half, τ, h,
                                      ε, ν_m0, v_a, kI, n_a_new,
                                      H0_func)
        push!(thrust_time, t+τ)
        push!(thrust_values, thrust_step / τ)

        add_new_particles!(particles, n_a_new, n_ion, x_grid, τ, kI, v_a, T_ion, h)
        remove_inactive_particles!(particles)

        for st in save_times
            if abs(t+τ - st) < τ/2 && !haskey(snapshots, st)
                snapshots[st] = (copy(x_grid), copy(n_a_new), copy(n_ion),
                                 copy(v_iz), copy(E_z))
            end
        end

        n_a_old .= n_a_new
        t += τ
        step += 1
        println("Step $step, t = $t, #particles = $(length(particles))")
    end

    if do_plot
        plot_results(snapshots, thrust_time, thrust_values, save_times)
    end

    return snapshots, thrust_time, thrust_values
end

# ------------------------------------------------------------
# Пример вызова для параметров из статьи (ксенон)
# ------------------------------------------------------------
let
    # Определяем функцию внешнего поля – здесь постоянное
    H0_func(z) = 1.0

    # Запускаем расчёт
    run_simulation(
        L = 1.0,
        M = 200,
        N1 = 200,
        T_ion = 1.0,
        v_a = 0.1,
        n_a_left = 10.0,
        kI = 1.0,
        ε = 1.0,
        ν_m0 = 15.0,
        γ = 5/3,
        mi_over_mΣ = 1.0,
        H0_func = H0_func,
        α = 0.75^2,
        α0 = 367.0,
        ζ = 0.061,
        ε_dim = 1.0,
        total_time = 0.2,
        save_times = [0.02, 0.05, 0.08, 0.12, 0.2],
        do_plot = true
    )
end