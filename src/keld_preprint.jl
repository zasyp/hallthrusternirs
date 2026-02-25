using LinearAlgebra

mutable struct Particle
    z::Float64
    vy::Float64
    vz::Float64
    T::Float64
    q::Float64
end

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

function update_neutrals(
    n_a_new::Vector{Float64}, n_a_old::Vector{Float64},
    n_ion::Vector{Float64}, τ::Float64,
    v_a::Float64, kI::Float64,
    h::Float64, na_const::Float64
)
    # Check Courant condition
    if τ <= h /v_a
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
    else
        error("Time step τ must satisfy the Courant condition: τ <= h / v_a")
    end
end

function intermidiate_temperature(
    T_tilde::Vector{Float64}, T_old::Vector{Float64},
    n::Vector{Float64}, vz::Vector{Float64},
    j::Vector{Float64}, n_a::Vector{Float64},
    τ::Float64, γ::Float64,
    mi_over_mΣ::Float64, ν_m0::Float64,
    kI::Float64, h::Float64
)
    M = length(T_old) - 1
    vz_left = 2vz[1] - vz[1]
    vz_right = 2vz[M+1] - vz[M]
    for k in 1:M+1
        if k == 1
            dvz = (vz[2] - vz_left) / (2.0 * h)
        elseif k == M+1
            dvz = (vz_right - vz[M]) / (2.0 * h)
        else
            dvz = (vz[k+1] - vz[k-1]) / (2.0 * h)
        end
        joule_heating = (γ - 1.0) * mi_over_mΣ * (ν_m0 / (T_old[k]^(3/2))) * (j[k]^2) / n[k]
        ioniz = (γ - 1.0) * kI * T_old[k] * n_a[k]
        compression = -(γ - 1.0) * T_safe * dvz
        T_tilde[k] = T_old[k] + τ * (compression + joule_heating + ioniz)
    end
    return T_tilde
end

function coumpute_current(
    j::Vector{Float64}, Hx::Vector{Float64}, h::Float64
)
    M = length(H_x)
    for k in 2:M
        j[k] = (Hx[k] - Hx[k-1]) / h
    end
    j[1] = 0.0
    j[M+1] = 0.0
    return j
end

function solve_electric_field!(
    E_y::Vector{Float64}, H_x::Vector{Float64}, H0::Float64,
    n::Vector{Float64}, vz::Vector{Float64}, T::Vector{Float64},
    j_old::Vector{Float64}, τ::Float64, α::Float64,
    ν_m0::Float64, h::Float64, bc_type::Symbol
)
    M = length(H_x)            # число полуцелых узлов = число ячеек
    @assert length(E_y) == M+1
    @assert length(n) == M+1 && length(vz) == M+1 && length(T) == M+1 && length(j_old) == M+1
    # Трёхдиагональная матрица для внутренних узлов (2..M)
    a = zeros(M+1)   # поддиагональ (связь с предыдущим)
    b = zeros(M+1)   # главная диагональ
    c = zeros(M+1)   # наддиагональ (связь со следующим)
    d = zeros(M+1)   # правая часть
    # Предварительно интерполируем H_x на целые узлы для вычисления H*
    H_int = zeros(M+1)
    H_int[1] = H_x[1]               # на левой границе берём ближайший полуцелый
    for k in 2:M
        H_int[k] = (H_x[k-1] + H_x[k]) / 2
    end
    H_int[M+1] = H_x[M]              # на правой границе
    for k in 2:M
        n_safe = max(n[k], 1e-12)
        T_safe = max(T[k], 1e-12)
        if n_safe > 0 && T_safe > 0
            coeff = α / (n_safe * h^2)
            a[k] = -coeff
            b[k] = 1.0 + 2.0 * coeff
            c[k] = -coeff
            # Вычисляем правую часть на основе известных величин с нижнего слоя
            H_star_k = H_int[k] + H0
            dvz = (vz[k+1] - vz[k-1]) / (2.0 * h)
            dj = (j_old[k+1] - j_old[k-1]) / (2.0 * h)
            d[k] = (ν_m0 / (T_safe^(3/2))) * j_old[k] - H_star_k * vz[k] +
                   (α * j_old[k] / n_safe) * dvz + (α * vz[k] / n_safe) * dj
        else
            # Если плазмы нет, поле не определено, положим E_y = 0
            b[k] = 1.0
            d[k] = 0.0
        end
    end
    # Граничные условия (j = 0 на торцах)
    if bc_type == :j0
        # Левая граница (k=1)
        if n[1] > 1e-12
            H_star_1 = H_int[1] + H0
            dj_left = j_old[2] / h          # аппроксимация ∂j/∂z при j=0 на границе
            E_y[1] = -H_star_1 * vz[1] + (α / n[1]) * dj_left * vz[1]
        else
            E_y[1] = 0.0
        end
        # Правая граница (k=M+1)
        if n[M+1] > 1e-12
            H_star_end = H_int[M+1] + H0
            dj_right = -j_old[M] / h        # аппроксимация ∂j/∂z при j=0 на правой границе
            E_y[M+1] = -H_star_end * vz[M+1] + (α / n[M+1]) * dj_right * vz[M+1]
        else
            E_y[M+1] = 0.0
        end
        # Исключаем граничные узлы из системы для внутренних
        d[2] -= a[2] * E_y[1]
        a[2] = 0.0
        d[M] -= c[M] * E_y[M+1]
        c[M] = 0.0
    else
        error("Неподдерживаемый тип граничных условий: $bc_type")
    end
    # Решение трёхдиагональной системы для узлов 2..M
    a_inner = a[2:M]
    b_inner = b[2:M]
    c_inner = c[2:M]
    d_inner = d[2:M]
    E_inner = solve_tridiagonal(a_inner, b_inner, c_inner, d_inner)
    for (idx, val) in enumerate(E_inner)
        E_y[idx+1] = val
    end
    # Обновление H_x по закону Фарадея (явная схема)
    H_x_new = similar(H_x)
    for k in 1:M
        H_x_new[k] = H_x[k] + τ * (E_y[k+1] - E_y[k]) / h
    end
    # Обновление j по новому H_x
    j_new = zeros(M+1)
    compute_current!(j_new, H_x_new, h)
    return E_y, H_x_new, j_new
end

function compute_Ez!(
    Ez::Vector{Float64}, H_x::Vector{Float64}, H0::Float64,
    n::Vector{Float64}, T::Vector{Float64}, j::Vector{Float64},
    vy::Vector{Float64}, n_a::Vector{Float64}, α0::Float64,
    ζ::Float64, kI::Float64, ε_dim::Float64,
    va::Float64, h::Float64
)
    M = length(H_x)   # число полуцелых узлов
    @assert length(Ez) == M+1 && length(n) == M+1 && length(T) == M+1
    @assert length(j) == M+1 && length(vy) == M+1 && length(n_a) == M+1
    # Интерполируем H_x на целые узлы
    H_int = zeros(M+1)
    H_int[1] = H_x[1]
    for k in 2:M
        H_int[k] = (H_x[k-1] + H_x[k]) / 2
    end
    H_int[M+1] = H_x[M]
    H_star = H_int .+ H0
    # Вычисляем производную ∂(nT)/∂z (центральные разности)
    d_nT = zeros(M+1)
    for k in 2:M
        d_nT[k] = (n[k+1]*T[k+1] - n[k-1]*T[k-1]) / (2.0 * h)
    end
    # Экстраполяция на границы
    d_nT[1] = d_nT[2]
    d_nT[M+1] = d_nT[M]
    for k in 1:M+1
        n_safe = max(n[k], 1e-12)
        # Формула (38)
        Ez[k] = H_star[k] * vy[k] - (α0 / n_safe) * H_star[k] * j[k] -
                (ζ * α0 / n_safe) * d_nT[k] - (kI / ε_dim) * n_a[k] * va
    end
    return Ez
end