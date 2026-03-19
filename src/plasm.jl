module PlasmaDynamics

using LinearAlgebra
using ..PartCount
using ..NumericalFunctionsSPT

export neutrals_evolution,
       intermidiate_temperature,
       compute_current,
       electric_field_solver,
       compute_Ez

    const N_FLOOR = 1e-8
    const T_FLOOR = 1e-6
    const SMOOTHING_PASSES = 3

    """
    Уравнения для концентраций и первый шаг расчета энергии
    """
    # Изменение концентрации нейтральных частиц
    function neutrals_evolution(
        n_a_new::Vector{Float64},
        n_a_old::Vector{Float64},
        n_ion::Vector{Float64},
        τ::Float64,
        v_a::Float64,
        kI::Float64,
        h::Float64,
        n_source::Float64
    )
        M = length(n_a_old) - 1
        C = v_a * τ / h
        if C > 1.0
            @warn "Условие Куранта нарушено: v_a * τ / h = $C > 1, рекомендуется уменьшить временной шаг"
        end
        n_a_new[1] = n_source
        for i in 2:M+1
            convection = -v_a * τ * (n_a_old[i] - n_a_old[i-1]) / h
            n_a_new[i] = (n_a_old[i] + convection) / (1.0 + τ * kI * n_ion[i])
            n_a_new[i] = max(n_a_new[i], 0.0)
        end
    end

    # Первый шаг вычисления температуры (джоулев нагрев, ионизация, и расширение/сжатие электронной жидкости)
    function intermidiate_temperature(
        T_new::Vector{Float64},
        T_old::Vector{Float64},
        n::Vector{Float64},
        vz::Vector{Float64},
        j::Vector{Float64},
        n_a::Vector{Float64},
        τ::Float64,
        γ::Float64,
        mi::Float64,
        me::Float64,
        ν_m0::Float64,
        kI::Float64,
        h::Float64
    )
        M = length(T_old) - 1
        for i in 2:M
            ν_m = ν_m0 / max(T_old[i], T_FLOOR) ^ (3/2)
            dvz = (vz[i+1] - vz[i-1]) / (2*h)
            Q_collision = (γ - 1) * (mi / (mi + me)) * ν_m * j[i] ^ 2 / max(n[i], N_FLOOR)
            Q_ionisation = (γ - 1) * kI * n_a[i] * n[i]
            T_new[i] = max(T_old[i] + τ * (Q_collision + Q_ionisation - (γ - 1) * T_old[i] * dvz),
                           T_FLOOR)
        end
        T_new[1] = T_new[2]
        T_new[M+1] = T_new[M]
        return T_new
    end

    """
    Уравнения электродинамики (законы Фарадея, Ампера, и Обобщенный закон Ома)
    """
    # Решатель для тока
    function compute_current(
        j::Vector{Float64},
        H_x::Vector{Float64},
        h::Float64
    )
        M = length(H_x)
        @assert length(j) == M+1
        for i in 2:M
            j[i] = (H_x[i] - H_x[i-1]) / h   # исправлено: j[i] = ...
        end
        j[1] = j[M+1] = 0.0
        return j
    end

    # Решатель для электрического поля
    function electric_field_solver(
        E_y::Vector{Float64},   # Значение электрического поля на предыдущем временном слое
        H_x_old::Vector{Float64},   # Значения магнитного поля в полуцелых узлах
        j_old::Vector{Float64},    # Значение тока на предыдущем временном слое
        n::Vector{Float64},    # Концентрация ионов на текущем временном слое
        vz::Vector{Float64},    # Продольная скорость ионов 
        T::Vector{Float64},    # Температура электронов
        τ::Float64,    # Временной шаг
        α::Float64,    # Коэффициент обезразмеривания α = (λi/λΣ)⋅ξ²≈ξ²
        ν_m0::Float64,  # Константа в выражении для магнитной вязкости
        h::Float64,    # Шаг пространственной сетки
        x_grid::AbstractVector{Float64},    # Сетка узлов
        H0_func,    # Функция для описания магнитного поля
        bc_type::Symbol    # Тип граничных условий для тока
    )
        M = length(H_x_old)
        @assert length(E_y) == M + 1
        @assert length(j_old) == M + 1
        @assert length(n) == M + 1
        @assert length(vz) == M + 1
        @assert length(T) == M + 1

        H_interpolated = zeros(M+1)    # Число целых узлов на один больше, чем полуцелых
        H_interpolated[1] = H_x_old[1]
        for i in 2:M
            H_interpolated[i] = (H_x_old[i-1] + H_x_old[i]) / 2  # Интерполируем значения в целых узлах как среднее из соседних полуцелых
        end
        H_interpolated[M+1] = H_x_old[M]    # Берем просто значение из последнего полуцелого узла
        H_star = H_interpolated .+ H0_func(x_grid)   # Добавляем внешнее поле
        a = zeros(M+1)
        b = zeros(M+1)
        c = zeros(M+1)
        d = zeros(M+1)
        for i in 2:M
            ni = max(n[i], N_FLOOR)
            Ti = max(T[i], T_FLOOR)
            A = α / (ni * h ^ 2)
            B = (ν_m0 / Ti ^ (3/2)) * (τ / h ^ 2)
            dvz = (vz[i+1] - vz[i-1]) / (2*h)
            C = vz[i] * τ / (4h)
            D = (α * τ / (ni * h ^ 2)) * dvz

            a[i] = - A - (B + D) - C
            b[i] = 1.0 + 2A + 2(B + D)
            c[i] = - A - (B + D) + C
            dj = (j_old[i+1] - j_old[i-1]) / (2h)    # Производную плотности тока по координате
            d[i] = (ν_m0 / Ti ^ (3/2)) * j_old[i] - H_star[i] * vz[i] +
                (α / ni) * j_old[i] * dvz + (α / ni) * vz[i] * dj
        end
        # Костыль, но просто символьно определяю тип ГУ 
        # (в данном случае говорю что азимутальные токи на границах равны нулю)
        if bc_type == :j0
            # Левая граница
            dj_left = j_old[2] / h
            E_y[1] = (-H_star[1] + (α / max(n[1], N_FLOOR)) * dj_left) * vz[1]
            # Правая граница
            dj_right = -j_old[M] / h
            E_y[M+1] = (-H_star[M+1] + (α / max(n[M+1], N_FLOOR)) * dj_right) * vz[M+1]
            if M - 1 > 0
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
            a_inner = view(a, 3:M)   # a[3], a[4], ..., a[M]
            b_inner = view(b, 2:M)   # b[2], ..., b[M]
            c_inner = view(c, 2:M-1) # c[2], ..., c[M-1]
            d_inner = view(d, 2:M)   # d[2], ..., d[M]
            # Преобразуем в векторы
            a_full = collect(a_inner)
            b_full = collect(b_inner)
            c_full = collect(c_inner)
            d_full = collect(d_inner)
            E_inner = solve_tridiagonal(a_full, b_full, c_full, d_full) 
            for (idx, val) in enumerate(E_inner)
                E_y[idx+1] = val
            end
        end
        # Обновление Hx по формуле (37) (закон Фарадея)
        H_x_new = similar(H_x_old)
        for i in 1:M
            H_x_new[i] = H_x_old[i] + τ * (E_y[i+1] - E_y[i]) / h
        end
        j_new = zeros(M+1)
        compute_current(j_new, H_x_new, h)
        return E_y, H_x_new, j_new
    end

    # Функция для вычисления электрического поля вдоль оси (38)
    function compute_Ez(
        Ez::Vector{Float64},
        H_x_old::Vector{Float64},
        H_x_new::Vector{Float64},
        j_old::Vector{Float64},
        j_new::Vector{Float64},
        n::Vector{Float64},
        T::Vector{Float64},
        vy::Vector{Float64},
        n_a::Vector{Float64},
        α0::Float64,
        ζ::Float64,
        kI::Float64,
        ε_dim::Float64,
        va::Float64,
        me::Float64,
        h::Float64,
        x_grid::AbstractVector{Float64},
        H0_func,
        N_REG::Float64 = N_FLOOR,
        νE::Float64 = 0.15 # Искуственная вязкость для размазывания поля
    )
        M = length(H_x_old)
        @assert length(Ez) == M + 1
        L = x_grid[end]
        # Аппроксимация полусуммами
        H_x_mid = (H_x_new + H_x_old) / 2
        j_mid = (j_new + j_old) / 2
        # Интерполяция на целые узлы
        H_interpolation = zeros(M+1)
        H_interpolation[1] = H_x_mid[1]
        for i in 2:M
            H_interpolation[i] = (H_x_mid[i] + H_x_mid[i-1]) / 2
        end
        H_interpolation[M+1] = H_x_mid[M]
        H_star = H_interpolation + H0_func.(x_grid)
        # Сглаживание для производной
        n_s = copy(n)
        T_s = copy(T)
        Steklov_smooth(n_s, 2, h, L)
        Steklov_smooth(T_s, 2, h, L)
        
        # Вычисление производной δnT/δz
        d_nT = zeros(M+1)
        for i in 2:M
            d_nT[i] = (n_s[i+1] * T_s[i+1] - n_s[i-1] * T_s[i-1]) / (2*h)
        end
        d_nT[1] = d_nT[2]
        d_nT[M+1] = d_nT[M]

        # Защита от деления на ноль
        n_safe = max.(n, N_REG)

        for i in 1:M+1
            term1 = H_star[i] * vy[i]
            term2 = (α0 / n_safe[i]) * H_star[i] * j_mid[i]
            term3 = (ζ * α0 / n_safe[i]) * d_nT[i]
            term4 = me * (kI / ε_dim) * n_a[i] * va
            Ez[i] = term1 - term2 - term3 - term4
        end
        # Дополнительное искуственное диффузное сглаживание
        for _ in 1:SMOOTHING_PASSES
            Ez_smooth = copy(Ez)
            for i in 2:M
                Ez_smooth[i] = Ez[i] + νE * (Ez[i+1] - 2Ez[i] + Ez[i-1])
            end
            Ez .= Ez_smooth
        end
        return Ez
    end

end # module PlasmaDynamics