# Основной файл, в котором выполняется главный цикл программы, а также построение графиков
"""
В цикле по времени:
    Осаждение частиц (deposit_particles)
    Сглаживание (smooth_field, но они в NumericalFunctionsSPT)
    Вычисление шага τ
    Вычисление тока (compute_current)
    Промежуточная температура (intermidiate_temperature)
    Отемпературивание частиц (присвоение p.T из интерполяции)
    Обновление нейтралов (neutrals_evolution)
    Сохранение старых полей
    Решение для Ey (electric_field_solver)
    Сглаживание полей (Steklov_smooth)
    Вычисление Ez (compute_Ez)
    Движение частиц (move_particles)
    Добавление новых частиц (new_particles_ionisation)
    Удаление неактивных (remove_inactive_particles)
    Сохранение снимков
    Вывод диагностики
"""

using LinearAlgebra
using Plots

# Подключение модулей (предполагается, что они находятся в той же директории)
include("src/structs.jl")
include("src/numerical_funcs.jl")
include("src/plasm.jl")
include("src/particle_movement.jl")
include("src/plotting.jl")

using .PartCount
using .NumericalFunctionsSPT
using .PlasmaDynamics
using .ParticleMovementSPT
using .Visualization

"""
    run_simulation(params::SimParams; total_time, save_times, do_plot)

Запускает гибридное моделирование СПД с заданными параметрами.
Возвращает словарь снимков и массивы времени и силы тяги.
"""
function run_simulation(params::SimParams; total_time=30.0, save_times=[10.0,20.0,30.0], do_plot=true)
    # Извлечение параметров из структуры
    L = params.L
    M = params.M
    h = params.h
    mi = params.mi
    me = params.me
    T_ion = params.T_ion
    v_a = params.v_a
    n_a_left = params.n_a_left
    kI = params.kI
    kR = params.kR
    γ = params.γ
    ε = params.ε
    ν_m0 = params.ν_m0
    α = params.α
    α0 = params.α0
    ζ = params.ζ
    ε_dim = params.ε_dim
    H0_func = params.H0_func
    N1 = params.N1

    # Сетки
    x_grid = range(0, L, length=M+1)       # целые узлы
    x_half = range(h/2, L-h/2, length=M)   # полуцелые узлы

    # Инициализация макрочастиц (начальное распределение)
    particles = Particle[]
    q0 = L / (N1 * M)                       # вес одной макрочастицы
    for k in 1:M
        z0 = x_grid[k] + h/2                 # центр ячейки
        for s in 1:N1
            φ = 2π*(s-1)/N1 + π/N1 + k*sqrt(2)
            vy = 5.8 * cos(φ)
            vz = 5.8 * sin(φ)
            push!(particles, Particle(z0, vy, vz, T_ion, q0, true))
        end
    end

    # Начальное распределение нейтралов (быстро спадающее)
    n_a_old = n_a_left * exp.(-5 * x_grid / L)

    # Поля
    H_x_half = zeros(M)          # магнитное поле на полуцелых
    j = zeros(M+1)               # ток на целых
    E_y = zeros(M+1)             # азимутальное поле
    E_z = zeros(M+1)             # продольное поле
    T_e = fill(T_ion, M+1)       # температура электронов на сетке

    # Гидродинамические величины на целых
    n_ion = zeros(M+1)           # концентрация ионов
    v_iy = zeros(M+1)            # азимутальная скорость ионов
    v_iz = zeros(M+1)            # продольная скорость ионов

    # Для хранения старых полей (интерполяция по времени)
    H_x_old = copy(H_x_half)
    j_old = copy(j)
    E_y_old = copy(E_y)
    E_z_old = copy(E_z)

    # Снимки и диагностика
    snapshots = Dict{Float64, Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}}()
    thrust_time = Float64[]
    thrust_values = Float64[]

    t = 0.0
    step = 0

    while t < total_time
        # 1. Осаждение частиц на сетку (получение n_ion, v_iy, v_iz, T_e)
        deposit_particles(particles, x_grid, n_ion, v_iy, v_iz, T_e, h)

        # Сглаживание гидродинамических величин для устойчивости
        smooth_field(n_ion, 4)
        smooth_field(v_iy, 4)
        smooth_field(v_iz, 4)
        smooth_field(T_e, 4)

        # 2. Определение шага по времени с учётом условий Куранта
        max_vz = max(maximum(abs.(v_iz)), 1e-12)
        ν_m_grid = ν_m0 ./ max.(T_e, PlasmaDynamics.T_FLOOR).^(3/2)
        ν_max = maximum(ν_m_grid)
        τ_coll = 0.5 / max(ν_max, 1e-8)
        τ = min(h / v_a, 0.2 * h / max_vz, τ_coll, total_time - t)

        # 3. Вычисление тока из H на старом слое
        compute_current(j, H_x_half, h)

        # 4. Промежуточная температура (первый этап, без переноса)
        T_tilde = similar(T_e)
        intermidiate_temperature(T_tilde, T_e, n_ion, v_iz, j, n_a_old, τ, γ, mi, me, ν_m0, kI, h)

        # 5. "Отемпературивание" частиц (интерполяция температуры на частицы)
        for p in particles
            p.active || continue
            k0, k1, w0, w1 = interpolation_weights(p.z, x_grid)
            p.T = w0 * T_tilde[k0] + w1 * T_tilde[k1]
        end

        # 6. Обновление концентрации нейтралов
        n_a_new = similar(n_a_old)
        neutrals_evolution(n_a_new, n_a_old, n_ion, τ, v_a, kI, h, n_a_left)

        # 7. Сохранение старых полей для временной интерполяции
        copyto!(H_x_old, H_x_half)
        copyto!(j_old, j)
        copyto!(E_y_old, E_y)
        copyto!(E_z_old, E_z)

        # 8. Решение для азимутального поля E_y (обобщённый закон Ома)
        E_y, H_x_half, j = electric_field_solver(E_y, H_x_old, j_old, n_ion, v_iz, T_e,
                                                  τ, α, ν_m0, h, x_grid, H0_func, :j0)

        # Сглаживание полей по методу Стеклова для подавления шума
        Steklov_smooth(E_y, 10, h, L, 10)
        Steklov_smooth(H_x_half, 10, h, L, 10)
        Steklov_smooth(j, 10, h, L, 10)

        # 9. Вычисление продольного поля E_z (явная формула)
        compute_Ez(E_z, H_x_old, H_x_half, j_old, j, n_ion, T_e, v_iy, n_a_new,
                   α0, ζ, kI, ε_dim, v_a, h, x_grid, H0_func)

        # Дополнительное сглаживание E_z
        Steklov_smooth(E_z, 10, h, L, 10)

        # 10. Движение макрочастиц под действием полей
        counters = Counters(0, 0, 0, 0)
        ν_m_grid = ν_m0 ./ max.(T_e, PlasmaDynamics.T_FLOOR).^(3/2)   # частота столкновений на новом слое
        thrust_step = move_particles(particles,
                                     E_y_old, E_y,
                                     E_z_old, E_z,
                                     H_x_old, H_x_half,
                                     j_old, j,
                                     ν_m_grid, ν_m_grid,   # старый и новый слои (пока одинаковы)
                                     x_grid, x_half, τ, h, ε,
                                     H0_func, counters)

        push!(thrust_time, t+τ)
        push!(thrust_values, thrust_step / τ)
        #Сглаживание тяги
        Steklov_smooth(thrust_values, 3, h, L, 5)
        # 11. Добавление новых частиц от ионизации
        new_particles_ionisation(particles, n_a_new, n_ion, x_grid, τ, kI, v_a, T_ion, h)

        # 12. Удаление частиц, покинувших область или рекомбинировавших
        remove_inactive_particles(particles, L, τ, kR)

        # 13. Сохранение снимков в заданные моменты времени
        for st in save_times
            if abs(t+τ - st) < τ/2 && !haskey(snapshots, st)
                snapshots[st] = (copy(x_grid), copy(n_a_new), copy(n_ion), copy(v_iz), copy(E_z))
            end
        end

        # 14. Подготовка к следующему шагу
        n_a_old .= n_a_new
        t += τ
        step += 1

        # Вывод диагностической информации
        println("Step $step, t=$t, #particles=$(length(particles)), ",
                "min_n=$(minimum(n_ion)), max_Ez=$(maximum(abs.(E_z))), ",
                "nan=$(counters.nan), overspeed=$(counters.overspeed), ",
                "exited=$(counters.exited_right), reflected=$(counters.reflected_left)")
    end

    # Построение графиков по окончании расчёта
    if do_plot
        plot_results(snapshots, thrust_time, thrust_values, save_times)
    end

    return snapshots, thrust_time, thrust_values
end

# -------------------------------------------------------------------
# Пример запуска с параметрами из статьи (случай с индукционными полями)
# -------------------------------------------------------------------
let
    params = PartCount.SimParams(
        L=2.5, M=100,
        mi=1.0, me=1/1836,                     # отношение масс для ксенона
        T_ion=1.0, v_a=1, n_a_left=10.0, kI=1.1, kR=0.1,
        γ=5/3, ε=1.0, ν_m0=15.0,
        α=0.283, α0=22.0, ζ=0.061, ε_dim=1.0,
        H0_func=z->1.9,                         # внешнее магнитное поле (константа)
        N1=100                                   # число макрочастиц на ячейку
    )
    run_simulation(params, total_time=20.0, save_times=[5.0,10.0, 15.0, 20.0], do_plot=true)
end 