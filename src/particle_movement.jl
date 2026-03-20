module ParticleMovementSPT

using ..PartCount
using ..NumericalFunctionsSPT
using LinearAlgebra

const VELOCITY_LIMIT = 1e6
const MIN_PARTICLE_MASS = 1e-8
const N_FLOOR = 1e-8          # добавлено, если не определено глобально
const T_FLOOR = 1e-6

export deposit_particles, move_particles, new_particles_ionisation, remove_inactive_particles

    # осаждение частиц на сетку
    function deposit_particles(
        particles::Vector{Particle},
        x_grid::AbstractVector{Float64},
        n::Vector{Float64},
        v_y::Vector{Float64},
        v_z::Vector{Float64},
        T::Vector{Float64},
        h::Float64
    )
        M = length(x_grid) - 1
        fill!(n, 0.0)
        n_vz = zeros(length(n))
        n_vy = zeros(length(n))
        n_T = zeros(length(n))

        for p in particles
            p.active || continue

            k0, k1, w0, w1 = interpolation_weights(p.z, x_grid)
            n[k0] += p.q * w0
            n[k1] += p.q * w1

            n_vy[k0] += p.q * w0 * p.vy
            n_vy[k1] += p.q * w1 * p.vy

            n_vz[k0] += p.q * w0 * p.vz
            n_vz[k1] += p.q * w1 * p.vz

            n_T[k0] += p.q * w0 * p.T
            n_T[k1] += p.q * w1 * p.T
        end
        # Нормировка
        for i in eachindex(n)
            vol = (i == 1 || i == length(n)) ? h/2 : h
            if n[i] > N_FLOOR
                v_y[i] = n_vy[i] / n[i]
                v_z[i] = n_vz[i] / n[i]
                T[i] = n_T[i] / n[i]
            else
                v_y[i] = 0.0
                v_z[i] = 0.0
                T[i] = T_FLOOR
            end
        
            n[i] /= vol
        end
        return n, v_y, v_z, T
    end

    function move_particles(
        particles::Vector{Particle},
        E_y0::Vector{Float64},
        E_y1::Vector{Float64},
        E_z0::Vector{Float64},
        E_z1::Vector{Float64},
        H_x0::Vector{Float64},
        H_x1::Vector{Float64},
        j0::Vector{Float64},
        j1::Vector{Float64},
        ν_m0_grid0::Vector{Float64}, # Частоты кулоновских столкновений
        ν_m0_grid1::Vector{Float64},
        x_grid::AbstractVector{Float64},
        x_half::AbstractVector{Float64},
        τ::Float64,
        h::Float64,
        ε::Float64,
        H0_func::Function,
        counters::Counters
    )
        thrust_step = 0.0
        L = x_grid[end]

        for p in particles
             p.active || continue
            z = p.z
            vy = p.vy
            vz = p.vz
            v_abs = sqrt(vy^2 + vz^2)            
            N0 = max(1, ceil(Int, τ * v_abs / (0.25 * h)))
            # частица за один подшаг не может переместиться больше, чем на долю размера ячейки
            τ0 = τ / N0
            for i in 1:N0
                if z < x_grid[1]
                    z = x_grid[1]
                    vz = abs(vz)
                elseif z > x_grid[end]
                    z = x_grid[end]
                end
                t_mid_relative = (i - 0.5) / N0   # исправлено имя переменной
                # Целые узлы
                k0, k1, w0, w1 = interpolation_weights(z, x_grid)  # исправлено имя функции
                function interpolate_field(
                    F0::Vector{Float64},
                    F1::Vector{Float64}
                )
                    val_now = w0 * F0[k0] + w1 * F0[k1]
                    val_next = w0 * F1[k0] + w1 * F1[k1]
                    return (1 - t_mid_relative) * val_now + t_mid_relative * val_next
                end
                E_y_mid = interpolate_field(E_y0, E_y1)
                E_z_mid = interpolate_field(E_z0, E_z1)
                j_mid = interpolate_field(j0, j1)
                ν_m_mid = interpolate_field(ν_m0_grid0, ν_m0_grid1)

                # Полуцелые узлы
                if z <= x_half[1]
                    kh, wh = 1, 1.0 # kh - номер левого узла, wh - относительное расстояние от узла до частицы (вес правого узла, вес левого = 1 - wh)
                elseif z >= x_half[end]
                    kh, wh = length(x_half), 1.0
                else
                    kh = floor(Int, (z - x_half[1]) / h) + 1
                    kh = clamp(kh, 1, length(x_half)-1)
                    wh = (z - x_half[kh]) / h
                end
                # Билинейная (по пространству и времени) интерполяция значений поля H
                H_now = (kh < length(x_half)) ? (1-wh)*H_x0[kh] + wh*H_x0[kh+1] : H_x0[kh]   # исправлено: второй индекс kh+1
                H_next = (kh < length(x_half)) ? (1-wh)*H_x1[kh] + wh*H_x1[kh+1] : H_x1[kh]
                H_mid = (1 - t_mid_relative) * H_now + t_mid_relative * H_next   # исправлено имя переменной

                H_star_mid = H_mid + H0_func(z)
                
                #Расчет скоростей по схеме предиктор-корректор
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

    # Появление новых ионов
    function new_particles_ionisation(
        particles::Vector{Particle},
        n_a_new::Vector{Float64},
        n_ion::Vector{Float64},
        x_grid::AbstractVector{Float64},
        τ::Float64,
        kI::Float64,
        v_a::Float64,           # исправлено: число, не вектор
        T_ion::Float64,         # исправлено: число, не вектор
        h::Float64,
        ionisation_factor::Float64 = 1.0  # Плавное включение ионизации (ДОБАВЛЕНО)
    )
        for i in 1:length(x_grid)-1   # исключаем выходной узел (z=L): ионы там сразу вылетают
            # Полное число частиц рождающихся в ячейке за временной шаг
            # Умножаем на ionisation_factor для плавного включения в начале
            q_new = ionisation_factor * kI * n_a_new[i] * n_ion[i] * τ * h
            q_new = min(q_new, n_a_new[i] * h) # Их не должно быть больше, чем нейтралов в этой ячейке
            if q_new < MIN_PARTICLE_MASS
                continue
            end
            # Высадка частицы в случайном месте в ячейке
            z = x_grid[i] + h * (rand() - 0.5)   # исправлено: x_grid[i]
            z = clamp(z, x_grid[1], x_grid[end])
            # Добавление частицы в массив частиц
            push!(particles, Particle(z, 0.0, v_a, T_ion, q_new, true))
        end
    end

    # Удаление частиц вылетевших за пределы системы или рекомбинировавших (так как ионы описываются частицами, а нейтральные частицы в модели описываются уравнениями переноса)
    function remove_inactive_particles(
        particles::Vector{Particle},   # исправлено: Vector{Particle}
        L::Float64,
        τ::Float64,
        kR::Float64
    )
        for p in particles
            if p.z < 0 || p.z > L
                p.active = false
                continue
            end
            P_rec = min(1.0, kR * p.q * τ)
            rand() < P_rec && (p.active = false)
        end
        filter!(p -> p.active, particles)
        return particles
    end

end