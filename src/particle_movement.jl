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
        n_vy::Vector{Float64},
        n_vz::Vector{Float64},
        n_T::Vector{Float64},
        h::Float64
    )

        fill!(n,0.0)
        fill!(n_vy,0.0)
        fill!(n_vz,0.0)
        fill!(n_T,0.0)

        for p in particles
            p.active || continue

            k0,k1,w0,w1 = interpolation_weights(p.z,x_grid)

            n[k0] += p.q*w0
            n[k1] += p.q*w1

            n_vy[k0] += p.q*w0*p.vy
            n_vy[k1] += p.q*w1*p.vy

            n_vz[k0] += p.q*w0*p.vz
            n_vz[k1] += p.q*w1*p.vz

            n_T[k0] += p.q*w0*p.T
            n_T[k1] += p.q*w1*p.T
        end

        for i in eachindex(n)

            vol = (i==1 || i==length(n)) ? h/2 : h

            if n[i] > N_FLOOR
                v_y[i] = n_vy[i]/n[i]
                v_z[i] = n_vz[i]/n[i]
                T[i]   = n_T[i]/n[i]
            else
                v_y[i] = 0.0
                v_z[i] = 0.0
                T[i]   = T_FLOOR
            end

            n[i] /= vol
        end

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
            τ0 = τ / N0

            # Начальный полушаг по z (лягушачий прыжок — формула препринта)
            z_half = z + τ0/2 * vz

            for i in 1:N0
                # Граничные условия на полупозиции z_half
                if z_half <= x_grid[1]
                    # Левая граница — вход в зону ускорения (со стороны зоны ионизации).
                    # Ион, вернувшийся к z=0, покидает расчётную область — поглощение.
                    counters.reflected_left += 1
                    p.active = false
                    break
                elseif z_half >= L
                    thrust_step += p.q * vz
                    counters.exited_right += 1
                    p.active = false
                    break
                end

                # Временна́я доля: t_{n+1/2}/τ = (i-0.5)/N0 (формула 40)
                t_rel = (i - 0.5) / N0

                # Интерполяция полей в z_half (целые узлы)
                k0, k1, w0, w1 = interpolation_weights(z_half, x_grid)

                # Формула (40): E_y, E_z, ν_m не зависят от t_{n+1/2}
                E_y_val = w0 * E_y0[k0] + w1 * E_y0[k1]
                E_z_val = w0 * E_z0[k0] + w1 * E_z0[k1]
                j_val   = (1 - t_rel) * (w0 * j0[k0] + w1 * j0[k1]) +
                           t_rel      * (w0 * j1[k0] + w1 * j1[k1])
                ν_m_val = w0 * ν_m0_grid0[k0] + w1 * ν_m0_grid0[k1]

                # Интерполяция H на полуцелых узлах с временно́й интерполяцией
                if z_half <= x_half[1]
                    kh, wh = 1, 1.0
                elseif z_half >= x_half[end]
                    kh, wh = length(x_half), 1.0
                else
                    kh = floor(Int, (z_half - x_half[1]) / h) + 1
                    kh = clamp(kh, 1, length(x_half)-1)
                    wh = (z_half - x_half[kh]) / h
                end
                H_now  = (kh < length(x_half)) ? (1-wh)*H_x0[kh] + wh*H_x0[kh+1] : H_x0[kh]
                H_next = (kh < length(x_half)) ? (1-wh)*H_x1[kh] + wh*H_x1[kh+1] : H_x1[kh]
                H_val  = (1 - t_rel) * H_now + t_rel * H_next
                H_star_val = H_val + H0_func(z_half)

                # Схема Кранка-Николсона для уравнений (39):
                # dvy/dt = ε*(E_z - H_*·vz)
                # dvz/dt = ε*(E_y + H_*·vy - ν_m·j)
                # Неявная схема: v^{n+1} = (I - α·A)^{-1}·(I + α·A)·v^n + τ0·(I - α·A)^{-1}·D
                # где α = τ0·ε·H*/2, A = [[0,-H*];[H*,0]], D = ε·[E_z; E_y - ν_m·j]
                α = τ0 * ε * H_star_val / 2
                det_val  = 1.0 + α^2
                E_y_eff  = E_y_val - ν_m_val * j_val   # E_y - ν_m·j
                vy_new = ((1.0 - α^2)*vy - 2*α*vz + τ0*ε*(E_z_val - α*E_y_eff)) / det_val
                vz_new = (2*α*vy + (1.0 - α^2)*vz + τ0*ε*(α*E_z_val + E_y_eff)) / det_val

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

                vy = vy_new
                vz = vz_new

                # Лягушачий прыжок: z^{n+3/2} = z^{n+1/2} + τ0·v_z^{n+1}
                if i < N0
                    z_half = z_half + τ0 * vz
                end
            end

            # Финальный полушаг: z^{N0} = z^{N0-1/2} + τ0/2·v_z^{N0}
            if p.active
                p.z  = z_half + τ0/2 * vz
                p.vy = vy
                p.vz = vz
                # Проверка границ после финального шага
                if p.z >= L
                    thrust_step += p.q * p.vz
                    counters.exited_right += 1
                    p.active = false
                elseif p.z <= 0.0
                    # Ион вернулся к z=0 — поглощается зоной ионизации
                    counters.reflected_left += 1
                    p.active = false
                end
            end
        end
        
        return thrust_step
    end

    # Появление новых ионов
    function new_particles_ionisation(
        particles::Vector{Particle},
        n_a_old::Vector{Float64},
        n_a_new::Vector{Float64},
        n_ion::Vector{Float64},
        x_grid::AbstractVector{Float64},
        τ::Float64,
        kI::Float64,
        v_a::Float64,
        T_ion::Float64,
        h::Float64,
        ionisation_factor::Float64 = 1.0
    )
        for i in 1:length(x_grid)-1   # исключаем выходной узел (z=L): ионы там сразу вылетают
            # Полуцелое среднее n_a^{1/2} = (n_a^0 + n_a^1)/2 согласно формуле шага 13 препринта
            n_a_half = (n_a_old[i] + n_a_new[i]) / 2
            q_new = ionisation_factor * kI * n_a_half * n_ion[i] * τ * h
            q_new = min(q_new, n_a_half * h)
            if q_new < MIN_PARTICLE_MASS
                continue
            end
            # Позиция точно в узле сетки z_s = (s-1)*h (согласно §3, шаг 13)
            z = x_grid[i]
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