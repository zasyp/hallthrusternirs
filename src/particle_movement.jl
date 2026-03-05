module ParticleMovementSPT

using ..PlasmaDynamics
using ..PartCount
using ..NumericalFunctionsSPT
using LinearAlgebra

export deposit_particles

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
    
end