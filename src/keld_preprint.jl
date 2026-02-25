using LinearAlgebra

mutable struct Particle
    z::Float64
    vy::Float64
    vz::Float64
    T::Float64
    q::Float64
end

function update_neutrals(
    n_a_new::Vector{Float64},
    n_a_old::Vector{Float64},
    n_ion::Vector{Float64},
    τ::Float64,
    v_a::Float64,
    kI::Float64,
    h::Float64,
    na_const::Float64
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
    T_tilde::Vector{Float64},
    T_old::Vector{Float64},
    n::Vector{Float64},
    vz::Vector{Float64},
    j::Vector{Float64},
    n_a::Vector{Float64},
    τ::Float64,
    γ::Float64,
    mi_over_mΣ::Float64,
    ν_m0::Float64,
    kI::Float64,
    h::Float64
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