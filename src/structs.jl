module PartCount

export Particle, Counters, SimParams
    mutable struct Particle
        z::Float64
        vy::Float64
        vz::Float64
        T::Float64
        q::Float64
        active::Bool
    end

    mutable struct Counters
        nan::Int
        overspeed::Int
        exited_right::Int
        reflected_left::Int
    end

    struct SimParams
        # Геометрия
        L::Float64
        M::Int
        h::Float64

        # Массы частиц (можно в безразмерных единицах, где характерная масса = mi?)
        mi::Float64   # масса иона
        me::Float64   # масса электрона
        T_ion::Float64
        v_a::Float64
        n_a_left::Float64
        kI::Float64
        kR::Float64
        γ::Float64
        ε::Float64
        ν_m0::Float64
        α::Float64
        α0::Float64
        ζ::Float64
        ε_dim::Float64
        H0_func::Function
        N1::Int
    end

    function SimParams(; L, M, mi, me, T_ion, v_a, n_a_left, kI, kR, γ, ε, ν_m0, α, α0, ζ, ε_dim, H0_func, N1)
        h = L / M
        return SimParams(L, M, h, mi, me, T_ion, v_a, n_a_left, kI, kR, γ, ε, ν_m0, α, α0, ζ, ε_dim, H0_func, N1)
    end


end