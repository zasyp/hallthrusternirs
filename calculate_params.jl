# Расчет безразмерных параметров в системе СГС

function calculate_sim_params(
    H::Float64,
    L::Float64,
    v::Float64,
    n::Float64,
    σ0::Float64,
    β0::Float64,
    mi::Float64,
    Te::Float64,
    T::Float64
)
    E = H * v / 3e10
    t = L / v
    j = 3e10 * H / (4 * π * L)
    f = n / (v^2)
    ρ = mi * n

    va = H / sqrt(4 * π * ρ)
    ωci = 4.8e-10 * H / (3e10 * mi)
    ϵ = t * ωci
    ν_m0 = (3e10) ^ 2 / (4 * π * σ0 * T ^ (3/2) * L * v)
    ν_m = ν_m0 / Te
    kI = β0 * n * t
    λe = 9.1e-28 / (4.8e-10)
    λi = mi / (4.8e-10)
    ξ = (3e10 * sqrt(λe * λi)) / (L * (4 * π * ρ) ^ 0.5)
    κ = va / v
    ζ = 1.38e-16 * T / (mi * va ^ 2)

    force_scale = mi * n * (L^3) * v / (100 * t)  # для силы    (В милиньютонах)
    return [E, t, j, f, ρ, va, ωci, ϵ, ν_m0, ν_m, kI, ξ, κ, ζ, force_scale]

end

sim_params_Kr_SPT70 = calculate_sim_params(
    190.0,  # H
    4.0, # L
    3e4,  # v
    3.62e19 * 1e-6, # n
    1.0e7, # σ0
    2.0e-8,  # β0
    1.39e-25 * 1e3, # mi (масса иона криптона)
    20.0 * 11600, # Te (электронная температура в эВ)
    14.0 * 11600   # T (температура ионизации в эВ)
)

println(sim_params_Kr_SPT70)


