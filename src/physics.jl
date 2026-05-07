module PartCount

export Particle, Counters, SimParams

"""
    Particle(z, vy, vz, T, q, active)

A single PIC macroparticle on a 1D-3V mesh:
- `z` — axial position;
- `vy`, `vz` — azimuthal and axial velocities;
- `T` — local electron temperature interpolated to the particle (used for diagnostics);
- `q` — macroparticle weight (sum equals total dimensionless charge / channel length);
- `active` — soft kill flag (compacted by `remove_inactive_particles`).
"""
mutable struct Particle
    z::Float64
    vy::Float64
    vz::Float64
    T::Float64
    q::Float64
    active::Bool
end

"""
    Counters(nan, exited_right, reflected_left)

Per-step diagnostic counters incremented inside `move_particles`:
- `nan` — particles killed because the pusher produced a non-finite velocity;
- `exited_right` — ions that crossed the anode plane (right boundary);
- `reflected_left` — ions absorbed at the left wall (reinjected as neutrals through the
  inlet boundary condition `n_a(0)` and the ionisation source).
"""
mutable struct Counters
    nan::Int
    exited_right::Int
    reflected_left::Int
end

"""
    SimParams(; L, M, mi, me, T_ion, v_a, n_a_left, kI, kR, γ, ε, ν_m0,
              α, α0, ζ, ε_dim, λ_e_λΣ, c_inv=1, H0_func, N1, v_pic0=1,
              collision_model=:spitzer, alpha_B=0, E_z0_dimless=0,
              pic_charge_factor=1, include_self_B=true)

Dimensionless parameter container used by every solver routine. All variables are in the
paper-§2 / §3 normalisation (`[L]=L_m`, `[v]=v_ref`, `[t]=L/v`, etc.):

- `L, M, h = L/M` — channel length, cells, cell width.
- `mi, me` — ion / electron masses (typically `mi=1`, `me=m_e/m_i`).
- `T_ion`, `v_a`, `n_a_left` — initial neutral temperature, axial neutral speed, inlet density.
- `kI`, `kR` — dimensionless ionisation / recombination Damköhler numbers
  (`kI = β₀ n L / v`, `kR = ν_R L / v`).
- `γ` — adiabatic index used in `intermediate_temperature`.
- `ε` — dimensionless time scale `ω_ci · t_char` (paper Eq. (11)).
- `ν_m0` — Spitzer prefactor, `ν_m = ν_m0 / T^{3/2}` (or constant if `collision_model=:constant`).
- `α, α0, ζ` — closure coefficients in Ohm’s law and Eq. (38), see paper §2.
- `ε_dim` — kept for compatibility with auxiliary scripts (not used in the core push).
- `λ_e_λΣ = m_e/(m_i+m_e)` — appears in the ionisation drag of Eq. (38).
- `c_inv` — inverse dimensionless light speed (Alfvén normalisation: 1).
- `H0_func(z)` — analytic external magnetic profile.
- `N1` — initial macroparticles per cell.
- `v_pic0` — initial PIC velocity scale (set from `v_th,ion`).
- `collision_model ∈ (:spitzer, :constant)`.
- `alpha_B` — Bohm anomaly coefficient (0 disables anomalous part of `ν_m`).
- `E_z0_dimless` — uniform axial bias added to `E_z` after Eq. (38).
- `pic_charge_factor` — global rescaling of macroparticle weights.
- `include_self_B` — default for Faraday accumulation when `accumulate_induced_H` is
  not explicitly passed to `run_simulation`.
"""
struct SimParams
    L::Float64
    M::Int
    h::Float64
    mi::Float64
    me::Float64
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
    λ_e_λΣ::Float64
    c_inv::Float64
    H0_func::Function
    N1::Int
    v_pic0::Float64
    collision_model::Symbol
    alpha_B::Float64
    E_z0_dimless::Float64
    pic_charge_factor::Float64
    include_self_B::Bool
end

function SimParams(;
    L,
    M,
    mi,
    me,
    T_ion,
    v_a,
    n_a_left,
    kI,
    kR,
    γ,
    ε,
    ν_m0,
    α,
    α0,
    ζ,
    ε_dim,
    λ_e_λΣ,
    c_inv = 1.0,
    H0_func,
    N1,
    v_pic0 = 1.0,
    collision_model::Symbol = :spitzer,
    alpha_B::Float64 = 0.0,
    E_z0_dimless::Float64 = 0.0,
    pic_charge_factor::Float64 = 1.0,
    include_self_B::Bool = true,
)
    h = L / M
    collision_model in (:spitzer, :constant) ||
        throw(ArgumentError("collision_model must be :spitzer or :constant (got $collision_model)"))
    return SimParams(
        L, M, h, mi, me, T_ion, v_a, n_a_left, kI, kR, γ, ε, ν_m0, α, α0, ζ, ε_dim,
        λ_e_λΣ, c_inv, H0_func, N1, Float64(v_pic0), collision_model, Float64(alpha_B),
        Float64(E_z0_dimless), Float64(pic_charge_factor), include_self_B,
    )
end

end

using LinearAlgebra

module MagneticField

export gaussian_Br, gaussian_dBr_dz, normalized_gradient_Br
export lorentzian_Br, lorentzian_dBr_dz
export sech2_Br, sech2_dBr_dz
export compute_Br_profile, estimate_LB

"""
    gaussian_Br(z, B_max, z0, σ) -> Float64

Gaussian radial-field amplitude `B(z) = B_max · exp(-(z-z0)² / (2σ²))`.
Default `(B_max, z0, σ) = (0.02, 0.7, 0.2)` (dimensionless).
"""
function gaussian_Br(z::Float64, B_max::Float64 = 0.02, z0::Float64 = 0.7, σ::Float64 = 0.2)
    return B_max * exp(-(z - z0)^2 / (2 * σ^2))
end

"""Derivative `dB/dz` of `gaussian_Br`."""
function gaussian_dBr_dz(z::Float64, B_max::Float64, z0::Float64, σ::Float64)
    return -(z - z0) / σ^2 * gaussian_Br(z, B_max, z0, σ)
end

"""Normalised gradient `(1/B_max) dB/dz` of the Gaussian profile."""
function normalized_gradient_Br(z::Float64, B_max::Float64, z0::Float64, σ::Float64)
    return gaussian_dBr_dz(z, B_max, z0, σ) / B_max
end

"""Lorentzian field `B(z) = B_max / (1 + ((z-z0)/w)²)`."""
function lorentzian_Br(z::Float64, B_max::Float64, z0::Float64, w::Float64)
    return B_max / (1.0 + ((z - z0) / w)^2)
end

"""Derivative `dB/dz` of `lorentzian_Br`."""
function lorentzian_dBr_dz(z::Float64, B_max::Float64, z0::Float64, w::Float64)
    u = (z - z0) / w
    return -2.0 * B_max * u / (w * (1.0 + u^2)^2)
end

"""Sech² field `B(z) = B_max · sech²((z-z0)/w)`."""
function sech2_Br(z::Float64, B_max::Float64, z0::Float64, w::Float64)
    return B_max * sech((z - z0) / w)^2
end

"""Derivative `dB/dz` of `sech2_Br`."""
function sech2_dBr_dz(z::Float64, B_max::Float64, z0::Float64, w::Float64)
    u = (z - z0) / w
    return -2.0 * B_max * sech(u)^2 * tanh(u) / w
end

"""
    compute_Br_profile(z_grid, B_max, z0, σ; model=:gaussian) -> NamedTuple

Evaluate the chosen analytic field model on `z_grid` and return
`(; Br, dBr_dz, grad_Br = dBr/B_max)`. Model selector: `:gaussian` (default),
`:lorentzian`, `:sech2`. Width parameter is `σ` for Gaussian, `w` for the others
(supplied through the same positional argument).
"""
function compute_Br_profile(
    z_grid::Vector{Float64},
    B_max::Float64,
    z0::Float64,
    σ::Float64;
    model::Symbol = :gaussian,
)
    N = length(z_grid)
    Br = zeros(N)
    dBr = zeros(N)
    grad_Br = zeros(N)
    if model == :gaussian
        Br_func = z -> gaussian_Br(z, B_max, z0, σ)
        dBr_func = z -> gaussian_dBr_dz(z, B_max, z0, σ)
    elseif model == :lorentzian
        Br_func = z -> lorentzian_Br(z, B_max, z0, σ)
        dBr_func = z -> lorentzian_dBr_dz(z, B_max, z0, σ)
    elseif model == :sech2
        Br_func = z -> sech2_Br(z, B_max, z0, σ)
        dBr_func = z -> sech2_dBr_dz(z, B_max, z0, σ)
    else
        error("Unknown model: $model. Use :gaussian, :lorentzian, or :sech2")
    end
    for i in 1:N
        z = z_grid[i]
        Br[i] = Br_func(z)
        dBr[i] = dBr_func(z)
        grad_Br[i] = dBr[i] / B_max
    end
    return (; Br, dBr_dz = dBr, grad_Br)
end

"""
    estimate_LB(σ; model=:gaussian) -> Float64

Characteristic gradient length `L_B = max|B| / max|dB/dz|` of the chosen profile,
expressed via the width parameter `σ` (or `w` for non-Gaussian models). Used as a
rough order-of-magnitude reference, not by the solver itself.
"""
function estimate_LB(σ::Float64; model::Symbol = :gaussian)
    if model == :gaussian
        return σ * sqrt(ℯ)
    elseif model == :lorentzian
        return σ * (4.0 / (3.0 * sqrt(3.0)))^(-1)
    elseif model == :sech2
        return σ / tanh(1.0)
    else
        error("Unknown model: $model")
    end
end

end

module PlasmaDynamics
# 1D2V planar hybrid EMHD, system (11) in the paper: ions/neutrals in z with (v_y,v_z) in velocity space (y = Hall/azimuthal);
# electron fluid: T_e, σ∝T_e^{3/2}, E_y elliptic, E_z algebraic; v_{iy},v_{iz} are ion moments, v_{ey}=v_{iy}-j/n is electron azimuthal drift.

using LinearAlgebra
using ..PartCount
using ..NumericalFunctionsSPT

export neutrals_evolution, intermediate_temperature, intermidiate_temperature,
    compute_current, electric_field_solver, compute_Ez, local_nu_m, H_star_channel_nonneg

const N_FLOOR = 1e-8
# Floor on T for ν_m = ν_{m0}/T^{3/2} (paper sec. 3, rec. 1).
const T_FLOOR = 0.01

"""
Total transverse field `H_* = H^{ind} + H^{ext}` without clipping via `max(H_*, 0)`.
Name `H_star_channel_nonneg` is kept for existing call sites.
"""
@inline H_star_channel_nonneg(H_ind_plus_ext::Float64) = H_ind_plus_ext

"""
Local electron collision frequency:

    ν_e = ν_base(T) + α_B · ω_ce_dim,     ω_ce_dim = ε · |H| / m_e,

with `ν_base` set by collision model:
- `:spitzer`  — ν_base = ν_m0 / (T^{3/2} + T_FLOOR^{3/2});
- `:constant` — ν_base = ν_m0 (effective anomalous rate, ν_e ≈ const).

Second term — Bohm-like anomaly (Hagelaar 2002, Boeuf–Garrigues); off when α_B = 0,
α_B = 1/16 matches classical Bohm limit.
"""
@inline function local_nu_m(
    ν_m0::Float64, T_loc::Float64, model::Symbol, H_loc::Float64,
    alpha_B::Float64, ε::Float64, me::Float64,
)
    ν_base = if model === :constant
        ν_m0
    else
        T_eff = max(T_loc, 0.0)
        ν_m0 / (T_eff^(3 / 2) + T_FLOOR^(3 / 2))
    end
    ν_anom = alpha_B * ε * abs(H_loc) / max(me, eps())
    return ν_base + ν_anom
end

@inline local_nu_m(ν_m0::Float64, T_loc::Float64, model::Symbol = :spitzer) =
    local_nu_m(ν_m0, T_loc, model, 0.0, 0.0, 0.0, 1.0)
"""
    neutrals_evolution(n_a_new, n_a_old, n_ion, τ, v_a, kI, h, n_source)

First-order upwind advection plus implicit ionisation sink for the neutral density:

    n_a^{n+1}_i = (n_a^n_i - v_a τ (n_a^n_i - n_a^n_{i-1}) / h) / (1 + τ kI n_i^n_i),
    n_a^{n+1}_1 = n_source                                          (Dirichlet inlet),
    n_a^{n+1}_i ≥ 0                                                 (clamped).

Issues a `@warn` when the upwind CFL `v_a τ / h` exceeds 1. Result is written into
`n_a_new`.
"""
function neutrals_evolution(
    n_a_new::Vector{Float64},
    n_a_old::Vector{Float64},
    n_ion::Vector{Float64},
    τ::Float64,
    v_a::Float64,
    kI::Float64,
    h::Float64,
    n_source::Float64,
)
    M = length(n_a_old) - 1
    C = v_a * τ / h
    if C > 1.0
        @warn "Neutral CFL violated: v_a*τ/h = $C > 1; reduce τ or increase resolution"
    end
    n_a_new[1] = n_source
    for i in 2:(M + 1)
        convection = -v_a * τ * (n_a_old[i] - n_a_old[i - 1]) / h
        n_a_new[i] = (n_a_old[i] + convection) / (1.0 + τ * kI * n_ion[i])
        n_a_new[i] = max(n_a_new[i], 0.0)
    end
end

"""
    intermediate_temperature(T_new, T_old, n, vz, j, n_a, τ, γ, mi, me, ν_m0, kI, h; …)

Single-step explicit update of the electron temperature on grid nodes:

    T^{n+1} = T^n + τ [Q_coll + Q_ion - (γ-1) T^n ∂_z v_iz]

with

    Q_coll = (γ-1) (m_i/m_Σ) ν_m(T,|H_*|) j² / n_safe,   m_Σ = m_i + m_e,
    Q_ion  = -(γ-1) k_I T^n n_a.

`n_safe = n + max(n_reg_min, N_FLOOR)` regularises the Joule heating denominator.
The result is clamped to `[T_FLOOR, T_cap]`. Boundary nodes are set by Neumann
copy from interior. Optional `H_total` (with `alpha_B`, `ε`, `me`) feeds the
Bohm-anomaly contribution to `ν_m`.
"""
function intermediate_temperature(
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
    h::Float64;
    n_reg_min::Union{Nothing, AbstractVector{Float64}} = nothing,
    T_cap::Float64 = 1e3,
    collision_model::Symbol = :spitzer,
    alpha_B::Float64 = 0.0,
    ε::Float64 = 0.0,
    H_total::Union{Nothing, AbstractVector{Float64}} = nothing,
)
    M = length(T_old) - 1
    mΣ = mi + me
    for i in 2:M
        T_loc = max(T_old[i], 0.0)
        n_loc = n[i] + (n_reg_min === nothing ? N_FLOOR : max(n_reg_min[i], N_FLOOR))
        H_loc = H_total === nothing ? 0.0 : H_star_channel_nonneg(H_total[i])
        ν_m = local_nu_m(ν_m0, T_loc, collision_model, H_loc, alpha_B, ε, me)
        vz_im1 = i == 2 ? (2 * vz[1] - vz[2]) : vz[i - 1]
        vz_ip1 = i == M ? (2 * vz[M + 1] - vz[M]) : vz[i + 1]
        dvz = (vz_ip1 - vz_im1) / (2h)
        Q_collision = (γ - 1) * (mi / mΣ) * ν_m * j[i]^2 / n_loc
        Q_ionisation = - (γ - 1) * kI * T_loc * n_a[i]
        T_new[i] = T_old[i] + τ * (Q_collision + Q_ionisation - (γ - 1) * T_loc * dvz)
        if !isfinite(T_new[i])
            T_new[i] = T_old[i]
        end
        T_new[i] = clamp(T_new[i], T_FLOOR, T_cap)
    end
    T_new[1] = T_new[2]
    T_new[M + 1] = T_new[M]
    return T_new
end

Base.@deprecate intermidiate_temperature intermediate_temperature

"""
Grid `j_y` from induced `H_x^{ind}` on half-nodes: `j ≈ ∂H^{ind}/∂z` (1D Ampere for plasma current).

External `H_ext(z)` is analytic (`H0_func`); its ∂/∂z is **not** part of plasma current dynamics here
(analogous to coil/ferrite currents outside the model domain).
`H_ext` enters **`H_* = H^{ind} + H_ext`** in Ohm’s law for `E_y`, in `ν_m(|H*|)`, particle pushes,
and Eq. (38) for `E_z`, but is **not** added again to `j` from `compute_current`.
"""
function compute_current(j::Vector{Float64}, H_x::Vector{Float64}, h::Float64)
    M = length(H_x)
    @assert length(j) == M + 1
    for i in 2:M
        j[i] = (H_x[i] - H_x[i - 1]) / h
    end
    j[1] = j[M + 1] = 0.0
    return j
end

"""
Elliptic equation for `E_y` plus optional half-node Faraday update of induced `H_x^{ind}`.

If `apply_faraday = true` (legacy single-pass): when `advance_induced_H = true`, Faraday accumulation
`H_x^{new} = H_x^{old} + τ·∂_z E_y`; otherwise `H_x^{new} = τ·∂_z E_y`. Then `j` is updated from `H_x^{new}`.

If `apply_faraday = false`, only `E_y` is solved; `H_x^{ind}` and `j` are returned unchanged so the driver can
advance particles at one time level and apply Faraday after the particle push using this `E_y`
(leapfrog-style: `H^{n+1} = H^n + τ·∂_z E_y^n`).

Closure uses sum of interpolated induced and external field at nodes.

External field via `H_ext_at_nodes` or `H0_func.(x_grid)`.
"""
function electric_field_solver(
    E_y::Vector{Float64},
    H_x_old::Vector{Float64},
    j_old::Vector{Float64},
    n::Vector{Float64},
    vz::Vector{Float64},
    T::Vector{Float64},
    τ::Float64,
    α::Float64,
    ν_m0::Float64,
    h::Float64,
    x_grid::AbstractVector{Float64},
    H0_func,
    bc_type::Symbol,
    v_a::Float64,
    c_inv::Float64;
    n_reg_min::Union{Nothing, AbstractVector{Float64}} = nothing,
    collision_model::Symbol = :spitzer,
    alpha_B::Float64 = 0.0,
    ε::Float64 = 0.0,
    me::Float64 = 1.0,
    H_ext_at_nodes::Union{Nothing, AbstractVector{Float64}} = nothing,
    advance_induced_H::Bool = false,
    apply_faraday::Bool = true,
)
    M = length(H_x_old)
    @assert length(E_y) == M + 1
    @assert length(j_old) == M + 1
    @assert length(n) == M + 1
    @assert length(vz) == M + 1
    @assert length(T) == M + 1
    H_interpolated = zeros(M + 1)
    H_interpolated[1] = H_x_old[1]
    for i in 2:M
        H_interpolated[i] = (H_x_old[i - 1] + H_x_old[i]) / 2
    end
    H_interpolated[M + 1] = H_x_old[M]
    if H_ext_at_nodes === nothing
        H_star = H_star_channel_nonneg.(H_interpolated .+ H0_func.(x_grid))
    else
        @assert length(H_ext_at_nodes) == M + 1
        H_star = H_star_channel_nonneg.(H_interpolated .+ H_ext_at_nodes)
    end
    a = zeros(M + 1)
    b = zeros(M + 1)
    c = zeros(M + 1)
    d = zeros(M + 1)
    for i in 2:M
        n_loc = n[i] + (n_reg_min === nothing ? N_FLOOR : max(n_reg_min[i], N_FLOOR))
        T_loc = max(T[i], 0.0)
        ν_m = local_nu_m(ν_m0, T_loc, collision_model, H_star[i], alpha_B, ε, me)
        # Semi-implicit scheme (36), p.35: j^{1/2} = j^0 + (τ/(2h²))(E_{k+1}-2E_k+E_{k-1}),
        # H^{1/2}_* = H^0_* + (τ/(4h))(E_{k+1}-E_{k-1}); coefficients B,C,D follow.
        A = α / (n_loc * h^2)
        B = ν_m * (τ / (2h^2))
        vz_im1 = i == 2 ? (2 * vz[1] - vz[2]) : vz[i - 1]
        vz_ip1 = i == M ? (2 * vz[M + 1] - vz[M]) : vz[i + 1]
        dvz = (vz_ip1 - vz_im1) / (2h)
        C = vz[i] * τ * c_inv / (4h)
        D = (α * τ / (2 * n_loc * h^2)) * dvz
        a[i] = -A - (B + D) - C
        b[i] = 1.0 + 2A + 2(B + D)
        c[i] = -A - (B + D) + C
        dj = (j_old[i + 1] - j_old[i - 1]) / (2h)
        d[i] = ν_m * j_old[i] - H_star[i] * vz[i] * c_inv +
            (α / n_loc) * j_old[i] * dvz + (α / n_loc) * vz[i] * dj
    end
    if bc_type == :j0
        n1 = n[1] + (n_reg_min === nothing ? N_FLOOR : max(n_reg_min[1], N_FLOOR))
        nM = n[M + 1] + (n_reg_min === nothing ? N_FLOOR : max(n_reg_min[M + 1], N_FLOOR))
        dj_left = j_old[2] / h
        E_y[1] = (-H_star[1] + (α / n1) * dj_left) * vz[1] * c_inv
        dj_right = -j_old[M] / h
        E_y[M + 1] = (-H_star[M + 1] + (α / nM) * dj_right) * vz[M + 1] * c_inv
        if M - 1 > 0
            d[2] -= a[2] * E_y[1]
            a[2] = 0.0
            d[M] -= c[M] * E_y[M + 1]
            c[M] = 0.0
        end
    else
        error("Unsupported boundary condition: $bc_type")
    end
    if M - 1 > 0
        E_inner = solve_tridiagonal(view(a, 3:M), view(b, 2:M), view(c, 2:(M - 1)), view(d, 2:M))
        E_y[2:M] .= E_inner
    end
    if !apply_faraday
        return E_y, H_x_old, j_old
    end
    H_x_new = similar(H_x_old)
    for i in 1:M
        d_faraday = τ * (E_y[i + 1] - E_y[i]) / h
        H_x_new[i] = advance_induced_H ? H_x_old[i] + d_faraday : d_faraday
    end
    j_new = zeros(M + 1)
    compute_current(j_new, H_x_new, h)
    return E_y, H_x_new, j_new
end

"""
    compute_Ez(Ez, H_x_old, H_x_new, j_old, j_new, n, T, vy, n_a_new, n_a_old,
               α0, ζ, kI, va, h, λ_e_λΣ, β_Ez_coef, c_inv, τ, x_grid, H0_func, n_floor;
               Ez_term1, Ez_term2, Ez_term3, Ez_term4, H_ext_at_nodes)

Algebraic axial electric field (paper Eq. (38)):

    E_z = (H_*/c) v_iy
        - (λ_e/λ_Σ / ε) k_I n_a v_a
        - (α_0 / n_safe) H_* j_y
        - (ζ α_0 / n_safe) ∂_z(n T_e).

Implementation notes:
- If `H_x_old === H_x_new` and `j_old === j_new` (single time-layer call from the driver),
  no half-sum buffers are allocated; otherwise mid-step `0.5(old+new)` averages are used.
- Steklov smoothing with `radius=1, passes=5` is applied to `n·T` and `v_iy`; the Hall
  source `j_y` is **not** smoothed (smoothing the curl-derived current would violate
  consistency with `compute_current = ∂_z H`).
- Optional output buffers `Ez_termK` capture the four contributions for diagnostics.
- `H_*` at nodes uses interpolation from half-nodes plus `H_ext_at_nodes` (or `H0_func`).
- Non-finite `Ez[i]` are reset to 0 to keep downstream PIC pushers stable.
"""
function compute_Ez(
    Ez::Vector{Float64},
    H_x_old::Vector{Float64},
    H_x_new::Vector{Float64},
    j_old::Vector{Float64},
    j_new::Vector{Float64},
    n::Vector{Float64},
    T::Vector{Float64},
    vy::Vector{Float64},
    n_a_new::Vector{Float64},
    n_a_old::Vector{Float64},
    α0::Float64,
    ζ::Float64,
    kI::Float64,
    va::Float64,
    h::Float64,
    λ_e_λΣ::Float64,
    β_Ez_coef::Float64,
    c_inv::Float64,
    τ::Float64,
    x_grid::AbstractVector{Float64},
    H0_func,
    n_floor::Float64 = N_FLOOR;
    Ez_term1::Union{Nothing, Vector{Float64}} = nothing,
    Ez_term2::Union{Nothing, Vector{Float64}} = nothing,
    Ez_term3::Union{Nothing, Vector{Float64}} = nothing,
    Ez_term4::Union{Nothing, Vector{Float64}} = nothing,
    H_ext_at_nodes::Union{Nothing, AbstractVector{Float64}} = nothing,
)
    M = length(H_x_old)
    @assert length(Ez) == M + 1
    same_hj = H_x_old === H_x_new && j_old === j_new
    same_na = n_a_new === n_a_old
    if !same_hj
        H_x_mid = 0.5 .* (H_x_new .+ H_x_old)
        j_mid = 0.5 .* (j_new .+ j_old)
    end
    H_interpolation = zeros(M + 1)
    if same_hj
        hx = H_x_old
        j_src = j_old
    else
        hx = H_x_mid
        j_src = j_mid
    end
    H_interpolation[1] = hx[1]
    for i in 2:M
        H_interpolation[i] = (hx[i] + hx[i - 1]) / 2
    end
    H_interpolation[M + 1] = hx[M]
    if H_ext_at_nodes === nothing
        H_star = H_star_channel_nonneg.(H_interpolation .+ H0_func.(x_grid))
    else
        @assert length(H_ext_at_nodes) == M + 1
        H_star = H_star_channel_nonneg.(H_interpolation .+ H_ext_at_nodes)
    end
    # Eq. (38): E_z = H_*·v_{iy} − …; Steklov (43) on PIC moments (n·T), v_iy only — do not smooth j (Hall source).
    nT = n .* T
    Steklov_smooth(nT, 1, 5)
    vy_sm = copy(vy)
    Steklov_smooth(vy_sm, 1, 5)
    d_nT = zeros(M + 1)
    for i in 2:M
        d_nT[i] = (nT[i + 1] - nT[i - 1]) / (2h)
    end
    d_nT[1] = (nT[2] - (2 * nT[1] - nT[2])) / (2h)
    d_nT[M + 1] = ((2 * nT[M + 1] - nT[M]) - nT[M - 1]) / (2h)
    n_safe = max.(n, 0.0) .+ n_floor
    term1_arr = zeros(M + 1)
    term2_arr = zeros(M + 1)
    term3_arr = zeros(M + 1)
    term4_arr = zeros(M + 1)
    for i in 1:(M + 1)
        n_a_mid = same_na ? n_a_new[i] : 0.5 * (n_a_new[i] + n_a_old[i])
        term1_arr[i] = H_star[i] * vy_sm[i] * c_inv
        term2_arr[i] = β_Ez_coef * kI * n_a_mid * va
        term3_arr[i] = (α0 / n_safe[i]) * H_star[i] * j_src[i]
        term4_arr[i] = (ζ * α0 / n_safe[i]) * d_nT[i]
    end
    Ez .= term1_arr .- term2_arr .- term3_arr .- term4_arr
    for i in 1:(M + 1)
        if !isfinite(Ez[i])
            Ez[i] = 0.0
        end
    end
    Ez_term1 !== nothing && (Ez_term1 .= term1_arr)
    Ez_term2 !== nothing && (Ez_term2 .= term2_arr)
    Ez_term3 !== nothing && (Ez_term3 .= term3_arr)
    Ez_term4 !== nothing && (Ez_term4 .= term4_arr)
    return Ez
end

end

module ParticleMovementSPT

using ..PartCount
using ..PlasmaDynamics: H_star_channel_nonneg
using ..NumericalFunctionsSPT
using LinearAlgebra

const MIN_PARTICLE_MASS = 1e-8
# n_a fraction: lower model bound on `n` in birth term as n_i→0; see `new_particles_ionisation`.
const N_ION_BIRTH_FLOOR_FRAC = 1e-4
const N_FLOOR = 1e-8
const T_FLOOR = 0.01

export deposit_particles, move_particles, new_particles_ionisation, remove_inactive_particles

"""
    deposit_particles(particles, x_grid, n, v_y, v_z, T, h, n_vy, n_vz, n_T)

CIC deposition of macroparticles to grid nodes. For each active particle with weight `q`
and velocity `(v_y, v_z)`:

    n[k] += q · w_k,                 n_vy[k] += q · w_k · v_y,   etc.

After the loop:

    v_y[i] = n_vy[i] / n[i]   (similarly v_z, T) when n[i] > N_FLOOR; else zero / T_FLOOR,
    n[i]   = n[i] / V_i       with V_i = h/2 at endpoints, h elsewhere (control-volume).

`n_vy`, `n_vz`, `n_T` are scratch arrays sized `M + 1`. The function returns
`(n, v_y, v_z, T)` but writes them in place.
"""
function deposit_particles(
    particles::Vector{Particle},
    x_grid::AbstractVector{Float64},
    n::Vector{Float64},
    v_y::Vector{Float64},
    v_z::Vector{Float64},
    T::Vector{Float64},
    h::Float64,
    n_vy::Vector{Float64},
    n_vz::Vector{Float64},
    n_T::Vector{Float64},
)
    fill!(n, 0.0)
    fill!(n_vz, 0.0)
    fill!(n_vy, 0.0)
    fill!(n_T, 0.0)
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
    for i in eachindex(n)
        vol = (i == 1 || i == length(n)) ? h / 2 : h
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

"""
    move_particles(particles, E_y0, E_y1, E_z0, E_z1, H_x0, H_x1, j0, j1,
                   ν_m_grid0, ν_m_grid1, x_grid, x_half, τ, h, ε, mi, c_inv, H0_func, counters)
        -> thrust_step

Advance every active macroparticle over `τ` using Boris-style midpoint predictor–corrector
for the ion characteristics (paper system (11), dimensionless):

    ∂v_y/∂t = ε (E_y + (H_*/c) v_z - ν_m j_y),
    ∂v_z/∂t = ε (E_z - (H_*/c) v_y),
    ∂z/∂t   = v_z.

Per-particle subcycling: `N0 = max(1, ceil(τ |v| / (0.25 h)))` so each substep moves
at most 0.25 cell widths. Fields are **time-linearly interpolated** between the old (`*0`)
and new (`*1`) snapshots — currently the driver passes the same arrays for both, which
makes the push effectively single-layer.

Boundary handling:
- Right exit (`z ≥ L`): particle deactivated, `thrust_step += m_i q max(v_z, 0)`,
  `counters.exited_right += 1` (∑ over the step gives the streamwise momentum flux out).
- Left wall (`z ≤ 0`): absorbed (`counters.reflected_left += 1`); reinjection happens
  through the neutral source.
- NaN velocities: particle deactivated, `counters.nan += 1`.

Returns `thrust_step` (raw axial momentum carried out at the right boundary).
"""
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
    ν_m0_grid0::Vector{Float64},
    ν_m0_grid1::Vector{Float64},
    x_grid::AbstractVector{Float64},
    x_half::AbstractVector{Float64},
    τ::Float64,
    h::Float64,
    ε::Float64,
    mi::Float64,
    c_inv::Float64,
    H0_func::Function,
    counters::Counters,
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
        for i in 1:N0
            if z < x_grid[1]
                z = x_grid[1]
                vz = abs(vz)
            elseif z > x_grid[end]
                z = x_grid[end]
            end
            t_mid_relative = (i - 0.5) / N0
            k0, k1, w0, w1 = interpolation_weights(z, x_grid)
            w_t0 = 1 - t_mid_relative
            E_y_mid = w_t0 * (w0 * E_y0[k0] + w1 * E_y0[k1]) + t_mid_relative * (w0 * E_y1[k0] + w1 * E_y1[k1])
            E_z_mid = w_t0 * (w0 * E_z0[k0] + w1 * E_z0[k1]) + t_mid_relative * (w0 * E_z1[k0] + w1 * E_z1[k1])
            j_mid = w_t0 * (w0 * j0[k0] + w1 * j0[k1]) + t_mid_relative * (w0 * j1[k0] + w1 * j1[k1])
            ν_m_mid = w_t0 * (w0 * ν_m0_grid0[k0] + w1 * ν_m0_grid0[k1]) + t_mid_relative * (w0 * ν_m0_grid1[k0] + w1 * ν_m0_grid1[k1])
            if z <= x_half[1]
                kh, wh = 1, 1.0
            elseif z >= x_half[end]
                kh, wh = length(x_half), 1.0
            else
                kh = floor(Int, (z - x_half[1]) / h) + 1
                kh = clamp(kh, 1, length(x_half) - 1)
                wh = (z - x_half[kh]) / h
            end
            H_now = (kh < length(x_half)) ? (1 - wh) * H_x0[kh] + wh * H_x0[kh + 1] : H_x0[kh]
            H_next = (kh < length(x_half)) ? (1 - wh) * H_x1[kh] + wh * H_x1[kh + 1] : H_x1[kh]
            H_mid = (1 - t_mid_relative) * H_now + t_mid_relative * H_next
            H_star_mid = H_star_channel_nonneg(H_mid + H0_func(z))
            # Ion characteristics (system 11): ∂v_y/∂t = ε(E_y + H_* v_z/c − j·ν_m),
            #                                  ∂v_z/∂t = ε(E_z − H_* v_y/c).
            j_over_σ = j_mid * ν_m_mid
            vy_pred = vy + 0.5 * τ0 * ε * (E_y_mid + H_star_mid * vz * c_inv - j_over_σ)
            vz_pred = vz + 0.5 * τ0 * ε * (E_z_mid - H_star_mid * vy * c_inv)
            vy_new = vy + τ0 * ε * (E_y_mid + H_star_mid * vz_pred * c_inv - j_mid * ν_m_mid)
            vz_new = vz + τ0 * ε * (E_z_mid - H_star_mid * vy_pred * c_inv)
            if !isfinite(vy_new) || !isfinite(vz_new)
                counters.nan += 1
                p.active = false
                break
            end
            z_new = z + τ0 * (vz + vz_new) / 2
            p.z = z_new
            p.vy = vy_new
            p.vz = vz_new
            z, vy, vz = z_new, vy_new, vz_new
        end
        if p.z >= L
            p.active = false
            thrust_step += mi * p.q * max(p.vz, 0.0)
            counters.exited_right += 1
        elseif p.z <= 0.0
            # Left wall: absorbed ion -> neutral reinjected via n_a boundary source.
            p.active = false
            counters.reflected_left += 1
        end
    end
    return thrust_step
end

"""
    new_particles_ionisation(particles, n_a_new, n_ion, x_grid, τ, kI, v_a, T_ion;
                             charge_factor=1.0)

Append macroparticles representing newly ionised xenon/krypton over `τ`. At each node `i`:

    q_new = charge_factor · τ · n_a · n_i · k_I · V_i

(`V_i = h/2` at endpoints, `h` elsewhere). If `q_new > MIN_PARTICLE_MASS` (1e-8) a
particle is spawned at `x_grid[i]` with `(v_y, v_z) = (0, v_a)` and temperature `T_ion`;
otherwise the contribution is dropped (avoids a long tail of ultra-small weights).

The growing population is bounded by exit at the right wall plus volumetric recombination
(`remove_inactive_particles`).
"""
function new_particles_ionisation(
    particles::Vector{Particle},
    n_a_new::Vector{Float64},
    n_ion::Vector{Float64},
    x_grid::AbstractVector{Float64},
    τ::Float64,
    kI::Float64,
    v_a::Float64,
    T_ion::Float64;
    charge_factor::Float64 = 1.0,
)
    M = length(x_grid) - 1
    h = x_grid[2] - x_grid[1]
    for i in eachindex(x_grid)
        vol = (i == 1 || i == M + 1) ? h / 2 : h
        n_a_loc = max(n_a_new[i], 0.0)
        n_i_loc = max(n_ion[i], 0.0)
        q_new = charge_factor * τ * n_a_loc * n_i_loc * kI * vol
        q_new < MIN_PARTICLE_MASS && continue
        push!(particles, Particle(x_grid[i], 0.0, v_a, T_ion, q_new, true))
    end
    return particles
end

"""
    remove_inactive_particles(particles, L, τ, kR)

Volumetric recombination + housekeeping. Each active particle is killed with probability
`P_rec = min(1, kR · τ)`; particles that drifted out of `[0, L]` are also marked inactive.
After flagging, `filter!` compacts the vector. This is the only routine that shrinks the
particle list, so it is the principal counterweight to `new_particles_ionisation`.
"""
function remove_inactive_particles(particles::Vector{Particle}, L::Float64, τ::Float64, kR::Float64)
    for p in particles
        if p.z < 0 || p.z > L
            p.active = false
            continue
        end
        P_rec = min(1.0, kR * τ)
        rand() < P_rec && (p.active = false)
    end
    filter!(p -> p.active, particles)
    return particles
end

end
