module DiagnosticsMetrics

using ..PartCount

export ewma_update, print_dimensionless_diagnostics, dimensionless_summary,
    print_physicality_report, collect_physicality_metrics, hall_parameter_electron

"""
    dimensionless_summary(params) -> NamedTuple

Quick "scaling sanity check" for `SimParams`:
- `ε_time = ε`,
- `λ_ratio = m_e/(m_i+m_e)`,
- `α0` (Eq. (38) closure coefficient),
- `Da_ion = kI · L / v_a`, `Da_rec = kR · L / v_a` — ionisation / recombination Damköhler,
- `Rm_star = v_a · L / ν_m0` — magnetic Reynolds-like number,
- `tau_cfl_neutral = h / v_a` — neutral CFL ceiling.
"""
function dimensionless_summary(params::SimParams)
    ε_time = params.ε
    λ_ratio = params.λ_e_λΣ
    Da_ion = params.kI * params.L / max(params.v_a, eps())
    Da_rec = params.kR * params.L / max(params.v_a, eps())
    Rm_star = params.v_a * params.L / max(params.ν_m0, eps())
    tau_cfl_neutral = params.h / max(params.v_a, eps())
    return (; ε_time, λ_ratio, α0 = params.α0, Da_ion, Da_rec, Rm_star, tau_cfl_neutral)
end

"""
    print_dimensionless_diagnostics(params)

Prints the dimensionless summary in human-readable form. Called once at the start of
`run_simulation`.
"""
function print_dimensionless_diagnostics(params::SimParams)
    s = dimensionless_summary(params)
    println("Dimensionless diagnostics:")
    println("  ε = $(s.ε_time), λ_e/λΣ = $(s.λ_ratio), α0 = $(s.α0)")
    println("  Da_ion = $(s.Da_ion), Da_rec = $(s.Da_rec), Rm* = $(s.Rm_star), τ ≲ h/v_a = $(s.tau_cfl_neutral)")
    println("  E_z0 = $(params.E_z0_dimless)")
end

"""Exponentially weighted moving average update: `α x + (1-α) prev`."""
@inline function ewma_update(prev::Float64, x::Float64, α::Float64)
    return α * x + (1 - α) * prev
end

"""
    hall_parameter_electron(params, H_total, nu_m; ν_en_const=0) -> Float64

Local Hall parameter for electrons in dimensionless units:

    β_e = ω_ce / ν_e = ε · |H_*| / (m_e · (ν_m + ν_en_const)).

Used both per-step (driver) and per-snapshot (`print_physicality_report`,
`collect_physicality_metrics`).
"""
@inline function hall_parameter_electron(params::SimParams, H_total::Float64, nu_m::Float64; ν_en_const::Float64 = 0.0)
    νe_total = max(nu_m + max(ν_en_const, 0.0), eps())
    return params.ε * abs(H_total) / (max(params.me, eps()) * νe_total)
end

"""
    print_physicality_report(snapshots, params; ν_en_const=0)

For every saved snapshot, prints `max|E_z_termK|` for K=1..4, the dominance ratio
`max|term1| / Σ_{k=2..4} max|termK|` (≈ 1 means the inductive term `(H_*/c) v_iy` carries
the field), and the min/mean/max of the electron Hall parameter on the same snapshot.
"""
function print_physicality_report(snapshots::Dict{Float64, <:NamedTuple}, params::SimParams; ν_en_const::Float64 = 0.0)
    isempty(snapshots) && return
    println("Physicality report:")
    for t in sort(collect(keys(snapshots)))
        s = snapshots[t]
        max_t1 = maximum(abs.(s.E_z_term1))
        max_t2 = maximum(abs.(s.E_z_term2))
        max_t3 = maximum(abs.(s.E_z_term3))
        max_t4 = maximum(abs.(s.E_z_term4))
        denominator = max_t2 + max_t3 + max_t4 + eps()
        dominance = max_t1 / denominator
        χ = hall_parameter_electron.(Ref(params), s.H_total, s.nu_m; ν_en_const = ν_en_const)
        println("  t=$t term1..4 max: $(round(max_t1, sigdigits=4)), $(round(max_t2, sigdigits=4)), $(round(max_t3, sigdigits=4)), $(round(max_t4, sigdigits=4)), dom=$(round(dominance, sigdigits=4)), β $(round(minimum(χ), sigdigits=4))/$(round(sum(χ)/length(χ), sigdigits=4))/$(round(maximum(χ), sigdigits=4))")
    end
end

"""
    collect_physicality_metrics(snapshots, params; ν_en_const=0) -> NamedTuple

Same data as `print_physicality_report` but returned as `(per_time, summary)` for
programmatic post-processing (e.g. parameter sweeps).
"""
function collect_physicality_metrics(snapshots::Dict{Float64, <:NamedTuple}, params::SimParams; ν_en_const::Float64 = 0.0)
    times = sort(collect(keys(snapshots)))
    per_time = NamedTuple[]
    for t in times
        s = snapshots[t]
        max_t1 = maximum(abs.(s.E_z_term1))
        max_t2 = maximum(abs.(s.E_z_term2))
        max_t3 = maximum(abs.(s.E_z_term3))
        max_t4 = maximum(abs.(s.E_z_term4))
        denominator = max_t2 + max_t3 + max_t4 + eps()
        dominance = max_t1 / denominator
        χ = hall_parameter_electron.(Ref(params), s.H_total, s.nu_m; ν_en_const = ν_en_const)
        push!(per_time, (;
            t = t,
            term1_max = max_t1,
            term2_max = max_t2,
            term3_max = max_t3,
            term4_max = max_t4,
            term1_dominance = dominance,
            chi_mean = sum(χ) / length(χ),
            chi_peak = maximum(χ),
            beta_min = minimum(χ),
            beta_mean = sum(χ) / length(χ),
            beta_peak = maximum(χ),
        ))
    end
    isempty(per_time) && return (; per_time, summary = nothing)
    summary = (;
        beta_peak_max = maximum(x.beta_peak for x in per_time),
        beta_peak_min = minimum(x.beta_peak for x in per_time),
        beta_mean_avg = sum(x.beta_mean for x in per_time) / length(per_time),
        term1_dominance_avg = sum(x.term1_dominance for x in per_time) / length(per_time),
        term1_dominance_final = per_time[end].term1_dominance,
        chi_peak_max = maximum(x.chi_peak for x in per_time),
    )
    return (; per_time, summary)
end

end
