module Visualization

using CairoMakie
using Printf

export plot_results

const _MAX_AXES_PER_FIG = 4
const _FONT_TNR = "Times New Roman"

"""Italicize latin tokens [A-Za-z] (+ optional `_latin`); keep `[...]` substrings verbatim (units); other text upright."""
function _rich_scientific_label(s::AbstractString)
    s = String(s)
    isempty(s) && return rich("")
    rx = r"[A-Za-z]+(?:_[A-Za-z]+)*"
    out = nothing
    i = firstindex(s)
    while i <= lastindex(s)
        if s[i] == '['
            j = findnext(']', s, i)
            if j === nothing
                out = _rich_cat(out, _rich_latin_runs(s[i:end], rx))
                break
            end
            out = _rich_cat(out, rich(String(s[i:j])))
            i = nextind(s, j)
        else
            j = findnext('[', s, i)
            seg = j === nothing ? s[i:end] : s[i:prevind(s, j)]
            out = _rich_cat(out, _rich_latin_runs(seg, rx))
            i = j === nothing ? lastindex(s) + 1 : j
        end
    end
    return out === nothing ? rich("") : out
end

function _rich_latin_runs(seg::AbstractString, rx::Regex)
    seg = String(seg)
    isempty(seg) && return rich("")
    out = nothing
    prev = firstindex(seg)
    for rng in findall(rx, seg)
        if rng.start > prev
            out = _rich_cat(out, rich(String(SubString(seg, prev, prevind(seg, rng.start)))))
        end
        out = _rich_cat(out, rich(String(seg[rng]), font = :italic))
        prev = nextind(seg, rng.stop)
    end
    if prev <= lastindex(seg)
        out = _rich_cat(out, rich(String(SubString(seg, prev))))
    end
    return out === nothing ? rich("") : out
end

function _rich_cat(out, x)
    if out === nothing
        return x
    end
    return rich(out, x)
end

function _rich_legend_time(t::Float64)
    return rich(rich("t", font = :italic), " = ", string(t))
end

function _tnr_theme(fontsize::Float64)
    return Theme(;
        fontsize = fontsize,
        font = _FONT_TNR,
        Axis = (;
            xlabelfont = _FONT_TNR,
            ylabelfont = _FONT_TNR,
            titlefont = _FONT_TNR,
            xticklabelfont = _FONT_TNR,
            yticklabelfont = _FONT_TNR,
            # Makie fails on zero-axis range data (IntervalsBetween → "range step cannot be zero").
            xminorgridvisible = false,
            yminorgridvisible = false,
        ),
        Legend = (; labelfont = _FONT_TNR, titlefont = _FONT_TNR),
    )
end

function _axis_slot(fig::Figure, k::Int)
    r = (k - 1) ÷ 2 + 1
    c = (k - 1) % 2 + 1
    return Axis(fig[r, c])
end

function _profile_line_dimless!(ax, gi::Int, snap, lbl, color)
    if gi == 1
        lines!(ax, snap.z, snap.n_a, color = color, label = lbl)
    elseif gi == 2
        lines!(ax, snap.z, snap.n_i, color = color, label = lbl)
    elseif gi == 3
        lines!(ax, snap.z, snap.v_iy, color = color, label = lbl)
    elseif gi == 4
        lines!(ax, snap.z, snap.v_ey, color = color, label = lbl)
    elseif gi == 5
        lines!(ax, snap.z, snap.v_iz, color = color, label = lbl)
    elseif gi == 6
        lines!(ax, snap.z, snap.T_e, color = color, label = lbl)
    elseif gi == 7
        lines!(ax, snap.z, snap.E_y, color = color, label = lbl)
    elseif gi == 8
        lines!(ax, snap.z, snap.E_z, color = color, label = lbl)
    elseif gi == 9
        lines!(ax, snap.z, snap.j, color = color, label = lbl)
    elseif gi == 10
        lines!(ax, snap.z, snap.H_x, color = color, label = lbl)
    elseif gi == 11
        lines!(ax, snap.z_half, snap.H_x_half, color = color, label = lbl)
    elseif gi == 12
        lines!(ax, snap.z, snap.H_ext, color = color, label = lbl)
    elseif gi == 13
        lines!(ax, snap.z, snap.H_total, color = color, label = lbl)
    elseif gi == 14
        lines!(ax, snap.z, snap.nu_m, color = color, label = lbl)
    elseif gi == 15
        lines!(ax, snap.z, snap.E_z_term1, color = color, label = lbl)
    elseif gi == 16
        lines!(ax, snap.z, snap.E_z_term2, color = color, label = lbl)
    elseif gi == 17
        lines!(ax, snap.z, snap.E_z_term3, color = color, label = lbl)
    elseif gi == 18
        lines!(ax, snap.z, snap.E_z_term4, color = color, label = lbl)
    end
    return nothing
end

"""
    plot_results(snapshots, diagnostics, save_times; output_dir)

Render the full set of dimensionless profile snapshots (18 fields per save time, paged
4-per-figure) plus the time-history diagnostics (thrust, `τ`, particle count, `max|E_z|`,
Hall parameter, peak `|v_ey|`, `Ey` RMS, `|H_self|/|H_ext|`) into PNGs in `output_dir`.
File names are `profiles_all_fields_NN.png` and `diagnostics_time_series_NN.png`.
"""
function plot_results(snapshots, diagnostics, save_times; output_dir::AbstractString = joinpath(pwd(), "output", "figures"))
    isdir(output_dir) || mkpath(output_dir)
    if !isempty(snapshots)
        times = sort(collect(keys(snapshots)))
        titles = [
            "n_a", "n_i", "v_iy (ions)", "v_ey = -j/(e n_e)", "v_iz (ions)", "T_e", "E_y",
            "E_z", "j_y", "H_x (nodes)", "H_x (half nodes)", "H_ext", "H_total",
            "nu_m", "E_z term1: (H/c)v_iy", "E_z term2: -(λ_iλ_e/λ_Σ)β n_a v_a", "E_z term3: Hall j", "E_z term4: pressure grad",
        ]
        ntot = length(titles)
        nrows_page = cld(_MAX_AXES_PER_FIG, 2)
        with_theme(_tnr_theme(11.0)) do
            pageno = 0
            for start in 1:_MAX_AXES_PER_FIG:(ntot - 1) + _MAX_AXES_PER_FIG
                pageno += 1
                stop = min(start + _MAX_AXES_PER_FIG - 1, ntot)
                nax = stop - start + 1
                fig = Figure(; size = (900, 280 * nrows_page + 40))
                axes = [_axis_slot(fig, k) for k in 1:nax]
                for (k, gi) in enumerate(start:stop)
                    ax = axes[k]
                    ax.title = _rich_scientific_label(titles[gi])
                    ax.xlabel = _rich_scientific_label("z")
                end
                colors = [Makie.wong_colors()[mod1(i, length(Makie.wong_colors()))] for i in 1:length(times)]
                for (idx, t) in enumerate(times)
                    snap = snapshots[t]
                    lbl = _rich_legend_time(t)
                    c = colors[idx]
                    for (k, gi) in enumerate(start:stop)
                        _profile_line_dimless!(axes[k], gi, snap, lbl, c)
                    end
                end
                for ax in axes
                    axislegend(ax, position = :rt, labelsize = 8)
                end
                save(joinpath(output_dir, @sprintf("profiles_all_fields_%02d.png", pageno)), fig)
            end
        end
    end
    if !isempty(diagnostics.time)
        _plot_diagnostics_dimensionless(diagnostics, output_dir)
    end
end

function _plot_diagnostics_dimensionless(diagnostics, output_dir::AbstractString)
    panels = Tuple{String, String, String, Function}[]
    push!(panels, ("Thrust", "t", "F_T", ax -> lines!(ax, diagnostics.time, diagnostics.thrust, color = :dodgerblue)))
    push!(panels, ("Adaptive step", "t", "tau", ax -> lines!(ax, diagnostics.time, diagnostics.tau, color = :darkorange)))
    push!(panels, ("Particle count", "t", "N_particles", ax -> lines!(ax, diagnostics.time, diagnostics.particle_count, color = :forestgreen)))
    push!(panels, ("Field amplitude", "t", "max |E_z|", ax -> lines!(ax, diagnostics.time, diagnostics.max_abs_Ez, color = :crimson)))
    if hasproperty(diagnostics, :hall_beta_mean)
        push!(panels, ("Hall parameter (mean)", "t", "beta_e mean", ax -> lines!(ax, diagnostics.time, diagnostics.hall_beta_mean, color = :purple)))
    end
    if hasproperty(diagnostics, :hall_beta_peak)
        push!(panels, ("Hall parameter (peak)", "t", "beta_e peak", ax -> lines!(ax, diagnostics.time, diagnostics.hall_beta_peak, color = :firebrick)))
    end
    if hasproperty(diagnostics, :vey_peak)
        push!(panels, ("peak |v_ey| (active zone)", "t", "|v_ey|", ax -> lines!(ax, diagnostics.time, diagnostics.vey_peak, color = :teal)))
    end
    if hasproperty(diagnostics, :ey_rms)
        push!(panels, ("Ey RMS (active zone)", "t", "rms(Ey)", ax -> lines!(ax, diagnostics.time, diagnostics.ey_rms, color = :brown)))
    end
    if hasproperty(diagnostics, :self_to_ext_B_ratio)
        push!(panels, ("max |Hself|/|Hext|", "t", "ratio", ax -> lines!(ax, diagnostics.time, diagnostics.self_to_ext_B_ratio, color = :black)))
    end
    nrows_page = cld(_MAX_AXES_PER_FIG, 2)
    with_theme(_tnr_theme(12.0)) do
        pageno = 0
        for chunk in Iterators.partition(panels, _MAX_AXES_PER_FIG)
            pageno += 1
            nax = length(chunk)
            fig = Figure(; size = (900, 280 * nrows_page + 40))
            axes = [_axis_slot(fig, k) for k in 1:nax]
            for (k, (title_s, xlab_s, ylab_s, plotfn)) in enumerate(chunk)
                ax = axes[k]
                ax.title = _rich_scientific_label(title_s)
                ax.xlabel = _rich_scientific_label(xlab_s)
                ax.ylabel = _rich_scientific_label(ylab_s)
                plotfn(ax)
            end
            save(joinpath(output_dir, @sprintf("diagnostics_time_series_%02d.png", pageno)), fig)
        end
    end
end

end

module VisualizationDimensional

using CairoMakie
using Printf
using ..PaperScales
using ..Visualization

export DimensionalScales, plot_results_dimensional

"""Centered moving average (odd window when possible); plots only — not used by the solver."""
function _thrust_box_smooth(x::AbstractVector, w::Int)
    n = length(x)
    n == 0 && return copy(x)
    w = max(1, min(n, w))
    half = w ÷ 2
    out = Vector{Float64}(undef, n)
    for i in 1:n
        a = max(1, i - half)
        b = min(n, i + half)
        s = 0.0
        @inbounds for j in a:b
            s += Float64(x[j])
        end
        out[i] = s / (b - a + 1)
    end
    return out
end

"""
    DimensionalScales(; z_m, n_m3, v_ms, t_s, Te_eV, E_vm, j_am2, B_t, nu_s, thrust_n)

Per-quantity dimensional factors (one `Float64` each) that map dimensionless solver
output to SI for the dimensional plots. Built from `si_plot_physical_scales`.
"""
Base.@kwdef struct DimensionalScales
    z_m::Float64
    n_m3::Float64
    v_ms::Float64
    t_s::Float64
    Te_eV::Float64
    E_vm::Float64
    j_am2::Float64
    B_t::Float64
    nu_s::Float64
    thrust_n::Float64
end

function _profile_line_dim!(ax, gi::Int, s, scales, lbl, color)
    z = s.z .* scales.z_m
    z_half = s.z_half .* scales.z_m
    v_iy_dim = s.v_iy .* scales.v_ms
    # Use the diagnostic v_ey already stored in the snapshot: `v_ey ≈ -j/(e n_e)`.
    v_ey_dim = s.v_ey .* scales.v_ms
    if gi == 1
        lines!(ax, z, s.n_a .* scales.n_m3, color = color, label = lbl)
    elseif gi == 2
        lines!(ax, z, s.n_i .* scales.n_m3, color = color, label = lbl)
    elseif gi == 3
        lines!(ax, z, v_iy_dim, color = color, label = lbl)
    elseif gi == 4
        lines!(ax, z, v_ey_dim, color = color, label = lbl)
    elseif gi == 5
        lines!(ax, z, s.v_iz .* scales.v_ms, color = color, label = lbl)
    elseif gi == 6
        lines!(ax, z, s.T_e .* scales.Te_eV, color = color, label = lbl)
    elseif gi == 7
        lines!(ax, z, s.E_y .* scales.E_vm, color = color, label = lbl)
    elseif gi == 8
        lines!(ax, z, s.E_z .* scales.E_vm, color = color, label = lbl)
    elseif gi == 9
        lines!(ax, z, s.j .* scales.j_am2, color = color, label = lbl)
    elseif gi == 10
        lines!(ax, z, s.H_x .* scales.B_t, color = color, label = lbl)
    elseif gi == 11
        lines!(ax, z_half, s.H_x_half .* scales.B_t, color = color, label = lbl)
    elseif gi == 12
        lines!(ax, z, s.H_ext .* scales.B_t, color = color, label = lbl)
    elseif gi == 13
        lines!(ax, z, s.H_total .* scales.B_t, color = color, label = lbl)
    elseif gi == 14
        lines!(ax, z, s.nu_m .* scales.nu_s, color = color, label = lbl)
    elseif gi == 15
        lines!(ax, z, s.E_z_term1 .* scales.E_vm, color = color, label = lbl)
    elseif gi == 16
        lines!(ax, z, s.E_z_term2 .* scales.E_vm, color = color, label = lbl)
    elseif gi == 17
        lines!(ax, z, s.E_z_term3 .* scales.E_vm, color = color, label = lbl)
    elseif gi == 18
        lines!(ax, z, s.E_z_term4 .* scales.E_vm, color = color, label = lbl)
    end
    return nothing
end

"""
    plot_results_dimensional(snapshots, diagnostics, save_times, scales; output_dir)

Same layout as `plot_results` but every axis is multiplied by the corresponding
`DimensionalScales` factor and labelled with SI units. Also produces a smoothed thrust
trace `diagnostics_thrust_smoothed_dimensional.png` (centred moving average with an
adaptive window proportional to `length(time)/40`).
"""
function plot_results_dimensional(snapshots, diagnostics, save_times, scales::DimensionalScales; output_dir::AbstractString = joinpath(pwd(), "output", "figures"))
    isdir(output_dir) || mkpath(output_dir)
    if !isempty(snapshots)
        times = sort(collect(keys(snapshots)))
        titles = [
            "n_a [m^-3]", "n_i [m^-3]", "v_iy ions [m/s]", "v_ey [m/s] = -j_y/(e n_e)", "v_iz ions [m/s]", "T_e [eV]", "E_y [V/m]",
            "E_z [V/m]", "j_y [A/m^2]", "H_x [T]", "H_x_half [T]", "H_ext [T]", "H_total [T]",
            "nu_m [1/s]", "E_z term1 [V/m]", "E_z term2 [V/m]", "E_z term3 [V/m]", "E_z term4 [V/m]",
        ]
        ntot = length(titles)
        nrows_page = cld(Visualization._MAX_AXES_PER_FIG, 2)
        with_theme(Visualization._tnr_theme(11.0)) do
            pageno = 0
            for start in 1:Visualization._MAX_AXES_PER_FIG:(ntot - 1) + Visualization._MAX_AXES_PER_FIG
                pageno += 1
                stop = min(start + Visualization._MAX_AXES_PER_FIG - 1, ntot)
                nax = stop - start + 1
                fig = Figure(; size = (900, 280 * nrows_page + 40))
                axes = [Visualization._axis_slot(fig, k) for k in 1:nax]
                for (k, gi) in enumerate(start:stop)
                    ax = axes[k]
                    ax.title = Visualization._rich_scientific_label(titles[gi])
                    ax.xlabel = Visualization._rich_scientific_label("z [m]")
                end
                colors = [Makie.wong_colors()[mod1(i, length(Makie.wong_colors()))] for i in 1:length(times)]
                for (idx, t) in enumerate(times)
                    s = snapshots[t]
                    lbl = Visualization._rich_legend_time(t)
                    c = colors[idx]
                    for (k, gi) in enumerate(start:stop)
                        _profile_line_dim!(axes[k], gi, s, scales, lbl, c)
                    end
                end
                for ax in axes
                    axislegend(ax, position = :rt, labelsize = 8)
                end
                save(joinpath(output_dir, @sprintf("profiles_all_fields_dimensional_%02d.png", pageno)), fig)
            end
        end
    end
    if !isempty(diagnostics.time)
        t = diagnostics.time .* scales.t_s
        thrust_N = diagnostics.thrust .* scales.thrust_n
        panels = Tuple{String, String, String, Function}[]
        push!(panels, ("Thrust", "t [s]", "F_T [N]", ax -> lines!(ax, t, thrust_N, color = :dodgerblue)))
        push!(panels, ("Adaptive step", "t [s]", "tau [s]", ax -> lines!(ax, t, diagnostics.tau .* scales.t_s, color = :darkorange)))
        push!(panels, ("Particle count", "t [s]", "N_particles", ax -> lines!(ax, t, diagnostics.particle_count, color = :forestgreen)))
        push!(panels, ("Field amplitude", "t [s]", "max |E_z| [V/m]", ax -> lines!(ax, t, diagnostics.max_abs_Ez .* scales.E_vm, color = :crimson)))
        if hasproperty(diagnostics, :vey_peak)
            push!(panels, ("peak |v_ey| (active zone)", "t [s]", "|v_ey| [m/s]", ax -> lines!(ax, t, diagnostics.vey_peak .* scales.v_ms, color = :teal)))
        end
        if hasproperty(diagnostics, :ey_rms)
            push!(panels, ("Ey RMS (active zone)", "t [s]", "rms(Ey) [V/m]", ax -> lines!(ax, t, diagnostics.ey_rms .* scales.E_vm, color = :brown)))
        end
        if hasproperty(diagnostics, :self_to_ext_B_ratio)
            push!(panels, ("max |Hself|/|Hext|", "t [s]", "ratio", ax -> lines!(ax, t, diagnostics.self_to_ext_B_ratio, color = :black)))
        end
        nrows_page = cld(Visualization._MAX_AXES_PER_FIG, 2)
        with_theme(Visualization._tnr_theme(12.0)) do
            pageno = 0
            for chunk in Iterators.partition(panels, Visualization._MAX_AXES_PER_FIG)
                pageno += 1
                nax = length(chunk)
                fig = Figure(; size = (900, 280 * nrows_page + 40))
                axes = [Visualization._axis_slot(fig, k) for k in 1:nax]
                for (k, (title_s, xlab_s, ylab_s, plotfn)) in enumerate(chunk)
                    ax = axes[k]
                    ax.title = Visualization._rich_scientific_label(title_s)
                    ax.xlabel = Visualization._rich_scientific_label(xlab_s)
                    ax.ylabel = Visualization._rich_scientific_label(ylab_s)
                    plotfn(ax)
                end
                save(joinpath(output_dir, @sprintf("diagnostics_time_series_dimensional_%02d.png", pageno)), fig)
            end
        end
        w_ma = let n = length(thrust_N)
            max(3, min(99, 2 * (n ÷ 40) + 1))
        end
        thrust_smooth = _thrust_box_smooth(thrust_N, w_ma)
        with_theme(Visualization._tnr_theme(12.0)) do
            figs = Figure(; size = (900, 400))
            axs = Axis(
                figs[1, 1];
                title = Visualization._rich_scientific_label("Thrust: raw + moving average (w=$w_ma) [N]"),
                xlabel = Visualization._rich_scientific_label("t [s]"),
                ylabel = Visualization._rich_scientific_label("F_T [N]"),
            )
            lines!(axs, t, thrust_N; color = (:dodgerblue, 0.25), linewidth = 0.8, label = "F_T (raw)")
            lines!(axs, t, thrust_smooth; color = :dodgerblue, linewidth = 1.6, label = "F_T (MA)")
            axislegend(axs; position = :rt, labelsize = 8)
            save(joinpath(output_dir, "diagnostics_thrust_smoothed_dimensional.png"), figs)
        end
    end
end

end

module CoreSolver

using ..PartCount
using ..NumericalFunctionsSPT
using ..PlasmaDynamics
using ..ParticleMovementSPT
using ..Visualization
using ..VisualizationDimensional
using ..DiagnosticsMetrics
using ..PaperScales

export run_simulation

const MI_XE_KG = 2.18e-25
# Fallback B [T] if `si_plot_scales === nothing`.
const B_REF_T = 0.012
default_plot_dir() = joinpath(@__DIR__, "..", "output", "figures")

"""Active set of `τ` constraints used by the adaptive timestep heuristic.

Members:
- `:neutral_cfl`  — `safety · h / v_a`              (axial neutral CFL).
- `:ion_cfl`      — `safety · h / max|v_iz|`        (axial ion CFL, synchronizes PIC subcycling).
- `:collision`    — `safety / max(ν_m)`             (electron-momentum collision frequency).
- `:hall`         — `safety · h / v_H`,
                    `v_H = α0 · |H_*| / (n · c)`     (Hall-whistler CFL for the explicit
                    part of `compute_current = ∂_z H`).

Electron-cyclotron CFL is intentionally absent: in the hybrid model electrons enter only via the
elliptic Ohm solve for `E_y`, they are not advanced by an explicit equation of motion, so a
`1/ω_ce` constraint would clamp `τ` for non-physical reasons.
"""
const _TAU_NAMES = (:neutral_cfl, :ion_cfl, :collision, :hall)

"""
    _resolve_tau_constraints(spec) -> Tuple{Vararg{Symbol}}

Internal helper that normalises the `tau_constraints` argument of `run_simulation`:
- a single `Symbol` such as `:full`, `:none`, `:fluid_only` or one of `_TAU_NAMES`,
- or a `Tuple`/`Vector` of `Symbol` entries from `_TAU_NAMES` (duplicates are removed).

Throws `ArgumentError` for unknown presets or unrecognised constraint names.
"""
function _resolve_tau_constraints(spec)
    if spec isa Symbol
        spec === :full          && return _TAU_NAMES
        spec === :none          && return ()
        spec === :fluid_only    && return (:neutral_cfl, :ion_cfl, :collision)
        spec in _TAU_NAMES      && return (spec,)
        throw(ArgumentError("Unknown tau_constraints preset :$spec; expected :full, :none, :fluid_only, " *
            "any of $_TAU_NAMES, or a Tuple/Vector of those names."))
    elseif spec isa Tuple || spec isa AbstractVector
        out = Symbol[]
        for k in spec
            k isa Symbol && k in _TAU_NAMES ||
                throw(ArgumentError("tau_constraints entries must be one of $_TAU_NAMES; got $k"))
            k in out || push!(out, k)
        end
        return Tuple(out)
    else
        throw(ArgumentError("tau_constraints must be a Symbol or a Tuple/Vector of Symbols; got $(typeof(spec))"))
    end
end

@inline function _steklov_dispatch!(f::AbstractVector{Float64}, radius::Int, passes::Int, boundary::Symbol)
    if boundary === :clamped
        NumericalFunctionsSPT.Steklov_smooth_clamped(f, radius, passes)
    elseif boundary === :reflect
        NumericalFunctionsSPT.Steklov_smooth(f, radius, passes; boundary = :reflect)
    else
        throw(ArgumentError("Steklov boundary must be :reflect ((43)) or :clamped ((44)); got $boundary"))
    end
    return nothing
end

"""
`mode = :case2` — hybrid EMHD (paper sec. 4): `E_y` from elliptic (Ohm); particles are advanced with a single
time level (`E^n`, `H^n`, `j^n`), then Faraday updates `H_ind^{n+1} = H_ind^n + τ·∂_z E_y^n` (or non-accumulating
equivalent). `E_z` from Eq. (38) uses one aligned layer (`H^n`, `j^n`, moments after temperature update).

If explicit `accumulate_induced_H` is not passed, defaults to `params.include_self_B`. With `accumulate_induced_H =
false`, Faraday gives `H_ind^{n+1} = τ·∂_z E_y^n` (no cross-step accumulation); with `true`, `H_ind^{n+1} = H_ind^n +
τ·∂_z E_y^n`. Induced field is never cleared at step start (that would mix quasi-static and accumulating induction).

With accumulation: each step adds `τ·∂_z E_y^n` to `H_ind` after the particle push.

`mode = :case1` — no induction (figures 4–7 style): E_y≡0, H_ind≡0, j≡0, E_z from `params.E_z0_dimless + E0_dimless`.

Steklov PIC smoothing: default half-window 20 on deposited moments only — **not** applied to `E`, `H`, or grid `j`.

# Adaptive timestep

`τ = min(active_constraints..., total_time - t)`. Active set is selected by `tau_constraints`
(a preset Symbol or an explicit collection of `_TAU_NAMES`). Per-constraint safety factors:

- `:neutral_cfl` — `tau_neutral_safety · h / v_a`     (default 1.0).
- `:ion_cfl`     — `tau_ion_safety · h / max|v_iz|`   (default 0.2).
- `:collision`   — `tau_collision_safety / max(ν_m)`  (default 0.5).
- `:hall`        — `tau_hall_safety · h / v_H` with `v_H = α0 |H_*|/(n c)` (default 1.0).

Presets:
- `:full` (default) — all four.
- `:fluid_only`   — `(:neutral_cfl, :ion_cfl, :collision)` (drop Hall, useful when Hall dominates τ).
- `:none`         — only `total_time - t` (debug).
- A single member of `_TAU_NAMES`, or a `Tuple`/`Vector` of names — explicit set.

`log_tau_constraint = true`: each periodic step report names the active limiter; a final summary
prints the share of steps each limiter dominated.
"""
function run_simulation(
    params::SimParams;
    total_time = 30.0,
    save_times = [10.0, 20.0, 30.0],
    do_plot = true,
    plot_output_dir::AbstractString = default_plot_dir(),
    si_plot_scales::Union{Nothing, PaperScales.CharacteristicScalesSI} = nothing,
    plot_profiles_dimensionless::Bool = true,
    mode::Symbol = :case2,
    E0_dimless::Float64 = 0.0,
    steklov_pic_half_width::Int = 1,
    steklov_pic_passes::Int = 5,
    steklov_pic_boundary::Symbol = :reflect,
    steklov_field_half_width::Int = 20,
    steklov_field_passes::Int = 5,
    steklov_field_boundary::Symbol = :reflect,
    accumulate_induced_H::Union{Nothing, Bool} = nothing,
    induced_H_damping::Float64 = 0.0,
    tau_constraints::Union{Symbol, Tuple, AbstractVector} = :full,
    tau_neutral_safety::Float64 = 1.0,
    tau_ion_safety::Float64 = 0.2,
    tau_collision_safety::Float64 = 0.5,
    tau_hall_safety::Float64 = 1.0,
    tau_min_floor::Float64 = 1e-14,
    log_tau_constraint::Bool = true,
)
    mode in (:case1, :case2) || throw(ArgumentError("mode must be :case1 or :case2 (got $mode)"))
    steklov_field_boundary in (:reflect, :clamped) ||
        throw(ArgumentError("steklov_field_boundary must be :reflect ((43))  or :clamped ((44)); got $steklov_field_boundary"))
    steklov_pic_boundary in (:reflect, :clamped) ||
        throw(ArgumentError("steklov_pic_boundary must be :reflect or :clamped; got $steklov_pic_boundary"))
    acc_ind = accumulate_induced_H === nothing ? params.include_self_B : accumulate_induced_H
    active_tau_constraints = _resolve_tau_constraints(tau_constraints)
    println("Adaptive τ constraints active: ",
            isempty(active_tau_constraints) ? "(none — only `total_time - t`)" :
            join(string.(active_tau_constraints), ", "))
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
    c_inv = params.c_inv
    H0_func = params.H0_func
    N1 = params.N1
    x_grid = range(0, L, length = M + 1)
    x_half = range(h / 2, L - h / 2, length = M)
    Hext_dim_max_pre = maximum(abs.(H0_func.(collect(x_grid))))
    # Dimensionless conversion factor for `v_ey ≈ -j/(e n)`: with `j_am2 = e n_ref v_ref`,
    # `j/(e n)` already has units of `v_ref` so the dimensionless factor is exactly 1. Kept as
    # a parameter so a non-Alfvén/plasma current scaling would still produce a correct diagnostic.
    vey_j_over_en = 1.0
    if si_plot_scales !== nothing
        _ps_vey = PaperScales.si_plot_physical_scales(si_plot_scales, Hext_dim_max_pre)
        vey_j_over_en = _ps_vey.j_am2 / (PaperScales.e_C * _ps_vey.n_m3 * _ps_vey.v_ms)
    end
    particles = Particle[]
    # PIC macroparticle weight (paper sec. 4): one channel length ⇒ total Σq/L = 1 dimensionless charge equiv.; independent of n_a(0) (set by neutral profile/source).
    q0 = params.pic_charge_factor * L / (N1 * M)
    for k in 1:M
        z0 = x_grid[k] + h / 2
        for s in 1:N1
            φ = 2π * (s - 1) / N1 + π / N1 + k * sqrt(2)
            vp = params.v_pic0
            push!(particles, Particle(z0, vp * cos(φ), vp * sin(φ), T_ion, q0, true))
        end
    end
    # Initial neutral density profile: constant along the whole channel.
    n_a_old = fill(n_a_left, M + 1)
    H_x_half = zeros(M)
    j = zeros(M + 1)
    E_y = zeros(M + 1)
    e0 = params.E_z0_dimless
    E_z = fill(e0, M + 1)
    E_z_term1 = zeros(M + 1)
    E_z_term2 = zeros(M + 1)
    E_z_term3 = zeros(M + 1)
    E_z_term4 = zeros(M + 1)
    T_e = fill(T_ion, M + 1)
    n_ion = zeros(M + 1)
    v_iy = zeros(M + 1)
    v_iz = zeros(M + 1)
    snapshots = Dict{Float64, NamedTuple}()
    thrust_time = Float64[]
    thrust_values = Float64[]
    tau_history = Float64[]
    particle_count_history = Int[]
    max_ez_history = Float64[]
    ion_to_neutral_ratio_history = Float64[]
    self_to_ext_B_ratio_history = Float64[]
    hall_beta_mean_history = Float64[]
    hall_beta_peak_history = Float64[]
    vey_mean_history = Float64[]
    vey_peak_history = Float64[]
    ey_mean_history = Float64[]
    ey_rms_history = Float64[]
    tau_start_max = 0.02 * h / max(v_a, 1e-8)
    n_vy_buf = zeros(M + 1)
    n_vz_buf = zeros(M + 1)
    n_T_buf = zeros(M + 1)
    # Cached H_ext on nodes: one evaluation of H0_func per step (avoids repeating H0_func.(x_grid)).
    H_ext_buf = zeros(M + 1)
    H_nodes_work = zeros(M + 1)
    H_total_work = zeros(M + 1)
    T_tilde_buf = similar(T_e)
    ν_m_work = zeros(M + 1)
    beta_e_buf = zeros(M + 1)
    tau_constraint_hits = Dict{Symbol, Int}(:total_time => 0)
    for k in active_tau_constraints
        tau_constraint_hits[k] = 0
    end
    last_tau_constraint = :total_time
    DiagnosticsMetrics.print_dimensionless_diagnostics(params)
    if si_plot_scales !== nothing
        _ps_chk = PaperScales.si_plot_physical_scales(si_plot_scales, Hext_dim_max_pre)
        # `compute_current` uses `j = ∂_z H` (Ampere/Maxwell form), so its natural scale is
        # `B/(μ0 L)`. The plotted scale `e n v_ref` matches **iff** `L = c/ω_pi`. The ratio below
        # equals 1 in pure Alfvén/skin-depth normalization; deviations measure the residual mismatch.
        j_maxwell = _ps_chk.B_t / (PaperScales.μ0_SI * _ps_chk.z_m)
        j_plasma  = PaperScales.e_C * _ps_chk.n_m3 * _ps_chk.v_ms
        ratio_jM_over_jP = j_maxwell / max(j_plasma, eps())
        println("  current-scale consistency: B/(μ0 L) / (e n v_ref) = ",
                round(ratio_jM_over_jP, sigdigits = 5),
                "   (=1 ⇔ L equals ion skin depth c/ω_pi; otherwise residual Alfvén mismatch)")
    end
    # Eq. (38): coefficient for ionization term in E_z.
    β_Ez_coef = params.λ_e_λΣ / max(params.ε, eps())
    t = 0.0
    step = 0
    # Sec. 3 regularization (denominators ~ 1% of mean PIC density).
    n_floor_physical = 0.01 / max(L, eps())
    T_cap = 30.0
    while t < total_time
        # Do NOT reset H_x_half at step start. Faraday update is applied after particle push at
        # end of step (`H^{n+1} = H^n + τ·∂_z E_y` with `acc_ind=true`, or `H^{n+1} = τ·∂_z E_y`
        # with `acc_ind=false`). Resetting here would mix quasi-static and full-induction closures.
        deposit_particles(particles, x_grid, n_ion, v_iy, v_iz, T_e, h, n_vy_buf, n_vz_buf, n_T_buf)
        # Sec. 3 Eq. (22); Steklov (43)/(44): deposit smoothing (default (43), reflect boundaries).
        _steklov_dispatch!(n_ion, steklov_pic_half_width, steklov_pic_passes, steklov_pic_boundary)
        _steklov_dispatch!(v_iy, steklov_pic_half_width, steklov_pic_passes, steklov_pic_boundary)
        _steklov_dispatch!(v_iz, steklov_pic_half_width, steklov_pic_passes, steklov_pic_boundary)
        _steklov_dispatch!(T_e, steklov_pic_half_width, steklov_pic_passes, steklov_pic_boundary)
        n_reg_min = max.(0.01 .* n_a_old, n_floor_physical)
        max_vz = max(maximum(abs, v_iz), 1e-12)
        @inbounds for i in 1:(M + 1)
            H_ext_buf[i] = H0_func(x_grid[i])
        end
        # Total magnetic channel field: H_* = H_ind + H_ext at nodes (`H_star_channel_nonneg` sums without clipping).
        H_nodes_work[1] = H_x_half[1]
        for i in 2:M
            H_nodes_work[i] = 0.5 * (H_x_half[i - 1] + H_x_half[i])
        end
        H_nodes_work[M + 1] = H_x_half[M]
        @. H_total_work = PlasmaDynamics.H_star_channel_nonneg(H_nodes_work + H_ext_buf)
        @. ν_m_work = PlasmaDynamics.local_nu_m(
            ν_m0, T_e, params.collision_model, H_total_work,
            params.alpha_B, params.ε, params.me,
        )
        ν_max = maximum(ν_m_work)
        # Single pass over interior nodes for the Hall-whistler speed `v_H = α0 |H_*|/(n c)`.
        hall_v_max = 0.0
        @inbounds for i in 2:M
            Hs = abs(H_total_work[i])
            n_loc = n_ion[i] + max(n_reg_min[i], n_floor_physical)
            vh = params.α0 * Hs * c_inv / max(n_loc, eps())
            hall_v_max = max(hall_v_max, vh)
        end
        τ = total_time - t
        last_tau_constraint = :total_time
        @inline function _consider!(name::Symbol, candidate::Float64)
            if isfinite(candidate) && candidate > 0 && candidate < τ
                τ = candidate
                last_tau_constraint = name
            end
            return nothing
        end
        for name in active_tau_constraints
            if name === :neutral_cfl
                _consider!(:neutral_cfl, tau_neutral_safety * h / max(v_a, 1e-12))
            elseif name === :ion_cfl
                _consider!(:ion_cfl, tau_ion_safety * h / max(max_vz, 1e-12))
            elseif name === :collision
                _consider!(:collision, tau_collision_safety / max(ν_max, 1e-8))
            elseif name === :hall
                _consider!(:hall, tau_hall_safety * h / max(hall_v_max, 1e-30))
            end
        end
        if !isfinite(τ) || τ <= 0
            τ = min(tau_start_max, 0.001 * h / max(v_a, 1e-12))
            last_tau_constraint = :recovery
            @warn "Time step τ invalid; reset to $τ"
        elseif τ < tau_min_floor
            τ_old = τ
            τ = min(1e-8, 0.01 * h / max(max_vz, 1e3), h / max(v_a, 1e-12), total_time - t)
            last_tau_constraint = :recovery
            @warn "Time step τ=$τ_old below floor ($tau_min_floor); using $τ"
        end
        tau_constraint_hits[last_tau_constraint] = get(tau_constraint_hits, last_tau_constraint, 0) + 1
        compute_current(j, H_x_half, h)
        kI_eff = kI
        intermediate_temperature(T_tilde_buf, T_e, n_ion, v_iz, j, n_a_old, τ, γ, mi, me, ν_m0, kI_eff, h;
            n_reg_min = n_reg_min, T_cap = T_cap,
            collision_model = params.collision_model,
            alpha_B = params.alpha_B, ε = params.ε, H_total = H_total_work)
        T_e .= T_tilde_buf
        for p in particles
            p.active || continue
            k0, k1, w0, w1 = interpolation_weights(p.z, x_grid)
            p.T = w0 * T_tilde_buf[k0] + w1 * T_tilde_buf[k1]
        end
        n_a_new = similar(n_a_old)
        neutrals_evolution(n_a_new, n_a_old, n_ion, τ, v_a, kI_eff, h, n_a_left)
        if mode === :case2
            E_y, H_x_half, j = electric_field_solver(E_y, H_x_half, j, n_ion, v_iz, T_e, τ, α, ν_m0, h, x_grid, H0_func, :j0, v_a, c_inv;
                n_reg_min = n_reg_min,
                collision_model = params.collision_model,
                alpha_B = params.alpha_B, ε = params.ε, me = params.me,
                H_ext_at_nodes = H_ext_buf, advance_induced_H = acc_ind, apply_faraday = false)
            compute_Ez(E_z, H_x_half, H_x_half, j, j, n_ion, T_e, v_iy, n_a_new, n_a_new, α0, ζ, kI_eff, v_a, h, params.λ_e_λΣ, β_Ez_coef, c_inv, τ, x_grid, H0_func, n_floor_physical;
                Ez_term1 = E_z_term1, Ez_term2 = E_z_term2, Ez_term3 = E_z_term3, Ez_term4 = E_z_term4, H_ext_at_nodes = H_ext_buf)
            E_z .+= E0_dimless
        else
            fill!(E_y, 0.0)
            fill!(H_x_half, 0.0)
            fill!(j, 0.0)
            fill!(E_z, e0 + E0_dimless)
            fill!(E_z_term1, 0.0)
            fill!(E_z_term2, 0.0)
            fill!(E_z_term3, 0.0)
            fill!(E_z_term4, 0.0)
        end
        counters = Counters(0, 0, 0)
        H_nodes_work[1] = H_x_half[1]
        for i in 2:M
            H_nodes_work[i] = 0.5 * (H_x_half[i - 1] + H_x_half[i])
        end
        H_nodes_work[M + 1] = H_x_half[M]
        @. H_total_work = PlasmaDynamics.H_star_channel_nonneg(H_nodes_work + H_ext_buf)
        @. ν_m_work = PlasmaDynamics.local_nu_m(
            ν_m0, T_e, params.collision_model, H_total_work,
            params.alpha_B, params.ε, params.me,
        )
        thrust_step = move_particles(particles, E_y, E_y, E_z, E_z, H_x_half, H_x_half, j, j, ν_m_work, ν_m_work, x_grid, x_half, τ, h, ε, mi, c_inv, H0_func, counters)
        if mode === :case2
            if acc_ind
                @inbounds for i in 1:M
                    H_x_half[i] += τ * (E_y[i+1] - E_y[i]) / h
                end
            else
                @inbounds for i in 1:M
                    H_x_half[i] = τ * (E_y[i+1] - E_y[i]) / h
                end
            end
            compute_current(j, H_x_half, h)
            if induced_H_damping > 0
                @. H_x_half *= (1.0 - induced_H_damping)
            end
        end
        H_nodes_work[1] = H_x_half[1]
        for i in 2:M
            H_nodes_work[i] = 0.5 * (H_x_half[i - 1] + H_x_half[i])
        end
        H_nodes_work[M + 1] = H_x_half[M]
        @. H_total_work = PlasmaDynamics.H_star_channel_nonneg(H_nodes_work + H_ext_buf)
        @. ν_m_work = PlasmaDynamics.local_nu_m(
            ν_m0, T_e, params.collision_model, H_total_work,
            params.alpha_B, params.ε, params.me,
        )
        beta_e_buf .= DiagnosticsMetrics.hall_parameter_electron.(Ref(params), H_total_work, ν_m_work)
        push!(thrust_time, t + τ)
        thrust_instant = thrust_step / τ
        push!(thrust_values, thrust_instant)
        push!(tau_history, τ)
        push!(max_ez_history, maximum(abs, E_z))
        active_ratio_cells = findall(>(0.0), n_a_new)
        if isempty(active_ratio_cells)
            push!(ion_to_neutral_ratio_history, 0.0)
        else
            push!(ion_to_neutral_ratio_history, maximum(n_ion[active_ratio_cells] ./ n_a_new[active_ratio_cells]))
        end
        push!(self_to_ext_B_ratio_history, maximum(abs.(H_nodes_work)) / max(maximum(abs.(H_ext_buf)), eps()))
        push!(hall_beta_mean_history, sum(beta_e_buf) / length(beta_e_buf))
        push!(hall_beta_peak_history, maximum(beta_e_buf))
        active_zone = findall(>(0.0), n_a_new)
        if isempty(active_zone)
            push!(vey_mean_history, 0.0)
            push!(vey_peak_history, 0.0)
            push!(ey_mean_history, 0.0)
            push!(ey_rms_history, 0.0)
        else
            n_safe_zone = max.(n_ion[active_zone], eps())
            # Diagnostic v_ey: dominant Hall drift is electronic, ions are largely radially confined; use
            # `v_ey ≈ -j_y/(e n_e)` (quasineutrality + ion azimuthal velocity ≈ 0). `vey_j_over_en` carries
            # the dimensionless→SI conversion of `j/(e n)` (=1 with `j_am2 = e n v`).
            vey_zone = .-vey_j_over_en .* j[active_zone] ./ n_safe_zone
            ey_zone = E_y[active_zone]
            push!(vey_mean_history, sum(vey_zone) / length(vey_zone))
            push!(vey_peak_history, maximum(abs.(vey_zone)))
            push!(ey_mean_history, sum(ey_zone) / length(ey_zone))
            push!(ey_rms_history, sqrt(sum(abs2, ey_zone) / length(ey_zone)))
        end
        new_particles_ionisation(particles, n_a_new, n_ion, x_grid, τ, kI_eff, v_a, T_ion; charge_factor = params.pic_charge_factor)
        remove_inactive_particles(particles, L, τ, kR)
        push!(particle_count_history, length(particles))
        for st in save_times
            if abs(t + τ - st) < τ / 2 && !haskey(snapshots, st)
                # Reuse end-of-step nodal fields (already consistent with H_x_half); avoids redundant
                # interpolation and extra `local_nu_m` / `H_star` allocations.
                H_ext = copy(H_ext_buf)
                n_safe = max.(n_ion, eps())
                v_ey = .-vey_j_over_en .* j ./ n_safe
                snapshots[st] = (;
                    z = collect(x_grid),
                    z_half = collect(x_half),
                    n_a = copy(n_a_new),
                    n_i = copy(n_ion),
                    v_iy = copy(v_iy),
                    v_ey = copy(v_ey),
                    v_iz = copy(v_iz),
                    T_e = copy(T_e),
                    E_y = copy(E_y),
                    E_z = copy(E_z),
                    E_z_term1 = copy(E_z_term1),
                    E_z_term2 = copy(E_z_term2),
                    E_z_term3 = copy(E_z_term3),
                    E_z_term4 = copy(E_z_term4),
                    j = copy(j),
                    H_x_half = copy(H_x_half),
                    H_x = copy(H_nodes_work),
                    H_ext = H_ext,
                    H_total = copy(H_total_work),
                    nu_m = copy(ν_m_work),
                )
            end
        end
        n_a_old .= n_a_new
        t += τ
        step += 1
        if step % 20 == 0 || t >= total_time
            tau_msg = log_tau_constraint ? ", τ=$(τ) (limited by :$(last_tau_constraint))" : ""
            println("Step $step, t=$t, #particles=$(length(particles))$tau_msg, ",
                "min_n=$(minimum(n_ion)), max_Ez=$(maximum(E_z)), ",
                "max(n_i/n_a)=$(ion_to_neutral_ratio_history[end]), ",
                "max(|Hself|/|Hext|)=$(self_to_ext_B_ratio_history[end]), ",
                "peak|v_ey|=$(vey_peak_history[end]), Ey_rms=$(ey_rms_history[end]), ",
                "nan=$(counters.nan), ",
                "exited=$(counters.exited_right), reflected=$(counters.reflected_left)")
        end
    end
    if log_tau_constraint && step > 0
        total_hits = sum(values(tau_constraint_hits))
        println("Adaptive τ — limiter share over $step steps:")
        for k in (:total_time, _TAU_NAMES..., :recovery)
            n_hits = get(tau_constraint_hits, k, 0)
            n_hits == 0 && continue
            pct = round(100 * n_hits / max(total_hits, 1), digits = 1)
            println("  :$k\t$n_hits / $total_hits ($(pct)%)")
        end
    end
    if do_plot
        diagnostics = (;
            time = copy(thrust_time),
            thrust = copy(thrust_values),
            tau = copy(tau_history),
            particle_count = copy(particle_count_history),
            max_abs_Ez = copy(max_ez_history),
            ion_to_neutral_ratio = copy(ion_to_neutral_ratio_history),
            self_to_ext_B_ratio = copy(self_to_ext_B_ratio_history),
            hall_beta_mean = copy(hall_beta_mean_history),
            hall_beta_peak = copy(hall_beta_peak_history),
            vey_mean = copy(vey_mean_history),
            vey_peak = copy(vey_peak_history),
            ey_mean = copy(ey_mean_history),
            ey_rms = copy(ey_rms_history),
        )
        plot_profiles_dimensionless && plot_results(snapshots, diagnostics, save_times; output_dir = plot_output_dir)
        Hext_max = Hext_dim_max_pre
        thrust_scale_n, scales = if si_plot_scales !== nothing
            ps = PaperScales.si_plot_physical_scales(
                si_plot_scales,
                Hext_max;
                effective_area_m2 = PaperScales.THRUSTER_EFFECTIVE_AREA_M2,
            )
            ps.thrust_n,
            VisualizationDimensional.DimensionalScales(;
                z_m = ps.z_m,
                n_m3 = ps.n_m3,
                v_ms = ps.v_ms,
                t_s = ps.t_s,
                Te_eV = ps.Te_eV,
                E_vm = ps.E_vm,
                j_am2 = ps.j_am2,
                B_t = ps.B_t,
                nu_s = ps.nu_s,
                thrust_n = ps.thrust_n,
            )
        else
            z_scale_m = 0.01
            v_scale_ms = 1.0e4
            scales_n_ref = 3.0e18
            b_scale_t = B_REF_T / max(Hext_max, 1e-6)
            thrust_b = MI_XE_KG * PaperScales.THRUSTER_EFFECTIVE_AREA_M2 * scales_n_ref * v_scale_ms^2
            thrust_b,
            VisualizationDimensional.DimensionalScales(;
                z_m = z_scale_m,
                n_m3 = scales_n_ref,
                v_ms = v_scale_ms,
                t_s = z_scale_m / v_scale_ms,
                Te_eV = 10.0,
                E_vm = v_scale_ms * b_scale_t,
                j_am2 = PaperScales.e_C * scales_n_ref * v_scale_ms,
                B_t = b_scale_t,
                nu_s = v_scale_ms / z_scale_m,
                thrust_n = thrust_b,
            )
        end
        plot_results_dimensional(snapshots, diagnostics, save_times, scales; output_dir = plot_output_dir)
        if !isempty(thrust_values)
            thrust_dim = thrust_values .* thrust_scale_n
            mean_thrust_mN = 1e3 * (sum(thrust_dim) / length(thrust_dim))
            peak_thrust_mN = 1e3 * maximum(thrust_dim)
            println("Thrust (dimensionalized): time-mean = $(round(mean_thrust_mN, sigdigits=5)) mN, peak = $(round(peak_thrust_mN, sigdigits=5)) mN")
        end
    end
    DiagnosticsMetrics.print_physicality_report(snapshots, params)
    return snapshots, thrust_time, thrust_values
end

end