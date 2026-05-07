using LinearAlgebra

module NumericalFunctionsSPT

export solve_tridiagonal, solve_tridiagonal!, interpolation_weights, smooth_field,
    Steklov_smooth, Steklov_smooth!, Steklov_smooth_clamped

"""
    solve_tridiagonal!(x, a, b, c, d, cp, dp) -> x

Mutating Thomas sweep over the same system as [`solve_tridiagonal`](@ref):

    b[1] x[1] +   c[1] x[2]                    = d[1]
    a[i-1] x[i-1] + b[i] x[i] + c[i] x[i+1]    = d[i],  2 ≤ i ≤ n-1
                                a[n-1] x[n-1] + b[n] x[n] = d[n]

Output `x` (length `n`) is written in place. `cp` and `dp` are caller-supplied scratch
buffers (length ≥ `n - 1`); reusing them across calls eliminates per-step allocations
inside the hot loop. No pivoting (matrix is diagonally dominant by construction).
"""
function solve_tridiagonal!(
    x::AbstractVector{Float64},
    a::AbstractVector{Float64},
    b::AbstractVector{Float64},
    c::AbstractVector{Float64},
    d::AbstractVector{Float64},
    cp::AbstractVector{Float64},
    dp::AbstractVector{Float64},
)
    n = length(b)
    @assert length(x) == n && length(a) == n - 1 && length(c) == n - 1 && length(d) == n
    @assert length(cp) >= n - 1 && length(dp) >= n - 1
    @inbounds begin
        cp[1] = c[1] / b[1]
        dp[1] = d[1] / b[1]
        for i in 2:(n - 1)
            denom = b[i] - a[i - 1] * cp[i - 1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
        end
        x[n] = (d[n] - a[n - 1] * dp[n - 1]) / (b[n] - a[n - 1] * cp[n - 1])
        for i in (n - 1):-1:1
            x[i] = dp[i] - cp[i] * x[i + 1]
        end
    end
    return x
end

"""
    solve_tridiagonal(a, b, c, d) -> Vector{Float64}

Thomas algorithm for the tridiagonal system

    b[1] x[1] +   c[1] x[2]                    = d[1]
    a[i-1] x[i-1] + b[i] x[i] + c[i] x[i+1]    = d[i],  2 ≤ i ≤ n-1
                                a[n-1] x[n-1] + b[n] x[n] = d[n]

Inputs are checked: `length(a) == length(c) == n - 1`, `length(b) == length(d) == n`.

`O(n)` operations, no pivoting (the elliptic `E_y` matrix in
`PlasmaDynamics.electric_field_solver` is diagonally dominant by construction).

Allocating wrapper around [`solve_tridiagonal!`](@ref); use the in-place variant
to avoid temporaries in tight loops.
"""
function solve_tridiagonal(
    a::AbstractVector{Float64},
    b::AbstractVector{Float64},
    c::AbstractVector{Float64},
    d::AbstractVector{Float64},
)
    n = length(b)
    @assert length(a) == n - 1 && length(c) == n - 1 && length(d) == n
    cp = similar(c)
    dp = similar(d)
    x = zeros(n)
    return solve_tridiagonal!(x, a, b, c, d, cp, dp)
end

"""
    interpolation_weights(x, x_grid) -> (k0, k1, w0, w1)

Linear (CIC) interpolation indices and weights for point `x` on the uniform mesh `x_grid`
(`length(x_grid) = M + 1`, spacing `h`):

    f(x) ≈ w0 · f[k0] + w1 · f[k1],   k1 = k0 + 1,   w0 + w1 = 1.

Out-of-domain points are clamped to the boundary cell with `(w0, w1) = (1, 0)` so the
caller falls back to a single-node value at `x_grid[1]` or `x_grid[end]`.

Used by `deposit_particles`, `move_particles`, and per-particle temperature interpolation
in `run_simulation`.
"""
function interpolation_weights(x::Float64, x_grid::AbstractVector{Float64})
    M = length(x_grid) - 1
    h = x_grid[2] - x_grid[1]
    x_min = x_grid[1]
    x_max = x_grid[end]
    if x <= x_min
        return 1, 2, 1.0, 0.0
    elseif x >= x_max
        return M, M + 1, 1.0, 0.0
    else
        k0 = floor(Int, (x - x_min) / h) + 1
        k0 = min(k0, M)
        w1 = (x - x_grid[k0]) / h
        w0 = 1.0 - w1
        return k0, k0 + 1, w0, w1
    end
end

"""
    smooth_field(f, window) -> f

In-place uniform box average of half-width `window` with prefix-sum acceleration
(`O(n)` regardless of window size). Boundary cells use truncated windows
(no reflection). Currently unused by the solver; kept for ad-hoc post-processing.
"""
function smooth_field(f::AbstractVector{Float64}, window::Int)
    n = length(f)
    g = similar(f)
    prefix = zeros(Float64, n + 1)
    @inbounds for i in 1:n
        prefix[i + 1] = prefix[i] + f[i]
    end
    @inbounds for i in 1:n
        left = max(1, i - window)
        right = min(n, i + window)
        g[i] = (prefix[right + 1] - prefix[left]) / (right - left + 1)
    end
    f .= g
    return f
end

"""
    Steklov_smooth!(f, buf, radius=1, passes=5; boundary=:reflect) -> f

Mutating variant of [`Steklov_smooth`](@ref). `buf` is a caller-supplied scratch
vector with `length(buf) == length(f)`; reusing it across calls eliminates
per-step allocations. The result is always written back into `f` (the function
internally ping-pongs between `f` and `buf` to skip an extra copy).
"""
function Steklov_smooth!(
    f::AbstractVector{Float64},
    buf::AbstractVector{Float64},
    radius::Int = 1,
    passes::Int = 5;
    boundary::Symbol = :reflect,
)
    n = length(f)
    @assert length(buf) == n
    radius = max(radius, 1)
    src = f
    dst = buf
    for _ in 1:passes
        @inbounds for i in 1:n
            s = 0.0
            for k in -radius:radius
                j = i + k
                # Reflection at boundaries: …, 2, 1, 2, …; …, n-1, n, n-1, …
                if boundary === :reflect
                    if j < 1
                        j = 2 - j
                    elseif j > n
                        j = 2n - j
                    end
                    j = clamp(j, 1, n)
                else
                    j = clamp(j, 1, n)
                end
                s += src[j]
            end
            dst[i] = s / (2radius + 1)
        end
        src, dst = dst, src
    end
    if src !== f
        copyto!(f, src)
    end
    return f
end

"""
Steklov smoothing (formula (43); Gavrikov–Tauyrsky-style preprint, 2021):
  f̂(z) = 1/(2η) ∫_{z-η}^{z+η} f̃(x) dx,  0 ≤ z ≤ L,
where f̃ is the even extension of f to [-L,L] then 2L-periodic continuation.
On a grid with spacing h and η = r·h this is a uniform box average over `2r+1` nodes
with reflected indices at boundaries.

`radius`: half-window in mesh points (default 1, i.e. η = h).
`passes`: successive applications (paper recommends 5).

Allocating wrapper around [`Steklov_smooth!`](@ref); use the in-place variant
to avoid per-call temporaries in tight loops.
"""
function Steklov_smooth(
    f::AbstractVector{Float64},
    radius::Int = 1,
    passes::Int = 5;
    boundary::Symbol = :reflect,
)
    buf = similar(f)
    return Steklov_smooth!(f, buf, radius, passes; boundary = boundary)
end

"""
Modified smoothing (formula (44)) — truncated-interval average:
  f̂(z) = 1/(b(z)-a(z)) ∫_{a(z)}^{b(z)} f(x) dx,
with a(z) = max(z-η,0), b(z) = min(z+η,L). On the mesh: truncated uniform averaging.
"""
function Steklov_smooth_clamped(
    f::AbstractVector{Float64},
    radius::Int = 1,
    passes::Int = 5,
)
    n = length(f)
    radius = max(radius, 1)
    buf = similar(f)
    src = f
    dst = buf
    for _ in 1:passes
        @inbounds for i in 1:n
            lo = max(1, i - radius)
            hi = min(n, i + radius)
            s = 0.0
            for j in lo:hi
                s += src[j]
            end
            dst[i] = s / (hi - lo + 1)
        end
        src, dst = dst, src
    end
    if src !== f
        copyto!(f, src)
    end
    return f
end

end
