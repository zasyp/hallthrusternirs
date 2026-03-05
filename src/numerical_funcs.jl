using LinearAlgebra

module NumericalFunctionsSPT

export solve_tridiagonal, interpolation_weights, smooth_field, Steklov_smooth

function solve_tridiagonal(
    a::Vector{Float64},
    b::Vector{Float64},
    c::Vector{Float64},
    d::Vector{Float64}
)
    n = length(b)
    @assert length(a) == n-1 && length(c) == n-1 && length(d) == n

    cp = similar(c)
    dp = similar(d)

    cp[1] = c[1] / b[1]          # без минуса, согласовано с обратным ходом
    dp[1] = d[1] / b[1]

    for i in 2:n-1
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    end

    x = zeros(n)
    x[n] = (d[n] - a[n-1] * dp[n-1]) / (b[n] - a[n-1] * cp[n-1])   # исправлено

    for i in n-1:-1:1
        x[i] = dp[i] - cp[i] * x[i+1]
    end

    return x
end

function interpolation_weights(
    x::Float64,
    x_grid::AbstractVector{Float64}
)
    M = length(x_grid) - 1
    h = x_grid[2] - x_grid[1]

    x_min = x_grid[1]
    x_max = x_grid[end]

    if x <= x_min
        return 1, 2, 1.0, 0.0
    elseif x >= x_max
        return M, M+1, 1.0, 0.0
    else
        k0 = floor(Int, (x - x_min) / h) + 1
        k0 = min(k0, M)
        w1 = (x - x_grid[k0]) / h
        w0 = 1.0 - w1
        return k0, k0+1, w0, w1
    end
end

function smooth_field(
    f::AbstractVector{Float64},
    window::Int
)
    n = length(f)
    g = copy(f)
    for i in 1:n
        left = max(1, i - window)
        right = min(n, i + window)
        g[i] = sum(@view f[left:right]) / (right - left + 1)
    end
    f .= g
    return f
end

function Steklov_smooth(
    f::AbstractVector{Float64},
    window::Int,
    h::Float64,
    L::Float64,
    smoothing_passes::Int = 3
)
    n = length(f)
    f_smooth = copy(f)

    for _ in 1:smoothing_passes
        for i in 1:n
            zi = (i - 1) * h
            left = max(1, i - window)
            right = min(n, i + window)

            total_weight = 0.0
            sum_val = 0.0

            for j in left:right
                zj = (j - 1) * h
                dist = abs(zi - zj)
                if dist <= window * h
                    weight = 1.0 - dist / (window * h)
                    sum_val += f[j] * weight
                    total_weight += weight
                end
            end

            if total_weight > 0
                f_smooth[i] = sum_val / total_weight
            end
        end
        f .= f_smooth            # обновляем исходный массив
    end
    return f
end

end # module