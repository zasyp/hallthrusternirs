#=
Paper sec. 4 dimensionless setups (typical figures from hybrid EMHD):

  Figures 8, 9  →  `case2_full_emhd.jl` — field smoothing (43), ℓ=20, 5 passes.
  Figures 10–11 →  `case2_average_eq44.jl` — truncated smoothing (44), same ℓ and similarity.

Dimensional example (Kr, SI → paper sec. 2 groups): `spt70_krypton.jl`.

(Figures 4–7 induction-free cases correspond to `mode = :case1` + uniform E_z; reproduce with
 parameters from the paper — a dedicated `case1_*.jl` may be added separately.)

Run from repo root:
  julia --project examples/case2_full_emhd.jl
=#

println("Script ↔ figure hints: see header in `examples/article_examples.jl`.")
