using LinearAlgebra, Printf

# Константы
const k = 1.38e-23
const m_e = 9.11e-31
const e = 1.6e-19
const ε₀ = 8.85e-12
const r_Kr = 198e-12
const I_Kr = 13.99
const a₀ = 0.529e-10
const m_Kr = 83.798 * 1.66e-27

# Входные массивы (3 режима)
distances        = [10.0, 20.0, 30.0]
plasma_potentials = [199.3, 186.1, 75.5]
B_gauss          = [4.59, 29.2, 117.2]
I_e_mA           = [2.59, 2.23, 0.5]
I_i_mA           = [0.108, 0.475, 2.19]
T_e_eV           = [4.0, 7.01, 2.47]
τ_el             = [0.764e-7, 0.506e-7, 1.84e-7]
τ_inel           = [7.23e-6, 1.44e-6, 2.44e-6]
n_n_ref          = [3.62e19, 3.03e19, 2.84e18]   # уже посчитанная концентрация нейтралов

T_n_K  = 400.0
d_Kr   = 360e-12           # kinetic diameter
V̇      = 0.55e-6          # м³/с
ρ_Kr   = 3.749             # кг/м³ (0°C, 1 атм)

# Геометрия канала
D_mean  = 56e-3
w_ch    = 28e-3
S_ch    = π*((D_mean + w_ch)^2 - (D_mean - w_ch)^2)/4

# Перевод в СИ
B        = B_gauss / 1e4
I_e      = I_e_mA * 1e-3 ./ S_ch
I_i      = I_i_mA * 1e-3 ./ S_ch

# Скорости (тепловые / направленные)
v_e_th   = sqrt.(8 * k * T_e_eV * 11604.5 ./ (π * m_e))
v_i      = sqrt.(e * (200 .- plasma_potentials) ./ (2 * m_Kr))
v_n      = sqrt(3 * k * T_n_K / m_Kr)

T_i_eV   = (m_Kr .* v_i.^2 ./ (2 * k)) / 11604.5

n_i      = I_i ./ (v_i * e)
n_e      = n_i
n_n      = n_n_ref

# Debye, плазменная частота, число частиц в сфере Дебая
λ_D      = sqrt.(ε₀ * k * T_e_eV * 11604.5 ./ (n_e * e^2))
N_D      = n_e .* (4π/3 .* λ_D.^3)
ω_pe     = sqrt.(n_e * e^2 ./ (ε₀ * m_e))

# Кулоновский логарифм (приближённо ee)
b_min    = e^2 ./ (4π * ε₀ * T_e_eV * 11604.5 * e)
lnΛ      = log.(λ_D ./ b_min)

# Циклотронные частоты
ω_ce     = e * B / m_e
ω_ci     = e * B / m_Kr

# Larmor (используем v_th / √2 для v⊥)
v⊥_e     = v_e_th / √2
v⊥_i     = v_i   / √2
ρ_e      = m_e * v⊥_e ./ (e * B)
ρ_i      = m_Kr * v⊥_i ./ (e * B)

# Поляризуемость
α        = (r_Kr / 0.62)^3

# Относительная энергия ион-нейтрал (эВ)
E_rel    = (m_Kr * (v_i .- v_n).^2 / 2) * 6.241509e18

# Сечения
σ_nn     = π * d_Kr^2
σ_Coul   = 2.87e-18 * lnΛ ./ T_e_eV.^2
σ_pol = 2 * π * √2 * a₀^2 .* sqrt.((α / a₀^3) .* (I_Kr ./ E_rel))
σ_ct      = σ_pol / 2

# Частоты столкновений (основные)
ν_ee     = √2 * σ_Coul .* n_e .* v_e_th
ν_ei     = σ_Coul .* n_i .* v_e_th
ν_en     = 1 ./ τ_el + 1 ./ τ_inel
ν_e_tot  = ν_ee + ν_ei + ν_en

ν_ii     = √2 * σ_Coul .* n_i .* v_i
ν_in     = 1.5 * (v_i .- v_n) .* n_n .* σ_ct
ν_i_tot  = ν_ii + ν_in + ν_ei           # ee↔ii приближение

ν_nn     = n_n .* v_n .* σ_nn
ν_n_tot  = ν_en + ν_in + ν_nn

# Длины свободного пробега
λ_e      = v_e_th ./ ν_e_tot
λ_i      = v_i    ./ ν_i_tot
λ_n      = v_n    ./ ν_n_tot

# Параметры Холла
β_e      = ω_ce ./ ν_e_tot
β_i      = ω_ci ./ ν_i_tot

# Продольная и поперечная проводимость (классика)
σ_par  = (n_e .* e^2) ./ (m_e .* (ν_en .+ ν_ei))
σ_perp = σ_par .* (β_e ./ (β_e.^2 .+ 1))

# Тяга и КПД (для третьего режима как в оригинале)
thrust   = I_i[3] * S_ch * sqrt(2 * m_Kr * 200 / e)
η_u      = thrust * v_i[3] / 540

# Вывод ключевых величин (пример)
println("S_channel     = ", round(S_ch*1e4, digits=2), " см²")
println("n_e (1–3)     = ", [round(n*1e-18,digits=2) for n in n_e], " ⋅10¹⁸ м⁻³")
println("λ_D (мм)      = ", round.(λ_D*1e3, digits=3))
println("β_e           = ", round.(β_e, digits=1))
println("Тяга (мН)     = ", round(thrust*1000, digits=3))
println("η_u           = ", round(η_u*100, digits=1), " %")