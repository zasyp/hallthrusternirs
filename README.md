# HallThrusterNIRS

Julia implementation of a **1D–2V hybrid EMHD / particle-in-cell** model for discharge channel physics in **stationary plasma thrusters (Hall-effect type, SPT)**. The formulation follows the analytical structure and equation numbering of the Gavrikov–Tauyrsky preprint (2021-class) on hybrid EMHD: fluid electrons with local temperature and Spitzer-type resistivity, kinetic ions as macroparticles with **(v_y, v_z)** in the plane normal to the axial direction **z** (Hall / azimuth is **y**).

This README summarizes the **physics**, the **effective “chemistry”** (source terms, not a full reaction mechanism), **dimensionless scaling**, and **how to run** the code.

---

## 1. Geometry and variables

- **1D in space:** channel axis **0 ≤ z ≤ L** (normalized to **L = 1** in the solver).
- **2D in ion velocity:** components **(v_y, v_z)**; **y** plays the role of the closed Hall (azimuthal) direction after reduction.
- **Electron temperature** **T_e(z)** on the same grid as densities; used in Ohm’s law and collision frequency.
- **Magnetic field:** normal **H** (dimensionless notation in the paper-related code). The code splits **H_*(z) = H_ind(z) + H_ext(z)** (induced vs applied).
- **Ion density** from PIC deposit **n_i**, **neutral density** **n_a**, **current** **j_y**.

---

## 2. Core physics (conceptual)

### 2.1 Electrons (fluid, reduced massless-electrons limit)

Electrons are treated as a **conducting fluid** with:

- **Resistivity / collision frequency** modeled as proportional to **ν_e / n** with **ν_e** from:
  - **Spitzer** scaling **ν_m ∝ ν_{m0}/T_e^{3/2}** with a floor on **T**, or
  - **Constant** effective **ν_e ≈ const** (anomalous plateau).
- Optional **Bohm-type anomalous scattering** **∝ α_B · ω_ce** on top of the base model (parameter **α_B**; Hagelaar / Boeuf–Garrigues style references appear in comments).
- **Ohm’s law in the Hall direction** yields an **elliptic boundary-value problem for E_y** coupled to **j_y**, **v_z**, and **H_*** (system (36) in the manuscript, semi-implicit discretization in code).
- **Axial electric field E_z** from an **algebraic closure (38)**: balance of drift, ionization contribution, Hall current correction, and pressure-gradient term projected on **z** (stored as four diagnostic arrays).

### 2.2 Ions and neutrals

- **Ions:** **PIC macroparticles** stepped in **(v_y, v_z)** using **H_***, **E**, and **ν_m(T_e, …)** consistent with electron model (paper system **(11)**).
- **Neutrals:** upstream advection with speed **v_a**, sink by ionization **∝ k_I n_a n_i** (macrosources use a stabilized **n_i** to curb numerical avalanche).
- **Particle loss:** recombination-like removal with coefficient **k_R**.

### 2.3 Induced field and solver modes

- **`mode = :case2`:** hybrid EMHD (paper sec. 4 style): solves for **E_y**, updates induced **H**, evaluates **E_z**.
- **`include_self_B` / accumulation flags:** toggle whether induced **H** is **Faraday-accreted** from step to step or treated in the reduced “instantaneous incremental” regime before the elliptic update (mirror of **`advance_induced_H`** in **`electric_field_solver`**).

- **`mode = :case1`:** induction-free (**E_y ≡ 0**, **H_ind ≡ 0**, **j ≡ 0**), uniform axial **E_z** from parameters — comparable to textbook figures **4–7** variants.

PIC and field quantities are smoothed with **Steklov** kernels: **reflecting (43)** or **truncated (44)** at domain ends.

---

## 3. “Chemistry” in this code

There is **no detailed reaction chemistry** (no excited states, wall sputtering, multiply charged ions, CHEMKIN-like network).

| Effective process       | Mathematical form                           | Interpretation                                               |
|-------------------------|--------------------------------------------|--------------------------------------------------------------|
| **Ionization**          | Sources **∝ τ k_I n_a n_{i,eff}** volume  | Collapsed ionization coefficient **k_I** × neutral × ion densities |
| **Volume loss / rec.** | **k_R τ** stochastic removal              | Stand-in for recombination or wall return                    |
| **Electron heating**    | Ohmic **∝ ν j²/n** + ionization terms in **T_e** advance | Fluid energy budget                              |
| **Species choice**       | **`m_i`**, **`β₀`**, densities, voltage** | Xenon vs krypton differs by mass and SI choice in `PaperScales` |

To compare gases, rerun **`paper_dimensionless_from_si`** (or **`sim_params_from_si_scales`**) with the target atomic mass — see **`examples/spt70_krypton.jl`** and xenon analogue examples.

---

## 4. Dimensionless groups (`PaperScales`)

From **`CharacteristicScalesSI`** plus ionization-volume coefficient **β₀** (SI):

- Dimensionless combinations **ε**, **ζ**, **κ**, **ξ**, **ν_{m0}**, **k_I** as in manuscript sec. 2.
- Maps to **`SimParams`**: closures **α**, **α₀**, Hall parameters, etc.

Diagnostics print **Hall β_e**, thrust scalings, optionally **SI** plots via **`si_plot_physical_scales`**.

---

## 5. Package layout

| Path | Contents |
|------|----------|
| `src/HallThrusterNIRS.jl` | Top-level exports |
| `src/physics.jl` | EMHD closures, PIC push, neutral advection |
| `src/numerical.jl` | Steklov (43)/(44), tridiagonal solves |
| `src/dimensionless.jl` | Diagnostics + **`PaperScales`** includes |
| `src/paper_scales/` | SI / dimensionless conversion |
| `src/solver.jl` | Time marching, **`run_simulation`**, Makie figures |

---

## 6. Usage

Activate the repo as a Julia package (ensure **`Project.toml`** has **`name = "HallThrusterNIRS"`**):

```julia
using Pkg; Pkg.instantiate(); using HallThrusterNIRS
```

Run examples from the repo root:

```shell
julia --project examples/case2_full_emhd.jl
julia --project examples/spt70_krypton.jl
```

Figure ↔ script hints are in **`examples/article_examples.jl`**.

---

## 7. Equation references

Equation numbers **(11), (22), (33), (36), (38), (43), (44)** in comments match the cited **hybrid EMHD manuscript** https://keldysh.ru/papers/2021/prep2021_35.pdf
