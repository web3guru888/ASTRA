# MHD Sweep Campaigns — April 2026

## Overview
Twelve Athena++ MHD+self-gravity simulations characterising the dependence of
filament fragmentation on plasma β and Mach number M.

## Campaigns
- **C1C2_beta_sweep**: β ∈ {0.22, 0.32, 0.50, 0.70, 1.00, 1.50, 2.00} at M=3.0 (7 sims)
- **C3_mach_sweep**: M ∈ {1.0, 2.0, 3.0, 4.0, 5.0} at β=0.85 (5 sims)

## Grid
256 × 64 × 64 cells, x₁∈[−8,8] λ_J, x₂/x₃∈[−2,2] λ_J, 32 MPI ranks/sim

## Key Results
- λ_frag = 2.0 λ_J universally (seeded mode dominates)
- β is the dominant parameter: C_final varies ×4.3 across β=0.22–2.00
- M is weak: C_final varies <20% across M=1–5
- Magnetic stability threshold: β_crit ≈ 0.667 for seeded λ=2.0 mode
- W3 prediction: λ_frag ≈ 0.20 pc ≈ 21\" at d=1.95 kpc (β=0.85, λ_J=0.10 pc)

## Files
- `report/ASTRA_Sweep_Report_Apr2026.md` — full analysis report
- `report/ASTRA_Sweep_Summary_Apr2026.json` — machine-readable summary
- `figures/` — 4 figures (PNG + PDF)
- `scripts/` — launch, analysis, and report-generation scripts

## Simulation Data
Raw HDF5 snapshots on astra-climate:
  `/home/fetch-agi/filament_sweeps/C1C2_beta_sweep/`  (147 files)
  `/home/fetch-agi/filament_sweeps/C3_mach_sweep/`    (105 files)
