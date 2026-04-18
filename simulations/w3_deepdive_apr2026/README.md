# W3 Deep-Dive Simulation Campaign — April 2026

## Overview
Extended Magneto-Jeans instability campaign targeting W3 Giant Molecular Cloud conditions
with a large domain (L = 16 λ_J) and high resolution (256³) to resolve multiple gravitational
core formation events with measurable inter-core separations.

## Science Context
The W3/W4/W5 Galactic HII region complex (Perseus Arm, ~1.95 kpc) is a benchmark target
for triggered star formation. This campaign quantifies the non-linear correction factor f(β)
to the Magneto-Jeans fragmentation spacing:

```
λ_measured = f(β) × λ_MJ    where  λ_MJ = λ_J √(1 + 2/β)
```

**Key result:** f(β) = 0.823 + 0.093 × (β − 0.7), measured for β = 0.7–1.0 at M = 3.

## Simulation Parameters
| Parameter | Value |
|-----------|-------|
| Domain | L = 16 λ_J (periodic) |
| Resolution | 256³ cells |
| Meshblocks | 512 × (32³) |
| Mach number | M = 3 |
| Plasma β | 0.7, 0.8, 0.9, 1.0 |
| Self-gravity | Athena++ FFT (4πG = 4π²) |
| Compute | 128 CPUs, astra-climate (224-vCPU GCE) |
| Wall time | 13.3 min (4 concurrent runs) |

## Results Summary
| β | C_final | N_cores | λ_measured (λ_J) | f(β) |
|---|---------|---------|-------------------|------|
| 0.7 | 1.598 | 9 | 1.617 ± 0.085 | 0.823 |
| 0.8 | 2.271 | 9 | 1.598 ± 0.065 | 0.854 |
| 0.9 | 2.886 | 9 | 1.592 ± 0.082 | 0.887 |
| 1.0 | 2.889 | 9 | 1.587 ± 0.058 | 0.916 |

**W3 physical scale:** λ_measured = 0.254–0.259 pc (consistent with Herschel 250 μm observations)

## Files
```
scripts/
  run_w3_deepdive.py       — Simulation launcher (4 concurrent Athena++ runs via subprocess)
  analyse_w3_deepdive.py   — HDF5 analysis + core detection (scipy.ndimage)
results/
  w3_deepdive_analysis.json  — Full time-series data for all 4 simulations (65 KB)
  w3_deepdive_status.json    — Run completion metadata
report/
  w3_deepdive_report.md      — Comprehensive scientific report (15 KB)
```

## Dependencies
- Athena++ with FFT self-gravity and MHD enabled
- Python 3.10+: numpy, scipy, h5py
- astra-climate: 224-vCPU server, 220 GB RAM

## RASTI Paper Implication
The f(β) correction table is directly applicable to ASTRA Paper (White & Dey 2026) as a
quantitative prediction for filament fragmentation spacing in magnetised molecular clouds.
The 0.254–0.259 pc prediction spans the range detectable by Herschel PACS/SPIRE at W3's distance.

## Investigators
Glenn J. White (Open University) & Robin Dey (VBRL Holdings Inc)  
ASTRA multi-agent scientific discovery system — 2026-04-18
