# Peer Review Response Campaign — April 2026

## Campaign Overview
314 Athena++ MHD simulations on astra-climate (224 vCPU) addressing referee concerns
T1/T2, T3, T9 for the RASTI filament spacing paper.
Completed: 24 April 2026 02:58 UTC | Runner: Ray (13 concurrent × 16 MPI)

## Phases
| Phase | N | Configuration | Result |
|---|---|---|---|
| 1 | 80 | Near-critical, longitudinal B, isothermal | All FRAG |
| 2 | 96 | Perpendicular B, isothermal | All FRAG |
| 3 | 108 | Oblique B θ=30/45/60°, isothermal | All FRAG |
| 4 | 30 | Longitudinal B, adiabatic γ=5/3 | All STABLE (5h timeout) |

## Key Results
- T1/T2: 100% fragmentation rate in all isothermal sims (f=1.0-1.2, β=0.3-1.0, M=1-2)
- T3: Perpendicular B accelerates fragmentation (t_frag 0.39 vs 1.15 t_J for longitudinal)
- T9: Zero fragmentation in 30 adiabatic γ=5/3 sims — isothermal is the conservative limit
- Oblique: t_frag decreases with θ: 0.604→0.508→0.454 t_J (30°→45°→60°)
