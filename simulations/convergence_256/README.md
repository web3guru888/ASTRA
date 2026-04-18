# ASTRA Resolution Convergence Suite — 256³

Three Athena++ MHD driven-turbulence simulations at 256³ resolution to test
convergence of key diagnostics against the existing 128³ parameter sweep.

**Generated**: 2026-04-17 by ASTRA PA  
**Author**: Glenn J. White (Open University)

---

## Simulations

| Name | β | M_sonic | Meshblocks | Purpose |
|------|---|---------|------------|---------|
| M3_beta1.0_256 | 1.0 | 3 | 512 | Most isotropic case |
| M3_beta0.1_256 | 0.1 | 3 | 512 | Highly magnetised (dynamo-quenched) |
| M1_beta1.0_256 | 1.0 | 1 | 512 | Subsonic case |

**Resolution**: 256³ cells, meshblock 32³ → 512 meshblocks  
**MPI ranks**: 18 (set via `NPROCS=18` or `--nprocs 18`)  
**Estimated wall-time**: 6 h/sim × 3 = 18–24 h total  
**HDF5 storage**: ~37 GB/sim (40 snapshots at dt=0.05), ~110 GB total

---

## Quick Start

```bash
cd simulations/convergence_256/

# 1. Run all three sims sequentially (foreground, ~18-24 h total)
ATHENA_BIN=/path/to/bin/athena ./run_convergence_suite.sh

# 2. Or run one sim at a time in the background
ATHENA_BIN=/path/to/bin/athena ./run_convergence_suite.sh \
    --background --sim M3_beta1.0_256

# 3. Monitor progress while sims run
./monitor_convergence.sh --interval 60

# 4. After completion: generate convergence_report.pdf + conclusion.txt
python3 analyse_convergence.py \
    --output-dir ./convergence_output \
    --ref-dir    /path/to/sweep_output
```

---

## Outputs (per simulation)

After each run, in `convergence_output/{sim_name}/`:

| File | Description |
|------|-------------|
| `job.log` | Full Athena++ stdout — monitor with `tail -f` |
| `*.hst` | History file (dt=0.005): mass, momentum, KE, ME components |
| `*.hdf5` | 3D primitive-variable snapshots (dt=0.05, ~40 files) |
| `diagnostics.json` | Saturated-state scalars + convergence vs 128³ |
| `energy_history.dat` | ASCII: time, KE, MEz, ME_perp, ME_tot, M_A, MEz_ratio |

---

## Key Diagnostics to Monitor in `job.log`

```
Expected evolution patterns:

M3_beta1.0_256:
  - KE saturates at t ≈ 1.0
  - ME grows slowly (dynamo)
  - Final MEz/(MEx+MEy) ≈ 3–10
  - Final M_A ≈ 1.0

M3_beta0.1_256:
  - KE saturates at t ≈ 1.5
  - ME shows little growth (dynamo quenched)
  - Final MEz/(MEx+MEy) ≈ 50–100
  - Final M_A ≈ 0.6

M1_beta1.0_256:
  - KE and ME saturate early (t ≈ 0.5)
  - Final MEz/(MEx+MEy) ≈ 20–50
  - Final M_A ≈ 0.8
```

> ⚠️ If you see >50% deviations from these, something may be wrong.

---

## Convergence Criterion

All key diagnostics within **10%** between 128³ and 256³:
- `KE_sat` — saturated kinetic energy
- `MEz/ME_perp` — magnetic anisotropy ratio
- `M_A` — Alfvénic Mach number

---

## Final Deliverables

After all three sims complete:

- `convergence_output/convergence_report.pdf` — 4-panel comparison figures
- `convergence_output/conclusion.txt` — convergence assessment
- `convergence_output/paper_statements.txt` — LaTeX statements for RASTI paper
