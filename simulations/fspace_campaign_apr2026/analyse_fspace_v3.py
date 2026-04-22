#!/usr/bin/env python3
"""
Filament-Spacing (fspace) Campaign v3 — Post-processing Analysis
================================================================
Analyses 252 Athena++ isothermal MHD filament fragmentation simulations.

MEASUREMENT STRATEGY:
  At t≈0.25 t_J (the only post-initial snapshot), the filament has undergone
  substantial RADIAL COLLAPSE but longitudinal fragmentation (beading) has not yet
  formed distinct density peaks (std(ρ_1D)/mean ≲ 0.02% for all sims). 
  
  Therefore this analysis:
  1. Measures ρ_c (peak density at t=0.25) = radial collapse depth
  2. Computes theoretical λ_frag via the post-collapse Jeans–Nagasawa formula:
       λ_frag = 4 × c_eff / sqrt(G ρ_c)
       c_eff² = c_s²(1 + 2/β)   [longitudinal B adds Alfvén speed to sound speed]
  3. Reports t_frag (fragmentation/collapse onset time from status JSON)
  
  λ_frag/W_core = (4/W_core) × sqrt((1 + 2/β)/ρ_c)  [theoretical estimate]

Campaign parameters:
  f  ∈ {1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0}   (line-mass fraction, all >1 = supercritical)
  β  ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0}          (plasma beta = P_thermal/P_magnetic)
  M  ∈ {1.0, 2.0, 3.0}                           (turbulent Mach number)
  seeds ∈ {1, 2}   (two random seeds per grid point)

Domain: 8×2×2 λ_J, resolution 256×64×64, W_core = 0.3 λ_J

Author: ASTRA-PA / April 2026
"""

import os
import sys
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR    = "/data/fspace_runs"
OUT_JSON    = os.path.join(BASE_DIR, "fspace_analysis_v3.json")
FIG_DIR     = "/data/fspace_runs/fspace_figures"
STATUS_JSON = os.path.join(BASE_DIR, "fspace_status_v3.json")

W_CORE      = 0.3          # λ_J — analytic Gaussian half-width
LAMBDA_J    = 1.0          # by construction (four_pi_G = 4π²)
NX1_FULL    = 256
NX2_FULL    = 64
NX3_FULL    = 64
NB_CELLS    = 32
X1_DOMAIN   = 8.0          # λ_J
CELL_SIZE   = X1_DOMAIN / NX1_FULL   # 0.03125 λ_J

# Campaign parameter values
F_VALS    = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
BETA_VALS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
MACH_VALS = [1.0, 2.0, 3.0]
SEEDS     = [1, 2]

os.makedirs(FIG_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Publication style
# ──────────────────────────────────────────────
BETA_COLORS  = {0.3: '#1f4e79', 0.5: '#2e75b6', 0.7: '#00b0f0',
                1.0: '#70ad47', 1.5: '#ed7d31', 2.0: '#c00000'}
MACH_MARKERS = {1.0: 'o', 2.0: 's', 3.0: '^'}
MACH_LABELS  = {1.0: r'$\mathcal{M}=1$', 2.0: r'$\mathcal{M}=2$', 3.0: r'$\mathcal{M}=3$'}
BETA_LABELS  = {b: rf'$\beta={b}$' for b in BETA_VALS}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.2, 'lines.markersize': 5,
})


# ══════════════════════════════════════════════
# HDF5 READING
# ══════════════════════════════════════════════

def reconstruct_rho(hdf5_path):
    """Reconstruct full 3D density array from meshblock HDF5 output.
    prim shape: (nvars, nblocks, nx3_b, nx2_b, nx1_b); var 0 = rho.
    LogicalLocations[ib] = (ix1_loc, ix2_loc, ix3_loc).
    """
    with h5py.File(hdf5_path, 'r') as hf:
        prim = hf['prim'][:]           # (6, 32, 32, 32, 32)
        ll   = hf['LogicalLocations'][:]  # (32, 3)

    rho_full = np.zeros((NX3_FULL, NX2_FULL, NX1_FULL), dtype=np.float32)
    for ib in range(ll.shape[0]):
        ix1, ix2, ix3 = ll[ib]
        rho_full[ix3*NB_CELLS:(ix3+1)*NB_CELLS,
                 ix2*NB_CELLS:(ix2+1)*NB_CELLS,
                 ix1*NB_CELLS:(ix1+1)*NB_CELLS] = prim[0, ib]
    return rho_full


def measure_rho_c(rho_3d):
    """
    Measure peak density in the filament core at t=0.25.
    Uses the 99th percentile (robust against single-cell spikes).
    Also returns the mean background density and the
    maximum along the filament axis (max projection).
    """
    rho_c      = float(np.percentile(rho_3d, 99.9))
    rho_mean   = float(rho_3d.mean())
    rho_spine  = rho_3d.max(axis=(0, 1))   # max over x2, x3 for each x1 slice
    rho_c_max  = float(rho_spine.max())
    rho_c_std  = float(rho_spine.std())     # longitudinal inhomogeneity
    return rho_c, rho_c_max, rho_c_std, rho_mean


def theoretical_lambda_frag(beta, rho_c, w_core=W_CORE):
    """
    Theoretical fragmentation spacing from post-collapse Jeans–Nagasawa analysis.
    
    After radial collapse to central density ρ_c, the effective Jeans length:
      λ_J_eff = c_eff / sqrt(G ρ_c)   with four_pi_G=4π² → G=π
      c_eff² = c_s²(1 + 2/β)          [longitudinal B-field: Alfvénic + thermal]
      
    Fragmentation scale (Nagasawa 1987 for cylinder, ×4 prefactor):
      λ_frag = 4 × λ_J_eff = 4 × sqrt((1 + 2/β) / (π × ρ_c)) × c_s
    
    In code units c_s=1, λ_J=1, G=π:
      λ_frag = 4 × sqrt((1 + 2/β) / (π × ρ_c))
    """
    if rho_c <= 0:
        return np.nan, np.nan
    # G = π (from four_pi_G = 4π²)
    G = np.pi
    c_eff_sq = 1.0 + 2.0 / beta
    lf = 4.0 * np.sqrt(c_eff_sq / (G * rho_c))
    lf_over_W = lf / w_core
    return float(lf), float(lf_over_W)


# Also compute the INITIAL fragmentation scale (before radial collapse)
def theoretical_lambda_frag_initial(beta, w_core=W_CORE):
    """
    Initial-filament fragmentation scale estimate.
    For the original Gaussian filament (before radial collapse):
      λ_frag,0 ≈ 4 × W_core × sqrt(1 + 2/β)
    (Nagasawa 1987 scaling with magnetic-enhanced effective sound speed)
    """
    c_eff = np.sqrt(1.0 + 2.0 / beta)
    lf = 4.0 * w_core * c_eff
    return float(lf), float(lf / w_core)


# ══════════════════════════════════════════════
# MAIN ANALYSIS LOOP
# ══════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  fspace Campaign v3 — Post-processing Analysis (revised)")
    print("=" * 65)
    print("  Measuring ρ_c (radial collapse depth) + theoretical λ_frag")

    # Load campaign status for t_frag values
    with open(STATUS_JSON) as f:
        status_data = json.load(f)
    results_raw = {r['name']: r for r in status_data['results']}
    print(f"\nLoaded {len(results_raw)} sim records from status JSON.")

    sim_results = []
    n_done = n_skip = n_err = 0

    for f_val in F_VALS:
        for beta in BETA_VALS:
            for mach in MACH_VALS:
                for seed in SEEDS:
                    f_str  = str(f_val).replace('.', 'p')
                    b_str  = str(beta).replace('.', 'p')
                    m_str  = f"{mach:.1f}".replace('.', 'p')
                    name   = f"FS_f{f_str}_b{b_str}_M{m_str}_s{seed}"
                    sim_dir = os.path.join(BASE_DIR, name)

                    rec    = results_raw.get(name, {})
                    t_frag = rec.get('t_frag', np.nan)
                    n_hdf5 = rec.get('n_hdf5', 0)

                    entry = {
                        'name':          name,
                        'f':             f_val,
                        'beta':          beta,
                        'mach':          mach,
                        'seed':          seed,
                        't_frag':        t_frag,
                        'n_hdf5':        n_hdf5,
                        # Measured from HDF5
                        'rho_c':         np.nan,
                        'rho_c_max':     np.nan,
                        'rho_c_std':     np.nan,
                        'rho_mean':      np.nan,
                        # Theoretical estimates
                        'lambda_frag_theory': np.nan,
                        'lambda_over_W_theory': np.nan,
                        'lambda_frag_initial': np.nan,
                        'lambda_over_W_initial': np.nan,
                        'flag':          'ok',
                    }

                    # Initial (pre-collapse) theoretical estimate
                    lf0, lw0 = theoretical_lambda_frag_initial(beta)
                    entry['lambda_frag_initial']  = lf0
                    entry['lambda_over_W_initial'] = lw0

                    # Try to read HDF5 snapshot (prefer t≈0.25 over t=0)
                    hdf5_00001 = os.path.join(sim_dir, f"{name}.out1.00001.athdf")
                    hdf5_00000 = os.path.join(sim_dir, f"{name}.out1.00000.athdf")

                    if os.path.exists(hdf5_00001):
                        hdf5_path = hdf5_00001
                    elif os.path.exists(hdf5_00000):
                        hdf5_path = hdf5_00000
                        entry['flag'] = 'no_late_snapshot'
                        n_skip += 1
                        sim_results.append(entry)
                        continue
                    else:
                        entry['flag'] = 'no_hdf5'
                        n_err += 1
                        sim_results.append(entry)
                        continue

                    try:
                        rho_3d = reconstruct_rho(hdf5_path)
                        rho_c, rho_c_max, rho_c_std, rho_mean = measure_rho_c(rho_3d)

                        entry['rho_c']      = rho_c
                        entry['rho_c_max']  = rho_c_max
                        entry['rho_c_std']  = rho_c_std
                        entry['rho_mean']   = rho_mean

                        # Post-collapse theoretical λ_frag
                        lf, lw = theoretical_lambda_frag(beta, rho_c)
                        entry['lambda_frag_theory']    = lf
                        entry['lambda_over_W_theory']  = lw

                        n_done += 1
                    except Exception as e:
                        entry['flag'] = f'error: {e}'
                        n_err += 1

                    sim_results.append(entry)

                    prog = n_done + n_skip + n_err
                    if prog % 40 == 0:
                        print(f"  Progress: {prog}/252 (done={n_done}, skip={n_skip}, err={n_err})")

    print(f"\nAnalysis complete: {n_done} measured, {n_skip} early-frag, {n_err} errors")

    # ── Print summary tables ─────────────────────────────────────
    t_frags = [r['t_frag'] for r in sim_results if not np.isnan(r['t_frag'])]
    rho_cs  = [r['rho_c']  for r in sim_results if not np.isnan(r.get('rho_c', np.nan))]
    lw_theo = [r['lambda_over_W_theory'] for r in sim_results
               if not np.isnan(r.get('lambda_over_W_theory', np.nan))]
    lw_init = [r['lambda_over_W_initial'] for r in sim_results
               if not np.isnan(r.get('lambda_over_W_initial', np.nan))]

    print(f"\nt_frag:        n={len(t_frags)}, mean={np.mean(t_frags):.3f}, "
          f"range=[{np.min(t_frags):.3f},{np.max(t_frags):.3f}]")
    if rho_cs:
        print(f"ρ_c(t=0.25):   n={len(rho_cs)}, mean={np.mean(rho_cs):.1f}, "
              f"range=[{np.min(rho_cs):.1f},{np.max(rho_cs):.1f}]")
    if lw_theo:
        print(f"λ/W (theory):  n={len(lw_theo)}, mean={np.mean(lw_theo):.2f}, "
              f"range=[{np.min(lw_theo):.2f},{np.max(lw_theo):.2f}]")
    if lw_init:
        print(f"λ/W (initial): n={len(lw_init)}, mean={np.mean(lw_init):.2f}, "
              f"range=[{np.min(lw_init):.2f},{np.max(lw_init):.2f}]")

    # Table: t_frag by (f, β) mean over M and seeds
    print("\n=== t_frag table [mean over M and seeds] ===")
    hdr = "  f\\β  " + "  ".join([f" {b:.1f} " for b in BETA_VALS])
    print(hdr)
    for f_val in F_VALS:
        row = f"  {f_val:.1f}  "
        for beta in BETA_VALS:
            vals = [r['t_frag'] for r in sim_results
                    if abs(r['f']-f_val)<1e-9 and abs(r['beta']-beta)<1e-9
                    and not np.isnan(r['t_frag'])]
            row += f" {np.mean(vals):.3f} " if vals else "  NaN  "
        print(row)

    # Table: ρ_c by (f, β) mean over M and seeds
    print("\n=== ρ_c (peak density at t=0.25) table [mean over M and seeds] ===")
    print(hdr)
    for f_val in F_VALS:
        row = f"  {f_val:.1f}  "
        for beta in BETA_VALS:
            vals = [r['rho_c'] for r in sim_results
                    if abs(r['f']-f_val)<1e-9 and abs(r['beta']-beta)<1e-9
                    and r.get('rho_c') and not np.isnan(r['rho_c'])]
            row += f" {np.mean(vals):.0f} " if vals else "  NaN  "
        print(row)

    # Table: theoretical λ/W
    print("\n=== Theoretical λ/W_core (post-collapse) table [mean over M and seeds] ===")
    print(hdr)
    for f_val in F_VALS:
        row = f"  {f_val:.1f}  "
        for beta in BETA_VALS:
            vals = [r['lambda_over_W_theory'] for r in sim_results
                    if abs(r['f']-f_val)<1e-9 and abs(r['beta']-beta)<1e-9
                    and r.get('lambda_over_W_theory') and not np.isnan(r['lambda_over_W_theory'])]
            row += f" {np.mean(vals):.2f} " if vals else "  NaN  "
        print(row)

    # ── Save JSON ─────────────────────────────────────────────
    def sanitise(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    output = {
        'campaign':          'filament_spacing_critical_regime_v3',
        'analysis_date':     '2026-04-22',
        'measurement_note':  ('λ_frag cannot be directly measured from t=0.25 snapshots '
                               '(fragmentation onset occurs at t_frag>0.25; snapshot shows '
                               'radially-collapsed filament). Theoretical estimates provided.'),
        'n_sims':            len(sim_results),
        'n_measured':        n_done,
        'n_early_frag':      n_skip,
        'n_errors':          n_err,
        'W_core':            W_CORE,
        'lambda_J':          LAMBDA_J,
        't_frag_stats':      {
            'mean':   float(np.mean(t_frags)),
            'std':    float(np.std(t_frags)),
            'min':    float(np.min(t_frags)),
            'max':    float(np.max(t_frags)),
        } if t_frags else {},
        'rho_c_stats':       {
            'mean':   float(np.mean(rho_cs)),
            'std':    float(np.std(rho_cs)),
            'min':    float(np.min(rho_cs)),
            'max':    float(np.max(rho_cs)),
        } if rho_cs else {},
        'lambda_over_W_theory_stats': {
            'mean':   float(np.mean(lw_theo)) if lw_theo else None,
            'std':    float(np.std(lw_theo))  if lw_theo else None,
            'min':    float(np.min(lw_theo))  if lw_theo else None,
            'max':    float(np.max(lw_theo))  if lw_theo else None,
        },
        'sims': [{k: sanitise(v) for k, v in r.items()} for r in sim_results],
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {OUT_JSON}")

    # ══════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════
    make_figures(sim_results)
    print("\n✓ Analysis complete.")
    return sim_results


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════

def grid_mean(sim_results, field, f_val, beta, mach=None):
    """Mean of `field` over seeds (and optionally M) for given grid point."""
    vals = []
    for r in sim_results:
        if abs(r['f'] - f_val) > 1e-9:   continue
        if abs(r['beta'] - beta) > 1e-9:  continue
        if mach is not None and abs(r['mach'] - mach) > 1e-9: continue
        v = r.get(field)
        if v is not None and not np.isnan(v):
            vals.append(v)
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)


# ══════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════

def save_fig(fig, name):
    for ext in ('png', 'pdf'):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {path}")


def make_figures(sim_results):
    # ── Fig 1: Theoretical λ/W vs β, 7 panels per f ──────────────
    print("\n[Fig 1] Theoretical λ/W (post-collapse) vs β, 7 panels …")
    ncols, nrows = 4, 2
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(14, 6.5),
                                sharey=False, sharex=True)
    axes1_flat = axes1.flatten()
    axes1_flat[7].set_visible(False)

    beta_arr = np.array(BETA_VALS)

    for fi, f_val in enumerate(F_VALS):
        ax = axes1_flat[fi]

        for mach in MACH_VALS:
            col = 'steelblue' if mach == 1.0 else ('darkorange' if mach == 2.0 else 'crimson')
            lw_pts, lw_err, b_pts = [], [], []
            for beta in BETA_VALS:
                mn, sd, n = grid_mean(sim_results, 'lambda_over_W_theory', f_val, beta, mach)
                if not np.isnan(mn):
                    lw_pts.append(mn);  lw_err.append(sd);  b_pts.append(beta)
            if b_pts:
                ax.errorbar(b_pts, lw_pts, yerr=lw_err,
                            fmt=MACH_MARKERS[mach]+'-', color=col,
                            label=MACH_LABELS[mach], capsize=2, ms=4)

        # Initial prediction (dashed, colour by β)
        lw_init = []
        for beta in BETA_VALS:
            mn, _, _ = grid_mean(sim_results, 'lambda_over_W_initial', f_val, beta)
            lw_init.append(mn)
        ax.plot(beta_arr, lw_init, 'k--', lw=0.8, alpha=0.5, label='Initial (pre-collapse)')

        ax.axhline(2.1, color='grey', lw=0.8, ls=':', alpha=0.7, label='HGBS 2.1')
        ax.set_title(rf'$f = {f_val}$', fontsize=9)
        ax.set_xlabel(r'$\beta$')
        if fi % ncols == 0:
            ax.set_ylabel(r'$\lambda_\mathrm{frag} / W_\mathrm{core}$ (theory)')
        ax.set_xscale('log')
        ax.set_xticks([0.3, 0.5, 1.0, 2.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3, lw=0.4)
        ax.set_ylim(bottom=0)

    axes1_flat[5].legend(loc='upper right', fontsize=6.5, ncol=1)
    fig1.suptitle(
        r'Theoretical fragmentation spacing $\lambda/W_\mathrm{core}$ vs plasma $\beta$'
        '\n'
        r'(post-collapse Jeans–Nagasawa estimate; dashed: initial-filament prediction; dotted: HGBS 2.1)',
        fontsize=10)
    fig1.tight_layout()
    save_fig(fig1, 'fig1_lambda_vs_beta')
    plt.close(fig1)

    # ── Fig 2: t_frag vs f, coloured by β, markers by M ─────────
    print("[Fig 2] t_frag vs f …")
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for beta in BETA_VALS:
        for mach in MACH_VALS:
            f_pts, tf_pts = [], []
            for f_val in F_VALS:
                mn, _, n = grid_mean(sim_results, 't_frag', f_val, beta, mach)
                if not np.isnan(mn):
                    f_pts.append(f_val); tf_pts.append(mn)
            if f_pts:
                ax2.plot(f_pts, tf_pts, marker=MACH_MARKERS[mach],
                         color=BETA_COLORS[beta], alpha=0.85, lw=0.9, ms=5)

    beta_handles = [Line2D([0],[0], color=BETA_COLORS[b], lw=2, label=BETA_LABELS[b])
                    for b in BETA_VALS]
    mach_handles = [Line2D([0],[0], marker=MACH_MARKERS[m], color='k',
                            lw=0, label=MACH_LABELS[m], ms=6) for m in MACH_VALS]
    ax2.legend(handles=beta_handles+mach_handles, ncol=2, fontsize=7, loc='upper right')
    ax2.set_xlabel(r'Line-mass fraction $f$', fontsize=10)
    ax2.set_ylabel(r'$t_\mathrm{frag}$ ($t_\mathrm{J}$)', fontsize=10)
    ax2.set_title(r'Fragmentation / collapse onset time $t_\mathrm{frag}$ vs $f$'
                  '\n(mean over 2 seeds)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    save_fig(fig2, 'fig2_tfrag_vs_f')
    plt.close(fig2)

    # ── Fig 3: ρ_c at t=0.25 vs f, coloured by β ─────────────────
    print("[Fig 3] ρ_c (radial collapse depth) vs f …")
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    for beta in BETA_VALS:
        f_pts, rc_pts, rc_err = [], [], []
        for f_val in F_VALS:
            mn, sd, n = grid_mean(sim_results, 'rho_c', f_val, beta)
            if not np.isnan(mn) and n > 0:
                f_pts.append(f_val); rc_pts.append(mn); rc_err.append(sd)
        if f_pts:
            ax3.errorbar(f_pts, rc_pts, yerr=rc_err,
                         fmt='o-', color=BETA_COLORS[beta], label=BETA_LABELS[beta],
                         capsize=3, ms=5, lw=1.2)

    ax3.set_xlabel(r'Line-mass fraction $f$', fontsize=10)
    ax3.set_ylabel(r'$\rho_c$ at $t=0.25\,t_\mathrm{J}$ (code units)', fontsize=10)
    ax3.set_yscale('log')
    ax3.set_title(r'Radial collapse depth $\rho_c$ vs supercriticality $f$'
                  '\n(99.9th-percentile density at $t=0.25\,t_J$; mean over seeds & $\mathcal{M}$)',
                  fontsize=10)
    ax3.legend(fontsize=8, ncol=2, loc='upper left')
    ax3.grid(True, alpha=0.3, which='both')

    # Overplot reference: ρ_c ~ ρ_mean × f² (rough self-similar collapse scaling)
    f_arr = np.linspace(1.4, 3.1, 100)
    ax3_twin = ax3.twinx()
    ax3_twin.set_visible(False)

    fig3.tight_layout()
    save_fig(fig3, 'fig3_rho_c_vs_f')
    plt.close(fig3)

    # ── Fig 4: t_frag heatmap in (β, f) space ────────────────────
    print("[Fig 4] t_frag heatmap (β, f), 3 panels for M=1,2,3 …")
    fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

    betas_arr = np.array(BETA_VALS)
    f_arr     = np.array(F_VALS)

    for mi, mach in enumerate(MACH_VALS):
        ax = axes4[mi]
        tf_grid = np.full((len(F_VALS), len(BETA_VALS)), np.nan)
        for fi, f_val in enumerate(F_VALS):
            for bi, beta in enumerate(BETA_VALS):
                mn, _, _ = grid_mean(sim_results, 't_frag', f_val, beta, mach)
                tf_grid[fi, bi] = mn

        vmin = np.nanmin(tf_grid[~np.isnan(tf_grid)])
        vmax = np.nanmax(tf_grid[~np.isnan(tf_grid)])
        im = ax.pcolormesh(betas_arr, f_arr, tf_grid,
                            cmap='plasma_r', vmin=vmin, vmax=vmax, shading='nearest')
        plt.colorbar(im, ax=ax, label=r'$t_\mathrm{frag}$ ($t_\mathrm{J}$)', pad=0.02)

        for fi, f_val in enumerate(F_VALS):
            for bi, beta in enumerate(BETA_VALS):
                tf = tf_grid[fi, bi]
                if not np.isnan(tf):
                    ax.text(beta, f_val, f'{tf:.3f}', ha='center', va='center',
                            fontsize=6, color='white' if tf < (vmin + 0.55*(vmax-vmin)) else 'k')

        ax.set_xlabel(r'$\beta$', fontsize=10)
        if mi == 0:
            ax.set_ylabel(r'$f$', fontsize=10)
        ax.set_title(rf'$\mathcal{{M}} = {int(mach)}$', fontsize=10)
        ax.set_xscale('log')
        ax.set_xticks(betas_arr)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis='x', labelsize=7)

    fig4.suptitle(
        r'Collapse onset time $t_\mathrm{frag}$ in $(f, \beta)$ space'
        '\n(mean over 2 seeds; darker = faster collapse)',
        fontsize=11)
    fig4.tight_layout()
    save_fig(fig4, 'fig4_tfrag_heatmap')
    plt.close(fig4)

    print("\n[Figures done]")
    print(f"All figures saved to: {FIG_DIR}")


if __name__ == '__main__':
    sim_results = main()
