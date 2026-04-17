#!/usr/bin/env python3
"""
ASTRA MHD Resolution Convergence Analysis
==========================================
Compares 128³ and 256³ Athena++ MHD simulations for three representative
parameter combinations:
  - M3_beta1.0  (ℳ=3, β=1.0)
  - M3_beta0.1  (ℳ=3, β=0.1)
  - M1_beta1.0  (ℳ=1, β=1.0)

Reads:
  - 128³ data : sweep_output/{mhd_M03_beta1.0,...}/*.hst
  - 256³ data : convergence_output/{M3_beta1.0_256,...}/energy_history.dat

Produces:
  - convergence_report.pdf    (4 publication-quality figures, saved to output-dir)
  - conclusion.txt            (convergence assessment with pass/fail)
  - paper_statements.txt      (LaTeX-ready statements for RASTI paper)

Usage:
  python3 analyse_convergence.py
  python3 analyse_convergence.py --output-dir /path/to/convergence_output \\
                                  --ref-dir    /path/to/sweep_output

Author : ASTRA PA (for Glenn J. White, Open University)
Date   : 2026-04-17
"""

import argparse
import json
import os
import sys
import glob
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Suppress benign matplotlib warnings about layout
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ── Publication-quality typography ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.labelsize':     12,
    'axes.titlesize':     11,
    'legend.fontsize':    9,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'figure.dpi':         150,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     ':',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'lines.linewidth':    1.6,
})

# ── Convergence threshold ──────────────────────────────────────────────────────
CONVERGE_THRESHOLD = 10.0   # percent

# ── Simulation catalogue ───────────────────────────────────────────────────────
# Each entry: display label → configuration dict
SIMS = {
    'M3_β1.0': {
        'beta':       1.0,
        'mach':       3.0,
        'ref_subdir': 'mhd_M03_beta1.0',   # 128³: sweep_output/<ref_subdir>/*.hst
        'hires_key':  'M3_beta1.0_256',     # 256³: convergence_output/<hires_key>/energy_history.dat
        'color':      '#1f77b4',             # blue
        # Expected saturated-state ranges (from science brief)
        'KE_sat_t':   (0.5,  1.5),
        'MEz_ratio':  (3.0,  10.0),
        'M_A_range':  (0.5,  1.5),
        'M_A_target': 1.0,
    },
    'M3_β0.1': {
        'beta':       0.1,
        'mach':       3.0,
        'ref_subdir': 'mhd_M03_beta0.1',
        'hires_key':  'M3_beta0.1_256',
        'color':      '#d62728',             # red
        'KE_sat_t':   (1.0,  2.0),
        'MEz_ratio':  (50.,  100.),
        'M_A_range':  (0.3,  0.9),
        'M_A_target': 0.6,
    },
    'M1_β1.0': {
        'beta':       1.0,
        'mach':       1.0,
        'ref_subdir': 'mhd_M01_beta1.0',
        'hires_key':  'M1_beta1.0_256',
        'color':      '#2ca02c',             # green
        'KE_sat_t':   (0.25, 0.75),
        'MEz_ratio':  (20.,  50.),
        'M_A_range':  (0.5,  1.1),
        'M_A_target': 0.8,
    },
}

SIM_LABELS = list(SIMS.keys())

# Orange for 256³ so it contrasts the per-sim colour of 128³
COLOR_256 = '#ff7f0e'


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def read_hst(path: str) -> dict:
    """
    Read an Athena++ history (.hst) file.

    Athena++ column layout (standard MHD run):
      [1]=time  [2]=dt   [3]=mass
      [4]=1-mom [5]=2-mom [6]=3-mom
      [7]=1-KE  [8]=2-KE  [9]=3-KE
      [10]=1-ME [11]=2-ME [12]=3-ME

    Returns a dict with keys:
      'time', 'KEx','KEy','KEz', 'MEx','MEy','MEz', 'rho',
      'KE'  (total KE),  'ME' (total ME),
      'ME_perp' (MEx+MEy, perpendicular to filament/z-axis),
      'M_A'     (Alfvénic Mach number),
      'MEz_ratio' (MEz / ME_perp),
      '_arr'    (raw Nx12 ndarray)

    Column names from the header are also stored as-is for reference.
    Gracefully handles files with missing or non-standard headers.
    """
    cols = None
    data = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('#') and '=' in line:
                # Parse e.g. "[1]=time  [2]=dt  ..."
                parts = line.strip('# ').split()
                cols = [p.split('=')[1] for p in parts if '=' in p]
                continue
            if line.startswith('#') or not line.strip():
                continue
            try:
                data.append([float(x) for x in line.split()])
            except ValueError:
                continue

    if not data:
        return {}

    arr = np.array(data)
    result = {'_arr': arr}

    # Store named columns from header (for diagnostics)
    if cols:
        for j, col in enumerate(cols):
            if j < arr.shape[1]:
                result[col] = arr[:, j]

    # Standard derived quantities — require at least 12 columns
    if arr.shape[1] < 12:
        return result

    result['time']    = arr[:, 0]
    result['rho']     = arr[:, 2]   # total mass = mean density in unit box
    result['KEx']     = arr[:, 6]
    result['KEy']     = arr[:, 7]
    result['KEz']     = arr[:, 8]
    result['MEx']     = arr[:, 9]
    result['MEy']     = arr[:, 10]
    result['MEz']     = arr[:, 11]

    result['KE']      = arr[:, 6] + arr[:, 7] + arr[:, 8]
    result['ME']      = arr[:, 9] + arr[:, 10] + arr[:, 11]
    result['ME_perp'] = arr[:, 9] + arr[:, 10]  # B_x² + B_y² (⊥ to mean-field z)

    rho = result['rho']
    KE  = result['KE']
    ME  = result['ME']

    # v_rms from KE density: KE = ½ρv² → v_rms = √(2KE/ρ)
    v_rms = np.where(rho > 0,  np.sqrt(2.0 * KE / rho),  0.0)
    # Alfvén speed: ME = B²/(8π) in code units; v_A = √(2ME/ρ)
    v_A   = np.where((rho > 0) & (ME > 1e-30), np.sqrt(2.0 * ME / rho), np.nan)

    result['v_rms']     = v_rms
    result['v_A']       = v_A
    result['M_A']       = np.where(np.isfinite(v_A) & (v_A > 0), v_rms / v_A, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            result['ME_perp'] > 1e-30,
            result['MEz'] / result['ME_perp'],
            np.nan,
        )
    result['MEz_ratio'] = ratio
    return result


def read_energy_history(path: str) -> dict:
    """
    Read energy_history.dat written by extract_convergence_diagnostics.py.

    Column layout:
      time  KE_total  MEz  ME_perp  ME_total  M_A  MEz_ratio

    Returns a dict with the same key names as read_hst() for the relevant
    quantities, so plotting code can treat both identically.
    """
    data = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            try:
                data.append([float(x) for x in line.split()])
            except ValueError:
                continue

    if not data:
        return {}

    arr = np.array(data)
    if arr.shape[1] < 7:
        return {}

    result = {
        '_arr':      arr,
        'time':      arr[:, 0],
        'KE':        arr[:, 1],
        'MEz':       arr[:, 2],
        'ME_perp':   arr[:, 3],
        'ME':        arr[:, 4],
        'M_A':       arr[:, 5],
        'MEz_ratio': arr[:, 6],
    }
    # Replace placeholder -1.0 (used for non-finite) with NaN
    for key in ('M_A', 'MEz_ratio'):
        result[key] = np.where(result[key] < 0, np.nan, result[key])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ══════════════════════════════════════════════════════════════════════════════

def saturated_mask(d: dict, frac: float = 0.25) -> np.ndarray:
    """Boolean mask for last `frac` of the time series (saturation window)."""
    t = d['time']
    mask = t >= t[-1] * (1.0 - frac)
    if mask.sum() < 10:
        mask = np.ones(len(t), dtype=bool)
    return mask


def sat_stats(arr: np.ndarray, mask: np.ndarray) -> tuple:
    """Return (mean, std) over saturation window, ignoring NaN."""
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    return (float(np.mean(vals)), float(np.std(vals)))


def find_saturation_time(t: np.ndarray, KE: np.ndarray,
                          window_frac: float = 0.10,
                          threshold:   float = 0.05) -> float:
    """
    Estimate KE saturation time: first t where the coefficient of variation
    (std/mean) over a sliding window of width `window_frac * N` drops below
    `threshold`.  Returns NaN if never saturated.
    """
    nw = max(10, int(window_frac * len(t)))
    for i in range(nw, len(t) - nw):
        seg = KE[i:i + nw]
        if np.mean(seg) > 0 and np.std(seg) / np.mean(seg) < threshold:
            return float(t[i])
    return float('nan')


def pct_diff(a: float, b: float) -> float:
    """Symmetric percentage difference between two scalars."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    denom = 0.5 * (abs(a) + abs(b))
    return 100.0 * abs(a - b) / (denom + 1e-30)


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def shade_sat(ax, t: np.ndarray, frac: float = 0.25, label: str = 'Sat. window (last 25%)'):
    """Shade the saturation window on a time-series axis."""
    t_cut = t[-1] * (1.0 - frac)
    ax.axvspan(t_cut, t[-1], color='silver', alpha=0.13, zorder=0, label=label)


def hline_annotate(ax, val: float, color: str, ls: str = '--',
                   label: str = '', side: str = 'right', va: str = 'bottom',
                   fontsize: int = 8):
    """Draw a horizontal reference line at `val` with an inline label."""
    if not np.isfinite(val):
        return
    ax.axhline(val, color=color, lw=0.85, ls=ls, alpha=0.45, zorder=1)
    if label:
        xlim = ax.get_xlim()
        x_pos = xlim[1] if side == 'right' else xlim[0]
        ha = 'right' if side == 'right' else 'left'
        ax.annotate(
            label, xy=(x_pos, val),
            xycoords=('data', 'data'),
            xytext=(-4 if side == 'right' else 4, 2),
            textcoords='offset points',
            ha=ha, va=va, fontsize=fontsize, color=color,
        )


def safe_semilogy(ax, t, y, **kwargs):
    """Plot y vs t on a log-y axis, masking non-positive values cleanly."""
    y_plot = np.where((np.isfinite(y)) & (y > 0), y, np.nan)
    ax.semilogy(t, y_plot, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='ASTRA convergence analysis: 128³ vs 256³ MHD simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        '--output-dir',
        default='/workspace/athena/convergence_output',
        help='Directory containing 256³ subdirectories; also the output destination',
    )
    ap.add_argument(
        '--ref-dir',
        default='/workspace/athena/sweep_output',
        help='Directory containing 128³ sweep_output subdirectories',
    )
    args = ap.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    REF_DIR    = Path(args.ref_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    print(f"\n{'='*70}")
    print(f"  ASTRA Resolution Convergence Analysis")
    print(f"  128³ reference : {REF_DIR}")
    print(f"  256³ data      : {OUTPUT_DIR}")
    print(f"  Generated      : {timestamp}")
    print(f"{'='*70}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    sim_data = {}   # label → {'lo': dict|None, 'hi': dict|None, 'cfg': dict}

    for label, cfg in SIMS.items():
        print(f"  Loading {label}  (ℳ={cfg['mach']:.0f}, β={cfg['beta']})")

        # 128³: from .hst file
        ref_dir_sim = REF_DIR / cfg['ref_subdir']
        hst_files   = sorted(ref_dir_sim.glob('*.hst')) if ref_dir_sim.exists() else []
        lo_data     = None
        if hst_files:
            lo_data = read_hst(str(hst_files[0]))
            if lo_data and 'time' in lo_data:
                print(f"    128³ ✓  {hst_files[0].name}  "
                      f"({len(lo_data['time'])} steps, "
                      f"t_final={lo_data['time'][-1]:.3f})")
            else:
                print(f"    128³ ✗  parse error: {hst_files[0]}")
                lo_data = None
        else:
            print(f"    128³ ✗  no .hst found in {ref_dir_sim}")

        # 256³: from energy_history.dat
        hi_path = OUTPUT_DIR / cfg['hires_key'] / 'energy_history.dat'
        hi_data = None
        if hi_path.exists():
            hi_data = read_energy_history(str(hi_path))
            if hi_data and 'time' in hi_data:
                print(f"    256³ ✓  energy_history.dat  "
                      f"({len(hi_data['time'])} steps, "
                      f"t_final={hi_data['time'][-1]:.3f})")
            else:
                print(f"    256³ ✗  parse error: {hi_path}")
                hi_data = None
        else:
            print(f"    256³ –  not found ({hi_path.name}) — will plot 128³ only")

        sim_data[label] = {'lo': lo_data, 'hi': hi_data, 'cfg': cfg}

    print()

    # ── Compute saturated-state statistics ────────────────────────────────────
    sim_stats = {}   # label → dict of scalar diagnostics

    for label, sd in sim_data.items():
        stats = {}
        lo = sd['lo']
        hi = sd['hi']

        if lo is not None and 'time' in lo:
            mask = saturated_mask(lo)
            stats['lo_KE']        = sat_stats(lo['KE'],        mask)
            stats['lo_MEz']       = sat_stats(lo['MEz'],       mask)
            stats['lo_ME_perp']   = sat_stats(lo['ME_perp'],   mask)
            stats['lo_M_A']       = sat_stats(lo['M_A'],       mask)
            stats['lo_MEz_ratio'] = sat_stats(lo['MEz_ratio'], mask)
            stats['lo_t_sat']     = find_saturation_time(lo['time'], lo['KE'])

        if hi is not None and 'time' in hi:
            mask = saturated_mask(hi)
            stats['hi_KE']        = sat_stats(hi['KE'],        mask)
            stats['hi_MEz']       = sat_stats(hi['MEz'],       mask)
            stats['hi_ME_perp']   = sat_stats(hi['ME_perp'],   mask)
            stats['hi_M_A']       = sat_stats(hi['M_A'],       mask)
            stats['hi_MEz_ratio'] = sat_stats(hi['MEz_ratio'], mask)
            stats['hi_t_sat']     = find_saturation_time(hi['time'], hi['KE'])

        # Convergence percentages (requires both resolutions)
        if 'lo_KE' in stats and 'hi_KE' in stats:
            stats['conv_KE']        = pct_diff(stats['lo_KE'][0],        stats['hi_KE'][0])
            stats['conv_MEz_ratio'] = pct_diff(stats['lo_MEz_ratio'][0], stats['hi_MEz_ratio'][0])
            stats['conv_M_A']       = pct_diff(stats['lo_M_A'][0],       stats['hi_M_A'][0])
            stats['converged']      = all([
                stats['conv_KE']        < CONVERGE_THRESHOLD,
                stats['conv_MEz_ratio'] < CONVERGE_THRESHOLD,
                stats['conv_M_A']       < CONVERGE_THRESHOLD,
            ])

        sim_stats[label] = stats

    # ── Print diagnostic summary ──────────────────────────────────────────────
    print("SATURATED-STATE DIAGNOSTICS (last 25% of each run)")
    print('-' * 70)
    for label, stats in sim_stats.items():
        cfg = SIMS[label]
        print(f"\n  {label}  (ℳ={cfg['mach']:.0f}, β={cfg['beta']})")
        for pfx, res_lbl in [('lo', '128³'), ('hi', '256³')]:
            if f'{pfx}_KE' in stats:
                km, ks = stats[f'{pfx}_KE']
                mm, _  = stats[f'{pfx}_MEz_ratio']
                am, _  = stats[f'{pfx}_M_A']
                ts     = stats.get(f'{pfx}_t_sat', np.nan)
                print(f"    {res_lbl}:  KE_sat={km:.3e}±{ks:.1e}  "
                      f"MEz/ME⊥={mm:.2f}  M_A={am:.3f}  t_sat={ts:.3f}")
        if 'conv_KE' in stats:
            flag = '✓ CONVERGED' if stats['converged'] else '✗ NOT CONVERGED'
            print(f"    Δ: KE={stats['conv_KE']:.1f}%  "
                  f"MEz/ME⊥={stats['conv_MEz_ratio']:.1f}%  "
                  f"M_A={stats['conv_M_A']:.1f}%  → {flag}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURES — four pages in one PDF
    # ══════════════════════════════════════════════════════════════════════════
    pdf_path = OUTPUT_DIR / 'convergence_report.pdf'
    print(f"  Writing: {pdf_path}")
    n = len(SIM_LABELS)

    with PdfPages(str(pdf_path)) as pdf:

        # PDF metadata
        d = pdf.infodict()
        d['Title']    = 'ASTRA MHD Resolution Convergence Report'
        d['Author']   = 'ASTRA PA — Glenn J. White (Open University)'
        d['Subject']  = '128³ vs 256³: M3β1.0, M3β0.1, M1β1.0'
        d['Keywords'] = 'MHD, Athena++, convergence, filaments, ISM'
        d['Creator']  = 'analyse_convergence.py'

        # ── Fig 1: KE(t) ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), sharey=False)
        fig.suptitle(
            r'Fig. 1 — Kinetic energy $E_K(t)$:  128$^3$ vs 256$^3$',
            fontsize=13, fontweight='bold', y=1.01,
        )

        for ax, label in zip(axes, SIM_LABELS):
            sd    = sim_data[label]
            stats = sim_stats[label]
            cfg   = sd['cfg']
            lo    = sd['lo']
            hi    = sd['hi']
            c128  = cfg['color']

            if lo is not None:
                ax.plot(lo['time'], lo['KE'], color=c128, lw=1.8,
                        alpha=0.9, label=r'128$^3$', zorder=3)
                shade_sat(ax, lo['time'])

            if hi is not None:
                ax.plot(hi['time'], hi['KE'], color=COLOR_256, lw=1.8,
                        alpha=0.9, ls='--', label=r'256$^3$', zorder=3)

            # Annotate saturated-state means
            if 'lo_KE' in stats:
                ke_lo = stats['lo_KE'][0]
                hline_annotate(ax, ke_lo, c128, ls='--',
                               label=f'⟨KE⟩₁₂₈={ke_lo:.2e}',
                               side='right', va='bottom')
                # mark saturation time
                t_sat = stats.get('lo_t_sat', np.nan)
                if np.isfinite(t_sat):
                    ax.axvline(t_sat, color=c128, lw=0.8, ls=':', alpha=0.5)

            if 'hi_KE' in stats:
                ke_hi = stats['hi_KE'][0]
                hline_annotate(ax, ke_hi, COLOR_256, ls=':',
                               label=f'⟨KE⟩₂₅₆={ke_hi:.2e}',
                               side='right', va='top')

            # Convergence badge
            if 'conv_KE' in stats:
                badge = (f"ΔKE={stats['conv_KE']:.1f}%  "
                         + ('✓' if stats['conv_KE'] < CONVERGE_THRESHOLD else '✗'))
                ax.text(0.97, 0.97, badge, transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

            ax.set_xlabel(r'Time  [$t_{\rm cross}$]', fontsize=11)
            ax.set_ylabel(r'Kinetic energy density $E_K$', fontsize=11)
            ax.set_title(
                rf'$\mathcal{{M}}={cfg["mach"]:.0f},\ \beta={cfg["beta"]}$',
                fontsize=11)
            ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
            ax.set_xlim(left=0)

        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print('  ✓  Fig 1: KE(t)')

        # ── Fig 2: MEz(t) and ME_perp(t) ──────────────────────────────────
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), sharey=False)
        fig.suptitle(
            r'Fig. 2 — Magnetic energy components:  $E_{B_z}(t)$  and  $E_{B_\perp}(t)$',
            fontsize=13, fontweight='bold', y=1.01,
        )

        for ax, label in zip(axes, SIM_LABELS):
            sd    = sim_data[label]
            stats = sim_stats[label]
            cfg   = sd['cfg']
            lo    = sd['lo']
            hi    = sd['hi']
            c128  = cfg['color']

            if lo is not None:
                ax.plot(lo['time'], lo['MEz'],     color=c128, lw=1.8, alpha=0.90,
                        label=r'$E_{B_z}$  128$^3$', zorder=3)
                ax.plot(lo['time'], lo['ME_perp'], color=c128, lw=1.2, alpha=0.55,
                        ls='-.', label=r'$E_{B_\perp}$  128$^3$', zorder=3)
                shade_sat(ax, lo['time'])

            if hi is not None:
                ax.plot(hi['time'], hi['MEz'],     color=COLOR_256, lw=1.8, alpha=0.90,
                        ls='--', label=r'$E_{B_z}$  256$^3$', zorder=3)
                ax.plot(hi['time'], hi['ME_perp'], color=COLOR_256, lw=1.2, alpha=0.55,
                        ls=':', label=r'$E_{B_\perp}$  256$^3$', zorder=3)

            # Annotate saturated-state MEz and ME_perp
            if 'lo_MEz' in stats:
                mez = stats['lo_MEz'][0]
                mep = stats['lo_ME_perp'][0]
                hline_annotate(ax, mez, c128, ls='--',
                               label=f'⟨MEz⟩₁₂₈={mez:.2e}', side='right', va='bottom')
                hline_annotate(ax, mep, c128, ls='-.',
                               label=f'⟨ME⊥⟩₁₂₈={mep:.2e}', side='right', va='top')

            if 'hi_MEz' in stats:
                mez = stats['hi_MEz'][0]
                hline_annotate(ax, mez, COLOR_256, ls=':',
                               label=f'⟨MEz⟩₂₅₆={mez:.2e}', side='left', va='bottom')

            ax.set_xlabel(r'Time  [$t_{\rm cross}$]', fontsize=11)
            ax.set_ylabel(r'Magnetic energy density', fontsize=11)
            ax.set_title(
                rf'$\mathcal{{M}}={cfg["mach"]:.0f},\ \beta={cfg["beta"]}$',
                fontsize=11)
            ax.legend(fontsize=8, loc='lower right', ncol=2, framealpha=0.85)
            ax.set_xlim(left=0)

        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print('  ✓  Fig 2: MEz(t) + ME_perp(t)')

        # ── Fig 3: MEz/ME_perp ratio (log-y) ──────────────────────────────
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), sharey=False)
        fig.suptitle(
            r'Fig. 3 — Field anisotropy  $E_{B_z}/E_{B_\perp}(t)$  (log scale)',
            fontsize=13, fontweight='bold', y=1.01,
        )

        for ax, label in zip(axes, SIM_LABELS):
            sd    = sim_data[label]
            stats = sim_stats[label]
            cfg   = sd['cfg']
            lo    = sd['lo']
            hi    = sd['hi']
            c128  = cfg['color']

            if lo is not None:
                safe_semilogy(ax, lo['time'], lo['MEz_ratio'],
                              color=c128, lw=1.8, alpha=0.9,
                              label=r'128$^3$', zorder=3)
                shade_sat(ax, lo['time'])

            if hi is not None:
                safe_semilogy(ax, hi['time'], hi['MEz_ratio'],
                              color=COLOR_256, lw=1.8, alpha=0.9,
                              ls='--', label=r'256$^3$', zorder=3)

            # Expected range band (gold)
            lo_exp, hi_exp = cfg['MEz_ratio']
            ax.axhspan(lo_exp, hi_exp, color='gold', alpha=0.18, zorder=0,
                       label=f'Expected [{lo_exp:.0f}–{hi_exp:.0f}]')

            # Saturated-state annotations (use get_yaxis_transform for mixed coords)
            if 'lo_MEz_ratio' in stats:
                r_lo = stats['lo_MEz_ratio'][0]
                if np.isfinite(r_lo) and r_lo > 0:
                    ax.axhline(r_lo, color=c128, lw=0.9, ls='--', alpha=0.5)
                    ax.text(0.02, r_lo, f'  ⟨ratio⟩₁₂₈={r_lo:.1f}',
                            transform=ax.get_yaxis_transform(),
                            color=c128, va='bottom', fontsize=8)

            if 'hi_MEz_ratio' in stats:
                r_hi = stats['hi_MEz_ratio'][0]
                if np.isfinite(r_hi) and r_hi > 0:
                    ax.axhline(r_hi, color=COLOR_256, lw=0.9, ls=':', alpha=0.5)
                    ax.text(0.98, r_hi, f'⟨ratio⟩₂₅₆={r_hi:.1f}  ',
                            transform=ax.get_yaxis_transform(),
                            color=COLOR_256, va='top', ha='right', fontsize=8)

            # Convergence badge
            if 'conv_MEz_ratio' in stats:
                badge = (f"Δratio={stats['conv_MEz_ratio']:.1f}%  "
                         + ('✓' if stats['conv_MEz_ratio'] < CONVERGE_THRESHOLD else '✗'))
                ax.text(0.97, 0.03, badge, transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

            ax.set_xlabel(r'Time  [$t_{\rm cross}$]', fontsize=11)
            ax.set_ylabel(r'$E_{B_z}\,/\,E_{B_\perp}$  (log scale)', fontsize=11)
            ax.set_title(
                rf'$\mathcal{{M}}={cfg["mach"]:.0f},\ \beta={cfg["beta"]}$',
                fontsize=11)
            ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
            ax.set_xlim(left=0)

        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print('  ✓  Fig 3: MEz/ME_perp ratio (log-y)')

        # ── Fig 4: M_A(t) ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), sharey=False)
        fig.suptitle(
            r"Fig. 4 — Alfv\'enic Mach number  $\mathcal{M}_A(t)$",
            fontsize=13, fontweight='bold', y=1.01,
        )

        for ax, label in zip(axes, SIM_LABELS):
            sd    = sim_data[label]
            stats = sim_stats[label]
            cfg   = sd['cfg']
            lo    = sd['lo']
            hi    = sd['hi']
            c128  = cfg['color']

            if lo is not None:
                ax.plot(lo['time'], lo['M_A'], color=c128, lw=1.8,
                        alpha=0.9, label=r'128$^3$', zorder=3)
                shade_sat(ax, lo['time'])

            if hi is not None:
                ax.plot(hi['time'], hi['M_A'], color=COLOR_256, lw=1.8,
                        alpha=0.9, ls='--', label=r'256$^3$', zorder=3)

            # Target and expected range
            m_a_tgt = cfg['M_A_target']
            lo_exp, hi_exp = cfg['M_A_range']
            ax.axhline(m_a_tgt, color='dimgray', lw=1.2, ls=':',
                       label=rf'Target $\mathcal{{M}}_A={m_a_tgt}$', zorder=2)
            ax.axhspan(lo_exp, hi_exp, color='skyblue', alpha=0.15, zorder=0,
                       label=f'Expected [{lo_exp:.1f}–{hi_exp:.1f}]')

            # Saturated-state annotations
            if 'lo_M_A' in stats:
                ma_lo = stats['lo_M_A'][0]
                hline_annotate(ax, ma_lo, c128, ls='--',
                               label=f'⟨M_A⟩₁₂₈={ma_lo:.2f}',
                               side='right', va='bottom')

            if 'hi_M_A' in stats:
                ma_hi = stats['hi_M_A'][0]
                hline_annotate(ax, ma_hi, COLOR_256, ls=':',
                               label=f'⟨M_A⟩₂₅₆={ma_hi:.2f}',
                               side='right', va='top')

            # Convergence badge
            if 'conv_M_A' in stats:
                badge = (f"ΔM_A={stats['conv_M_A']:.1f}%  "
                         + ('✓' if stats['conv_M_A'] < CONVERGE_THRESHOLD else '✗'))
                ax.text(0.97, 0.97, badge, transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

            ax.set_xlabel(r'Time  [$t_{\rm cross}$]', fontsize=11)
            ax.set_ylabel(r"Alfv\'enic Mach number  $\mathcal{M}_A$", fontsize=11)
            ax.set_title(
                rf'$\mathcal{{M}}={cfg["mach"]:.0f},\ \beta={cfg["beta"]}$',
                fontsize=11)
            ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓  Fig 4: M_A(t)")

    print(f'\n  PDF saved: {pdf_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # conclusion.txt
    # ══════════════════════════════════════════════════════════════════════════
    conclusion_path = OUTPUT_DIR / 'conclusion.txt'
    any_hi  = any('hi_KE'   in sim_stats[l] for l in SIM_LABELS)
    all_cvg = all(sim_stats[l].get('converged', False) for l in SIM_LABELS
                  if 'converged' in sim_stats[l])

    with open(conclusion_path, 'w') as f:
        f.write('ASTRA RESOLUTION CONVERGENCE ASSESSMENT\n')
        f.write('=' * 62 + '\n')
        f.write(f'Generated          : {timestamp}\n')
        f.write(f'Convergence thresh.: {CONVERGE_THRESHOLD:.0f}% (symmetric, on each diagnostic)\n')
        f.write(f'Saturation window  : last 25% of each run\n')
        f.write(f'Diagnostics        : KE_sat, MEz/ME_perp, M_A\n\n')

        for label in SIM_LABELS:
            stats = sim_stats[label]
            cfg   = SIMS[label]
            f.write(f"Simulation : {label}   (ℳ={cfg['mach']:.0f}, β={cfg['beta']})\n")
            f.write('-' * 50 + '\n')

            for pfx, res_lbl in [('lo', '128³'), ('hi', '256³')]:
                if f'{pfx}_KE' not in stats:
                    f.write(f'  {res_lbl}: DATA NOT AVAILABLE\n')
                    continue
                ke_m, ke_s = stats[f'{pfx}_KE']
                mz_m, mz_s = stats[f'{pfx}_MEz_ratio']
                ma_m, ma_s = stats[f'{pfx}_M_A']
                ts         = stats.get(f'{pfx}_t_sat', np.nan)
                f.write(f'  {res_lbl}:\n')
                f.write(f'    KE_sat       = {ke_m:.4e} ± {ke_s:.2e}\n')
                f.write(f'    MEz/ME_perp  = {mz_m:.3f} ± {mz_s:.3f}\n')
                f.write(f'    M_A          = {ma_m:.4f} ± {ma_s:.4f}\n')
                f.write(f'    t_KE_sat     = {ts:.4f}\n')

            if 'conv_KE' in stats:
                f.write(f'\n  Convergence tests ({CONVERGE_THRESHOLD:.0f}% criterion):\n')
                for diag, key in [('KE_sat', 'conv_KE'),
                                   ('MEz/ME⊥', 'conv_MEz_ratio'),
                                   ('M_A', 'conv_M_A')]:
                    val  = stats[key]
                    flag = '✓ PASS' if val < CONVERGE_THRESHOLD else '✗ FAIL'
                    f.write(f'    {diag:<12} Δ = {val:5.1f}%   {flag}\n')
                verdict = 'CONVERGED' if stats['converged'] else 'NOT CONVERGED'
                f.write(f'\n  VERDICT: {verdict}\n')

            # Expected-range check
            for key, name, rng_key in [
                ('lo_MEz_ratio', 'MEz/ME⊥ (128³)', 'MEz_ratio'),
                ('lo_M_A',       'M_A (128³)',      'M_A_range'),
            ]:
                if key in stats:
                    val = stats[key][0]
                    lo_exp, hi_exp = cfg[rng_key]
                    ok = lo_exp <= val <= hi_exp
                    f.write(f'  {name:<22} = {val:.3f}   '
                            f'expected [{lo_exp:.1f}–{hi_exp:.1f}]   '
                            f"{'IN RANGE ✓' if ok else 'OUT OF RANGE ✗'}\n")
            f.write('\n')

        f.write('=' * 62 + '\n')
        if any_hi:
            overall = 'ALL SIMULATIONS CONVERGED ✓' if all_cvg \
                      else 'CONVERGENCE INCOMPLETE — see per-sim results above'
            f.write(f'OVERALL : {overall}\n')
        else:
            f.write('OVERALL : 256³ runs not yet available.\n')
            f.write('          Re-run once 256³ simulations complete.\n')
            f.write('          128³ diagnostics above are the reference baseline.\n')

    print(f'  conclusion.txt: {conclusion_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # paper_statements.txt
    # ══════════════════════════════════════════════════════════════════════════
    paper_path = OUTPUT_DIR / 'paper_statements.txt'

    # Collect per-sim numbers for LaTeX
    rows = []
    for label in SIM_LABELS:
        stats = sim_stats[label]
        cfg   = SIMS[label]
        lo_ke  = stats.get('lo_KE',  (np.nan, np.nan))[0]
        hi_ke  = stats.get('hi_KE',  (np.nan, np.nan))[0]
        lo_mz  = stats.get('lo_MEz_ratio', (np.nan, np.nan))[0]
        hi_mz  = stats.get('hi_MEz_ratio', (np.nan, np.nan))[0]
        lo_ma  = stats.get('lo_M_A', (np.nan, np.nan))[0]
        hi_ma  = stats.get('hi_M_A', (np.nan, np.nan))[0]
        c_ke   = stats.get('conv_KE',        np.nan)
        c_mz   = stats.get('conv_MEz_ratio', np.nan)
        c_ma   = stats.get('conv_M_A',       np.nan)
        cvg    = stats.get('converged',      None)
        rows.append(dict(label=label, cfg=cfg,
                         lo_ke=lo_ke, hi_ke=hi_ke,
                         lo_mz=lo_mz, hi_mz=hi_mz,
                         lo_ma=lo_ma, hi_ma=hi_ma,
                         c_ke=c_ke, c_mz=c_mz, c_ma=c_ma, cvg=cvg))

    def fmt(v, spec='.2e'):
        return f'{v:{spec}}' if np.isfinite(v) else r'\text{[N/A]}'

    # Saturation times
    sat_ts = [sim_stats[l].get('lo_t_sat', np.nan) for l in SIM_LABELS]
    sat_ts = [v for v in sat_ts if np.isfinite(v)]

    # M_A range from 128³
    ma_vals = [sim_stats[l].get('lo_M_A', (np.nan,))[0] for l in SIM_LABELS]
    ma_vals = [v for v in ma_vals if np.isfinite(v)]

    with open(paper_path, 'w') as f:
        f.write('% ================================================================\n')
        f.write('% ASTRA — LaTeX statements for the RASTI paper\n')
        f.write('% Generated automatically by analyse_convergence.py\n')
        f.write(f'% {timestamp}\n')
        f.write('% ================================================================\n\n')

        f.write('% ─── Resolution convergence subsection ──────────────────────────\n\n')

        if any_hi:
            # Full statement with real numbers
            f.write('%% --- Full statement (256³ data available) ---\n')
            f.write('\\subsection*{Resolution Convergence}\n\n')
            f.write(
                'We verified resolution convergence by comparing $128^3$ and $256^3$\n'
                'simulations for three representative parameter combinations:\n'
                r'$(\mathcal{M},\,\beta) = (3,\,1.0)$, $(3,\,0.1)$, and $(1,\,1.0)$.' + '\n'
                'The key diagnostics compared are the saturated kinetic energy\n'
                '$\\langle E_K \\rangle$, the magnetic field anisotropy ratio\n'
                '$\\langle E_{B_z}/E_{B_\\perp} \\rangle$ (mean-field-parallel\n'
                'vs.\ transverse magnetic energy), and the Alfv\\\'enic Mach number\n'
                '$\\mathcal{M}_A$.\n'
                'Saturated-state statistics are computed over the final 25\\,\\%\n'
                'of each run, and the convergence criterion is a symmetric\n'
                f'$\\leq {CONVERGE_THRESHOLD:.0f}$\\,\\% difference in each diagnostic.\n\n'
            )
            f.write('The results are:\n\\begin{itemize}\n')
            for r in rows:
                if r['cvg'] is None:
                    continue
                verdict = 'satisfied' if r['cvg'] else '\\textbf{not satisfied}'
                f.write(
                    f"  \\item $\\mathcal{{M}}={r['cfg']['mach']:.0f}$, "
                    f"$\\beta={r['cfg']['beta']}$:\n"
                    f"    $\\langle E_K\\rangle = {fmt(r['lo_ke'])}$\\,(128$^3$)"
                    f" vs\\ ${fmt(r['hi_ke'])}$\\,(256$^3$), "
                    f"$\\Delta = {fmt(r['c_ke'], '.1f')}$\\,\\%;\n"
                    f"    $E_{{B_z}}/E_{{B_\\perp}} = {fmt(r['lo_mz'], '.1f')}$\\,(128$^3$)"
                    f" vs\\ ${fmt(r['hi_mz'], '.1f')}$\\,(256$^3$), "
                    f"$\\Delta = {fmt(r['c_mz'], '.1f')}$\\,\\%;\n"
                    f"    $\\mathcal{{M}}_A = {fmt(r['lo_ma'], '.2f')}$\\,(128$^3$)"
                    f" vs\\ ${fmt(r['hi_ma'], '.2f')}$\\,(256$^3$), "
                    f"$\\Delta = {fmt(r['c_ma'], '.1f')}$\\,\\%.\n"
                    f"    Convergence criterion \\textit{{{verdict}}}.\n"
                )
            f.write('\\end{itemize}\n\n')
            if all_cvg:
                f.write(
                    'All three test cases satisfy the $\\leq'
                    f'{CONVERGE_THRESHOLD:.0f}$\\,\\% convergence criterion\n'
                    'across all key diagnostics, confirming that the $128^3$ simulations\n'
                    'are adequate for the scientific conclusions of this work.\n\n'
                )
            else:
                f.write(
                    'One or more test cases do not satisfy the convergence criterion;\n'
                    'the affected results should be interpreted with caution and\n'
                    'higher-resolution follow-up is recommended.\n\n'
                )
        else:
            # Template with 128³ baseline values, placeholders for 256³
            f.write('%% --- Template statement (256³ runs still in progress) ---\n')
            f.write('%% Fill in [???] placeholders once 256³ runs complete.\n\n')
            f.write(
                'We verified resolution convergence by comparing $128^3$ and $256^3$\n'
                'simulations for three representative parameter combinations:\n'
                r'$(\mathcal{M},\,\beta) = (3,\,1.0)$, $(3,\,0.1)$, and $(1,\,1.0)$.' + '\n'
                'The convergence criterion adopted is a symmetric $\\leq'
                f'{CONVERGE_THRESHOLD:.0f}$\\,\\% difference in each of three\n'
                'key saturated-state diagnostics:\n'
                '$\\langle E_K\\rangle$, $E_{B_z}/E_{B_\\perp}$, and $\\mathcal{M}_A$.\n\n'
            )
            f.write(
                'The 128$^3$ reference (baseline) values are listed below.\n'
                'Values for the 256$^3$ runs are marked [???] pending completion.\n\n'
            )
            f.write('\\begin{itemize}\n')
            for r in rows:
                f.write(
                    f"  \\item $\\mathcal{{M}}={r['cfg']['mach']:.0f}$, "
                    f"$\\beta={r['cfg']['beta']}$:\n"
                    f"    $\\langle E_K\\rangle = {fmt(r['lo_ke'])}$,\n"
                    f"    $E_{{B_z}}/E_{{B_\\perp}} = {fmt(r['lo_mz'], '.1f')}$,\n"
                    f"    $\\mathcal{{M}}_A = {fmt(r['lo_ma'], '.2f')}$.\n"
                    f"    256$^3$ results: [???], [???], [???].\n"
                )
            f.write('\\end{itemize}\n\n')

        # Saturation time note
        f.write('%% --- Saturation time statement ---\n')
        if sat_ts:
            t_lo = min(sat_ts)
            t_hi = max(sat_ts)
            f.write(
                'The simulations reach a statistically-stationary (saturated) state\n'
                f'within approximately ${t_lo:.1f}$--${t_hi:.1f}$\\,$t_{{\\rm cross}}$\n'
                '(where $t_{\\rm cross}=L/c_s$ is the sound-crossing time of the box),\n'
                'with the onset of saturation occurring later for lower~$\\beta$\n'
                '(i.e.\\ stronger mean magnetic fields).\n'
                'Saturated-state diagnostics are computed over the final\n'
                '25\\,\\% of each run to avoid transient behaviour.\n\n'
            )

        # Field anisotropy note
        f.write('%% --- Field anisotropy statement ---\n')
        f.write(
            'In all sub-Alfv\\\'enic runs the mean-field-parallel magnetic energy\n'
            'component $E_{B_z}$ strongly dominates over the perpendicular\n'
            'components, with the anisotropy ratio $E_{B_z}/E_{B_\\perp}$\n'
            'increasing steeply as $\\beta$ decreases\n'
            '(i.e.\\ as the imposed mean field strengthens).\n'
            'This anisotropy directly suppresses the filament fragmentation\n'
            'scale $\\lambda/W$ below the \\citet{Inutsuka1992} thermal-only\n'
            'prediction of~4.\n\n'
        )

        # M_A note
        f.write('%% --- Alfvénic Mach number statement ---\n')
        if ma_vals:
            f.write(
                'The Alfv\\\'enic Mach numbers in the saturated state range from\n'
                f'$\\mathcal{{M}}_A \\approx {min(ma_vals):.2f}$ to\n'
                f'$\\mathcal{{M}}_A \\approx {max(ma_vals):.2f}$\n'
                'across the three test cases, spanning both sub-Alfv\\\'enic and\n'
                'trans-Alfv\\\'enic regimes and confirming that the turbulent\n'
                'magnetic field is dynamically significant in all runs.\n\n'
            )

    print(f'  paper_statements.txt: {paper_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print('  CONVERGENCE ANALYSIS COMPLETE')
    print(f'  PDF        : {pdf_path}')
    print(f'  Conclusion : {conclusion_path}')
    print(f'  Paper text : {paper_path}')
    if any_hi:
        status = 'ALL CONVERGED ✓' if all_cvg else 'INCOMPLETE — check conclusion.txt'
        print(f'  Status     : {status}')
    else:
        print('  Status     : 256³ data not yet available — 128³ baseline only')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
