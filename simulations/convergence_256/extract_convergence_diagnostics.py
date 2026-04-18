#!/usr/bin/env python3
"""
ASTRA Convergence Diagnostics Extractor
========================================
Reads Athena++ history (.hst) files from a completed 256³ simulation run and
produces:
  - diagnostics.json   — saturated-state scalar diagnostics
  - energy_history.dat — ASCII time-series of KE, ME, MEz, ME⊥, M_A

Also performs a preliminary convergence check against the corresponding
128³ reference run.

Usage:
  python3 extract_convergence_diagnostics.py \\
      --run-dir  /path/to/convergence_output/M3_beta1.0_256 \\
      --sim-name M3_beta1.0_256 \\
      --ref-dir  /path/to/sweep_output

Author : ASTRA PA (for Glenn J. White, Open University)
Date   : 2026-04-17
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
from datetime import datetime

# ─── Reference 128³ sim name mapping ──────────────────────────────────────────
REF_MAP = {
    "M3_beta1.0_256": "mhd_M03_beta1.0",
    "M3_beta0.1_256": "mhd_M03_beta0.1",
    "M1_beta1.0_256": "mhd_M01_beta1.0",
}

# ─── Expected ranges for sanity check (50% threshold) ─────────────────────────
EXPECTED = {
    "M3_beta1.0_256": {"MEz_ratio": (3.0, 10.0), "M_A": (0.5, 1.5),  "KE_sat_time": (0.5, 1.5)},
    "M3_beta0.1_256": {"MEz_ratio": (50., 100.), "M_A": (0.3, 0.9),  "KE_sat_time": (1.0, 2.0)},
    "M1_beta1.0_256": {"MEz_ratio": (20., 50.),  "M_A": (0.5, 1.1),  "KE_sat_time": (0.25, 0.75)},
}


# ─── HST reader ───────────────────────────────────────────────────────────────
def read_hst(path: str) -> dict:
    """Read Athena++ .hst file → dict of named numpy arrays."""
    cols = None
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') and '=' in line:
                parts = line.strip('#').strip().split()
                cols = [p.split('=')[1] for p in parts]
                continue
            if line.startswith('#'):
                continue
            try:
                data.append([float(x) for x in line.split()])
            except ValueError:
                continue
    if not data:
        return {}
    arr = np.array(data)
    result = {}
    if cols:
        for j, col in enumerate(cols):
            if j < arr.shape[1]:
                result[col] = arr[:, j]
    result['_arr'] = arr
    return result


# ─── Saturation detector ──────────────────────────────────────────────────────
def find_saturation_time(t: np.ndarray, KE: np.ndarray,
                          window_frac: float = 0.1,
                          threshold: float = 0.05) -> float:
    """
    Estimate the KE saturation time as the first t where
    the running std/mean over a window_frac fraction of tlim
    drops below threshold.
    Returns NaN if never saturated.
    """
    nw = max(10, int(window_frac * len(t)))
    for i in range(nw, len(t) - nw):
        segment = KE[i:i + nw]
        if np.mean(segment) > 0:
            if np.std(segment) / np.mean(segment) < threshold:
                return float(t[i])
    return float('nan')


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Extract convergence diagnostics")
    ap.add_argument("--run-dir",  required=True, help="256³ run directory")
    ap.add_argument("--sim-name", required=True, help="Simulation name key")
    ap.add_argument("--ref-dir",  default=None,  help="128³ reference sweep_output directory")
    args = ap.parse_args()

    run_dir  = args.run_dir
    sim_name = args.sim_name
    ref_dir  = args.ref_dir

    print(f"\n{'='*70}")
    print(f"ASTRA Convergence Diagnostics: {sim_name}")
    print(f"{'='*70}")
    print(f"Run directory : {run_dir}")
    print(f"Generated     : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # ── Locate .hst file ──────────────────────────────────────────────────────
    hst_files = glob.glob(os.path.join(run_dir, "*.hst"))
    if not hst_files:
        print("ERROR: No .hst file found in run directory.")
        sys.exit(1)
    hst_path = sorted(hst_files)[0]
    print(f"History file  : {hst_path}")

    d = read_hst(hst_path)
    if not d:
        print("ERROR: .hst file is empty or unreadable.")
        sys.exit(1)

    arr = d['_arr']
    t    = arr[:, 0]
    KE   = arr[:, 6] + arr[:, 7] + arr[:, 8]  # total KE
    KEx  = arr[:, 6]; KEy = arr[:, 7]; KEz = arr[:, 8]
    MEx  = arr[:, 9]; MEy = arr[:, 10]; MEz = arr[:, 11]
    ME_perp = MEx + MEy
    ME_tot  = MEx + MEy + MEz
    rho  = arr[:, 2]  # total mass (= mean density for unit-box)

    # Velocity and Alfvén speed from energy densities
    v_rms = np.where(rho > 0, np.sqrt(2.0 * KE / rho), 0.0)
    v_A   = np.where(ME_tot > 0, np.sqrt(2.0 * ME_tot / rho), 0.0)
    M_A   = np.where(v_A > 0, v_rms / v_A, np.nan)

    MEz_ratio = np.where(ME_perp > 1e-30, MEz / ME_perp, np.inf)

    # ── Saturation window: last 25% of run ────────────────────────────────────
    t_max = t[-1]
    sat_mask = t >= 0.75 * t_max
    if sat_mask.sum() < 10:
        sat_mask = np.ones_like(t, dtype=bool)

    KE_sat      = float(np.mean(KE[sat_mask]))
    MEz_sat     = float(np.mean(MEz[sat_mask]))
    MEperp_sat  = float(np.mean(ME_perp[sat_mask]))
    MEtot_sat   = float(np.mean(ME_tot[sat_mask]))
    MEz_ratio_sat = MEz_sat / MEperp_sat if MEperp_sat > 1e-30 else float('inf')
    M_A_sat     = float(np.nanmean(M_A[sat_mask]))
    rho_sat     = float(np.mean(rho[sat_mask]))

    # ── KE saturation time ────────────────────────────────────────────────────
    t_sat_KE = find_saturation_time(t, KE)

    # ── Alfvénic Mach number from saturated KE and total ME ──────────────────
    v_rms_sat = np.sqrt(2.0 * KE_sat / rho_sat) if rho_sat > 0 else 0.0
    v_A_sat   = np.sqrt(2.0 * MEtot_sat / rho_sat) if rho_sat > 0 else 0.0
    M_A_sat_scalar = v_rms_sat / v_A_sat if v_A_sat > 0 else float('nan')

    # Mach number (sonic) from KE (v_rms / c_s with c_s=1)
    M_sonic_sat = v_rms_sat  # c_s = 1 in code units

    print(f"\nSATURATED-STATE DIAGNOSTICS (t ≥ {0.75*t_max:.2f})")
    print(f"  KE_sat          = {KE_sat:.4e}")
    print(f"  MEz_sat         = {MEz_sat:.4e}")
    print(f"  ME⊥_sat         = {MEperp_sat:.4e}")
    print(f"  MEz/ME⊥_sat     = {MEz_ratio_sat:.2f}")
    print(f"  M_A_sat         = {M_A_sat_scalar:.3f}")
    print(f"  M_sonic_sat     = {M_sonic_sat:.3f}")
    print(f"  t_KE_sat        = {t_sat_KE:.3f}")
    print(f"  Final cycle     = {len(t)}")
    print(f"  Run completed   = {t_max >= 1.99:.0f}  (tlim=2.0)")

    # ── Sanity check ──────────────────────────────────────────────────────────
    warnings = []
    if sim_name in EXPECTED:
        exp = EXPECTED[sim_name]
        lo, hi = exp["MEz_ratio"]
        if not (lo <= MEz_ratio_sat <= hi):
            warnings.append(
                f"MEz/ME⊥={MEz_ratio_sat:.1f} outside expected [{lo},{hi}]")
        lo, hi = exp["M_A"]
        if not (lo <= M_A_sat_scalar <= hi):
            warnings.append(
                f"M_A={M_A_sat_scalar:.3f} outside expected [{lo},{hi}]")
        lo, hi = exp["KE_sat_time"]
        if not np.isnan(t_sat_KE) and not (lo <= t_sat_KE <= hi):
            warnings.append(
                f"t_KE_sat={t_sat_KE:.2f} outside expected [{lo},{hi}]")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ⚠  {w}")
    else:
        print("\n  ✓  All diagnostics within expected ranges")

    # ── Load 128³ reference if available ──────────────────────────────────────
    ref_name = REF_MAP.get(sim_name)
    ref_data = None
    convergence = {}

    if ref_dir and ref_name:
        ref_hst = os.path.join(ref_dir, ref_name,
                               f"{ref_name.replace('mhd_', '').replace('_', '_').replace('M0', 'M0').replace('beta', 'b')}.hst")
        # Try alternate naming
        ref_hst_candidates = glob.glob(os.path.join(ref_dir, ref_name, "*.hst"))
        if ref_hst_candidates:
            ref_d = read_hst(sorted(ref_hst_candidates)[0])
            if ref_d:
                r_arr = ref_d['_arr']
                r_t   = r_arr[:, 0]; r_KE = r_arr[:, 6:9].sum(axis=1)
                r_MEx = r_arr[:, 9]; r_MEy = r_arr[:, 10]; r_MEz = r_arr[:, 11]
                r_ME_perp = r_MEx + r_MEy
                r_ME_tot  = r_MEx + r_MEy + r_MEz
                r_rho  = r_arr[:, 2]
                r_mask = r_t >= 0.75 * r_t[-1]

                r_KE_sat    = float(np.mean(r_KE[r_mask]))
                r_MEz_sat   = float(np.mean(r_MEz[r_mask]))
                r_MEp_sat   = float(np.mean(r_ME_perp[r_mask]))
                r_MEt_sat   = float(np.mean(r_ME_tot[r_mask]))
                r_rho_sat   = float(np.mean(r_rho[r_mask]))
                r_ratio_sat = r_MEz_sat / r_MEp_sat if r_MEp_sat > 1e-30 else float('inf')
                r_vA_sat    = np.sqrt(2.0 * r_MEt_sat / r_rho_sat) if r_rho_sat > 0 else 0
                r_vr_sat    = np.sqrt(2.0 * r_KE_sat / r_rho_sat) if r_rho_sat > 0 else 0
                r_MA_sat    = r_vr_sat / r_vA_sat if r_vA_sat > 0 else float('nan')

                def pct_diff(a, b):
                    return 100.0 * abs(a - b) / (0.5 * (abs(a) + abs(b)) + 1e-30)

                convergence = {
                    "ref_sim":          ref_name,
                    "ref_resolution":   "128³",
                    "test_resolution":  "256³",
                    "KE_sat_128":       r_KE_sat,
                    "KE_sat_256":       KE_sat,
                    "KE_sat_pct_diff":  pct_diff(KE_sat, r_KE_sat),
                    "MEz_ratio_128":    r_ratio_sat,
                    "MEz_ratio_256":    MEz_ratio_sat,
                    "MEz_ratio_pct_diff": pct_diff(MEz_ratio_sat, r_ratio_sat),
                    "M_A_128":          r_MA_sat,
                    "M_A_256":          M_A_sat_scalar,
                    "M_A_pct_diff":     pct_diff(M_A_sat_scalar, r_MA_sat),
                }
                print(f"\nCONVERGENCE vs 128³ ({ref_name}):")
                print(f"  KE_sat:     128³={r_KE_sat:.3e}  256³={KE_sat:.3e}  Δ={convergence['KE_sat_pct_diff']:.1f}%")
                print(f"  MEz/ME⊥:   128³={r_ratio_sat:.2f}  256³={MEz_ratio_sat:.2f}  Δ={convergence['MEz_ratio_pct_diff']:.1f}%")
                print(f"  M_A:        128³={r_MA_sat:.3f}  256³={M_A_sat_scalar:.3f}  Δ={convergence['M_A_pct_diff']:.1f}%")
                converged = all([
                    convergence['KE_sat_pct_diff']      < 10.0,
                    convergence['MEz_ratio_pct_diff']   < 10.0,
                    convergence['M_A_pct_diff']         < 10.0,
                ])
                convergence['converged'] = converged
                print(f"\n  Convergence assessment: {'✓ CONVERGED (<10% for all diagnostics)' if converged else '✗ NOT CONVERGED (>10% difference in at least one diagnostic)'}")

    # ── Write diagnostics.json ────────────────────────────────────────────────
    diag = {
        "simulation":       sim_name,
        "resolution":       "256^3",
        "generated":        datetime.utcnow().isoformat() + "Z",
        "run_dir":          run_dir,
        "hst_file":         hst_path,
        "n_cycles":         int(len(t)),
        "t_final":          float(t[-1]),
        "completed":        bool(t[-1] >= 1.99),
        "t_KE_saturation":  float(t_sat_KE),
        "saturated_state": {
            "t_window":         [float(0.75 * t_max), float(t_max)],
            "KE_sat":           KE_sat,
            "MEz_sat":          MEz_sat,
            "ME_perp_sat":      MEperp_sat,
            "ME_tot_sat":       MEtot_sat,
            "MEz_over_MEperp":  MEz_ratio_sat,
            "M_A":              M_A_sat_scalar,
            "M_sonic":          M_sonic_sat,
        },
        "warnings":         warnings,
        "convergence":      convergence,
    }
    json_path = os.path.join(run_dir, "diagnostics.json")
    with open(json_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\nWritten: {json_path}")

    # ── Write energy_history.dat ──────────────────────────────────────────────
    dat_path = os.path.join(run_dir, "energy_history.dat")
    header = (
        "# ASTRA Convergence Suite — energy history\n"
        f"# Simulation: {sim_name}  Resolution: 256³\n"
        f"# Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        "# Columns: time  KE_total  MEz  ME_perp  ME_total  M_A  MEz_ratio\n"
    )
    rows = np.column_stack([
        t,
        KE,
        MEz,
        ME_perp,
        ME_tot,
        np.where(np.isfinite(M_A), M_A, -1.0),
        np.where(np.isfinite(MEz_ratio), MEz_ratio, -1.0),
    ])
    with open(dat_path, "w") as f:
        f.write(header)
        np.savetxt(f, rows, fmt="%.6e",
                   delimiter="  ",
                   comments="")
    print(f"Written: {dat_path}")
    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
