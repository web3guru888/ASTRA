"""
W3 Deep-Dive Analysis: Core detection and separation measurement.
Reads 256³ Athena++ HDF5 output, finds gravitational cores,
measures separations, compares with magneto-Jeans theory.
Glenn J. White (Open University), 2026-04-18
"""

import json, glob, sys
import numpy as np
import h5py
from pathlib import Path
from scipy import ndimage

WORK_DIR    = Path("/home/fetch-agi/w3_deepdive")
OUT_DIR     = Path("/home/fetch-agi/analysis_w3")
FOUR_PI_G   = 4.0 * np.pi**2
LAMBDA_J    = 1.0
L           = 16.0
NX          = 256
dx          = L / NX          # 0.0625 λ_J per cell
N_MODES     = 8
LAM_SEED    = L / N_MODES     # 2.0 λ_J

SIMS = [
    {'beta': 0.7, 'name': 'W3_M30_b07'},
    {'beta': 0.8, 'name': 'W3_M30_b08'},
    {'beta': 0.9, 'name': 'W3_M30_b09'},
    {'beta': 1.0, 'name': 'W3_M30_b10'},
]

# ─── Assembly ────────────────────────────────────────────────────────────────
def assemble_density(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        t    = float(f.attrs['Time'])
        prim = np.array(f['prim'], dtype=np.float32)
        locs = np.array(f['LogicalLocations'])
        nvars, nblocks, nz, ny, nx_blk = prim.shape
        max_lx = locs[:,0].max() + 1
        max_ly = locs[:,1].max() + 1
        max_lz = locs[:,2].max() + 1
        full = np.zeros((max_lz*nz, max_ly*ny, max_lx*nx_blk), dtype=np.float32)
        for b in range(nblocks):
            lx, ly, lz = locs[b]
            full[lz*nz:(lz+1)*nz, ly*ny:(ly+1)*ny, lx*nx_blk:(lx+1)*nx_blk] = prim[0,b]
    return t, full

# ─── Core detection ──────────────────────────────────────────────────────────
def find_cores(rho, threshold_factor=2.5, smooth_sigma=1.5):
    """Find local density maxima = gravitational cores."""
    rho_mean = rho.mean()
    threshold = rho_mean * threshold_factor

    # Smooth to remove noise, then find connected over-dense regions
    rho_smooth = ndimage.gaussian_filter(rho, sigma=smooth_sigma)
    above = rho_smooth > threshold

    if not above.any():
        return [], rho_mean

    labeled, n_labels = ndimage.label(above)
    cores = []
    for label in range(1, n_labels + 1):
        mask = labeled == label
        # Centre of mass of the dense region
        indices = np.array(np.where(mask)).T   # (n_cells, 3)
        weights = rho[mask]
        com = np.average(indices, axis=0, weights=weights)  # (k,j,i) in cells
        # Convert to physical coordinates
        pos = com * dx   # in λ_J units
        peak_density = rho[mask].max()
        volume = mask.sum() * dx**3
        cores.append({
            'position': pos.tolist(),
            'peak_rho': float(peak_density),
            'volume': float(volume),
            'contrast': float(peak_density / rho_mean),
        })
    return cores, float(rho_mean)

# ─── Separation statistics ────────────────────────────────────────────────────
def core_separations(cores, L=L):
    """Compute all pairwise nearest-neighbour separations (periodic)."""
    if len(cores) < 2:
        return [], np.nan, np.nan
    
    positions = np.array([c['position'] for c in cores])
    n = len(positions)
    separations = []
    nn_dists = []

    for i in range(n):
        dists_to_others = []
        for j in range(n):
            if i == j: continue
            d = positions[i] - positions[j]
            # Periodic boundary correction
            d = d - L * np.round(d / L)
            dist = np.sqrt((d**2).sum())
            dists_to_others.append(dist)
        nn_dists.append(min(dists_to_others))

    mean_sep = float(np.mean(nn_dists))
    std_sep  = float(np.std(nn_dists))
    return nn_dists, mean_sep, std_sep

# ─── 1D power spectrum ────────────────────────────────────────────────────────
def dominant_spacing_1d(rho):
    """Peak of 1D power spectrum along x1 (averaged over x2,x3)."""
    rho_1d = rho.mean(axis=(0, 1))    # (NX,)
    rho_fluct = rho_1d - rho_1d.mean()
    power = np.abs(np.fft.rfft(rho_fluct))**2
    n = len(rho_1d)
    # Angular wavenumber: k = (2π/L) × mode_number
    freqs = np.fft.rfftfreq(n, d=dx) * 2 * np.pi
    if len(power) > 1:
        pk = np.argmax(power[1:]) + 1
        return float(2*np.pi / freqs[pk]) if freqs[pk] > 0 else np.nan
    return np.nan

# ─── Per-sim analysis ─────────────────────────────────────────────────────────
def analyse_sim(sim):
    name, beta = sim['name'], sim['beta']
    files = sorted((WORK_DIR / name / 'outputs').glob('*.athdf'))
    print(f"\n{'='*60}")
    print(f"  {name}  β={beta}  ({len(files)} snapshots)")

    lam_MJ = np.sqrt(1 + 2/beta)
    k_MJ   = 2*np.pi / lam_MJ
    va2    = 2.0/beta
    gamma2 = FOUR_PI_G - (np.pi)**2 * (1 + va2)   # k_seed = π
    gamma  = np.sqrt(max(gamma2, 0))

    result = {
        'name': name, 'beta': beta,
        'lam_MJ': float(lam_MJ),
        'gamma_seed': float(gamma),
        'snapshots': [],
    }

    for hf in files:
        try:
            t, rho = assemble_density(hf)
        except Exception as e:
            print(f"    Skipping {hf.name}: {e}")
            continue

        C = float(rho.max() / rho.mean())
        lam_1d = dominant_spacing_1d(rho)

        # Core finding (only meaningful when C > 2)
        cores, rho_mean = find_cores(rho, threshold_factor=min(C*0.5, 2.0))
        nn_dists, mean_sep, std_sep = core_separations(cores)

        snap = {
            't': round(t, 3),
            'C': round(C, 4),
            'lam_1d': round(lam_1d, 4) if not np.isnan(lam_1d) else None,
            'n_cores': len(cores),
            'mean_sep': round(mean_sep, 4) if not np.isnan(mean_sep) else None,
            'std_sep': round(std_sep, 4) if not np.isnan(std_sep) else None,
            'cores': cores,
        }
        result['snapshots'].append(snap)
        print(f"  t={t:.2f}: C={C:.3f}  λ_1d={lam_1d:.3f}λ_J  "
              f"n_cores={len(cores)}  <sep>={mean_sep:.3f}λ_J")

    # Summary from final snapshot
    if result['snapshots']:
        last = result['snapshots'][-1]
        peak = max(result['snapshots'], key=lambda s: s['C'])
        result['C_final']       = last['C']
        result['C_peak']        = peak['C']
        result['t_peak']        = peak['t']
        result['n_cores_final'] = last['n_cores']
        result['mean_sep_final']= last['mean_sep']
        result['lam_1d_final']  = last['lam_1d']
        result['lam_MJ_theory'] = float(lam_MJ)
        if last['mean_sep']:
            result['sep_vs_theory'] = round(last['mean_sep'] / lam_MJ, 4)

    return result

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_results = []

    for sim in SIMS:
        r = analyse_sim(sim)
        all_results.append(r)

    # Save
    out = OUT_DIR / 'w3_deepdive_analysis.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out}")

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: W3 Deep-Dive Results")
    print(f"{'='*70}")
    print(f"  Theory: λ_MJ = λ_J × √(1+2/β),  λ_seed = {LAM_SEED:.1f} λ_J  ({N_MODES} modes in L={L})")
    print(f"\n  {'β':>5} {'λ_MJ':>7} {'C_peak':>8} {'t_peak':>8} "
          f"{'N_cores':>8} {'<sep>':>8} {'<sep>/λ_MJ':>12}")
    print("  " + "-"*65)
    for r in all_results:
        print(f"  {r['beta']:>5.1f} {r['lam_MJ']:>7.3f} "
              f"{r.get('C_peak',0):>8.3f} {r.get('t_peak',0):>8.2f} "
              f"{r.get('n_cores_final',0):>8d} "
              f"{r.get('mean_sep_final') or 0:>8.3f} "
              f"{r.get('sep_vs_theory') or 0:>12.3f}")

    print(f"\n  Column guide: sep = mean nearest-neighbour core separation")
    print(f"  sep/λ_MJ: ratio of measured to magneto-Jeans theory prediction")

if __name__ == '__main__':
    main()
