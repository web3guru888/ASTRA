#!/usr/bin/env python3
"""
Analysis script for filament_spacing sweep campaigns.
Campaign 1+2: β-sweep at M=3.0 (β=0.22, 0.32, 0.50, 0.70, 1.00, 1.50, 2.00)
Campaign 3:   M-sweep at β=0.85 (M=1.0, 2.0, 3.0, 4.0, 5.0)

HDF5 layout (confirmed from file inspection):
  - f.attrs['Time']         → simulation time (float32)
  - f['cons']               → shape (nvar, nblock, nz, ny, nx)
                              VariableNames: dens, mom1, mom2, mom3, phi
  - f['B']                  → shape (3, nblock, nz, ny, nx)
  - f['LogicalLocations']   → shape (nblock, 3) — (ix, iy, iz) zero-based block indices
  - f['Levels']             → shape (nblock,) — refinement level (all 0 for uniform mesh)
  - f.attrs['MeshBlockSize']→ [nx1, nx2, nx3] per block
  - f.attrs['RootGridSize'] → [NX1, NX2, NX3] global

Outputs to /home/fetch-agi/analysis_sweeps/
"""

import os, sys, json, glob, re
import numpy as np

try:
    import h5py
except ImportError:
    os.system("pip install h5py -q")
    import h5py

try:
    from scipy.signal import find_peaks
except ImportError:
    os.system("pip install scipy -q")
    from scipy.signal import find_peaks

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Constants (code units) ─────────────────────────────────────────
LAMBDA_J  = 1.0    # thermal Jeans wavelength in code units (confirmed: 4πG=4π²=39.478)
RHO0      = 1.0    # background density
SEED_LAMBDA = 2.0  # seeded perturbation wavelength

BASEDIR = "/home/fetch-agi/filament_sweeps"
OUTDIR  = "/home/fetch-agi/analysis_sweeps"
os.makedirs(OUTDIR, exist_ok=True)


# ── Sim catalogue ──────────────────────────────────────────────────
sims = []
for beta in [0.22, 0.32, 0.50, 0.70, 1.00, 1.50, 2.00]:
    btag = f"{beta:.2f}".replace(".", "p")
    sims.append(dict(campaign="C1C2_beta_sweep",
                     run_id=f"SWEEP_M30_b{btag}",
                     beta=beta, mach=3.0))
for mach in [1.0, 2.0, 3.0, 4.0, 5.0]:
    mtag = f"{mach:.1f}".replace(".", "p")
    sims.append(dict(campaign="C3_mach_sweep",
                     run_id=f"SWEEP_M{mtag}_b0p85",
                     beta=0.85, mach=mach))


# ── HDF5 reader ────────────────────────────────────────────────────
def read_athdf(fpath):
    """
    Read Athena++ HDF5 snapshot.
    Returns (time, rho_3d) where rho_3d has shape (NX3, NX2, NX1).
    """
    with h5py.File(fpath, "r") as f:
        time = float(f.attrs["Time"])
        cons = f["cons"][()]                # (nvar, nblock, nz, ny, nx)
        locs = f["LogicalLocations"][()]    # (nblock, 3) as (ix, iy, iz)
        bs   = f.attrs["MeshBlockSize"]     # [nx, ny, nz] per block
        gs   = f.attrs["RootGridSize"]      # [NX1, NX2, NX3]

    rho_blocks = cons[0]   # (nblock, nz, ny, nx)
    nblock, bz, by, bx = rho_blocks.shape

    NX1, NX2, NX3 = int(gs[0]), int(gs[1]), int(gs[2])
    rho_3d = np.zeros((NX3, NX2, NX1), dtype=np.float32)

    for ib in range(nblock):
        ix, iy, iz = int(locs[ib, 0]), int(locs[ib, 1]), int(locs[ib, 2])
        x1s = ix * bx;  x2s = iy * by;  x3s = iz * bz
        rho_3d[x3s:x3s+bz, x2s:x2s+by, x1s:x1s+bx] = rho_blocks[ib]

    return time, rho_3d


def make_x1_coords(gs, x1min=-8.0, x1max=8.0):
    NX1 = int(gs[0])
    return np.linspace(x1min, x1max, NX1, endpoint=False) + (x1max - x1min) / (2 * NX1)


# ── Analyse one simulation ─────────────────────────────────────────
def analyse_sim(s):
    rid    = s["run_id"]
    rundir = f"{BASEDIR}/{s['campaign']}/{rid}"
    print(f"\n{'='*60}")
    print(f"  {rid}  (β={s['beta']}, M={s['mach']})")
    print(f"{'='*60}")

    hdf5_files = sorted(glob.glob(f"{rundir}/*.athdf"))
    if not hdf5_files:
        print("  WARNING: no HDF5 files — skipping")
        return None
    print(f"  Found {len(hdf5_files)} snapshots")

    # C(t) curve — read all snapshots
    times, C_vals = [], []
    x1 = None
    for fpath in hdf5_files:
        try:
            t, rho3d = read_athdf(fpath)
            if x1 is None:
                # derive x1 from root grid size
                with h5py.File(fpath, "r") as f:
                    gs = f.attrs["RootGridSize"]
                x1 = make_x1_coords(gs)
            times.append(t)
            C_vals.append(float(rho3d.max() / RHO0))
        except Exception as e:
            print(f"  WARN: {os.path.basename(fpath)}: {e}")

    if not times:
        print("  ERROR: all snapshots unreadable")
        return None

    idx   = np.argsort(times)
    times = np.array(times)[idx]
    C_vals = np.array(C_vals)[idx]
    # re-sort file list
    hdf5_files_sorted = [hdf5_files[i] for i in idx]

    print(f"  t: {times[0]:.2f} → {times[-1]:.2f}")
    print(f"  C: {C_vals[0]:.3f} → {C_vals[-1]:.3f}  (peak={C_vals.max():.3f})")

    # Full spatial read of last good snapshot
    rho3d_last = None
    t_last = None
    for fpath in hdf5_files_sorted[::-1]:
        try:
            t_last, rho3d_last = read_athdf(fpath)
            break
        except Exception:
            continue

    if rho3d_last is None:
        print("  ERROR: cannot read any snapshot")
        return None

    print(f"  Spatial analysis at t={t_last:.3f}")

    # 1-D profile: average over x2 (axis 1) and x3 (axis 0)
    rho1d = rho3d_last.mean(axis=(0, 1))   # shape (NX1,)

    # Peak finding
    # Use adaptive threshold: 2σ above mean, at minimum 1.1×mean
    mu    = rho1d.mean()
    sigma = rho1d.std()
    height_thresh = max(mu + 2.0*sigma, 1.1*mu)
    min_distance  = max(3, int(len(x1) * 0.03))   # ≥3% of box

    peaks, props = find_peaks(rho1d, height=height_thresh, distance=min_distance)

    # Fallback: lower threshold
    if len(peaks) < 2:
        height_thresh = mu + 0.5*sigma
        peaks, props = find_peaks(rho1d, height=height_thresh, distance=min_distance)

    # Measure spacing
    if len(peaks) >= 2:
        dx         = x1[1] - x1[0]
        spacings   = np.diff(x1[peaks])
        lam_frag   = float(np.mean(np.abs(spacings)))
        lam_frag_s = float(np.std(np.abs(spacings)))
    elif len(peaks) == 1:
        lam_frag, lam_frag_s = float("nan"), 0.0
    else:
        lam_frag, lam_frag_s = float("nan"), 0.0

    ratio = lam_frag / LAMBDA_J if not np.isnan(lam_frag) else float("nan")

    print(f"  N_peaks = {len(peaks)}")
    print(f"  λ_frag  = {lam_frag:.3f} ± {lam_frag_s:.3f} λ_J  (ratio={ratio:.3f})")

    return {
        "run_id"          : rid,
        "beta"            : s["beta"],
        "mach"            : s["mach"],
        "campaign"        : s["campaign"],
        "n_snapshots"     : len(times),
        "t_final"         : float(times[-1]),
        "C_initial"       : float(C_vals[0]),
        "C_final"         : float(C_vals[-1]),
        "C_max"           : float(C_vals.max()),
        "n_peaks"         : int(len(peaks)),
        "lambda_frag"     : lam_frag,
        "lambda_frag_std" : lam_frag_s,
        "lambda_J"        : LAMBDA_J,
        "ratio_frag_J"    : ratio,
        "times"           : times.tolist(),
        "C_vals"          : C_vals.tolist(),
        "rho1d_last"      : rho1d.tolist(),
        "x1_coords"       : x1.tolist(),
        "peak_indices"    : peaks.tolist(),
    }


# ── Run all sims ───────────────────────────────────────────────────
results = []
for s in sims:
    r = analyse_sim(s)
    if r is not None:
        results.append(r)

# Save JSON (without the large arrays for readability; keep them in full JSON)
json_out = f"{OUTDIR}/sweep_analysis.json"
with open(json_out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON saved: {json_out}")


# ── Figures ────────────────────────────────────────────────────────
if not HAS_MPL or not results:
    print("Skipping figures (no matplotlib or no results)")
    sys.exit(0)

c12 = [r for r in results if r["campaign"] == "C1C2_beta_sweep"]
c3  = [r for r in results if r["campaign"] == "C3_mach_sweep"]

# ── Fig 1: C(t) time series ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

cmap_beta = plt.cm.plasma(np.linspace(0.1, 0.9, len(c12)))
for r, col in zip(c12, cmap_beta):
    ax1.semilogy(r["times"], r["C_vals"], color=col, lw=2,
                 label=f"β={r['beta']:.2f}")
ax1.set_xlabel("t  [t_J]", fontsize=11)
ax1.set_ylabel("C = ρ_max / ρ_0", fontsize=11)
ax1.set_title("β-sweep (M=3.0): density contrast vs time", fontsize=12)
ax1.legend(fontsize=8, ncol=4, loc="upper left")
ax1.grid(True, alpha=0.3)

cmap_mach = plt.cm.viridis(np.linspace(0.1, 0.9, len(c3)))
for r, col in zip(c3, cmap_mach):
    ax2.semilogy(r["times"], r["C_vals"], color=col, lw=2,
                 label=f"M={r['mach']:.1f}")
ax2.set_xlabel("t  [t_J]", fontsize=11)
ax2.set_ylabel("C = ρ_max / ρ_0", fontsize=11)
ax2.set_title("M-sweep (β=0.85): density contrast vs time", fontsize=12)
ax2.legend(fontsize=8, ncol=5, loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig1_contrast_vs_time.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig1_contrast_vs_time.pdf", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig1")

# ── Fig 2: 1-D density profiles at t=4 ───────────────────────────
ncols = 4
nrows = int(np.ceil(len(results) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4*nrows))
axes = np.array(axes).ravel()

for i, r in enumerate(results):
    ax = axes[i]
    x1  = np.array(r["x1_coords"])
    rho = np.array(r["rho1d_last"])
    ax.plot(x1, rho, "b-", lw=1.5, label="⟨ρ⟩")
    if r["peak_indices"]:
        pk = np.array(r["peak_indices"])
        ax.plot(x1[pk], rho[pk], "rv", ms=7, zorder=5, label=f"N={r['n_peaks']}")
    ax.set_title(f"{r['run_id']}\nλ={r['lambda_frag']:.2f}±{r['lambda_frag_std']:.2f}, C={r['C_final']:.1f}",
                 fontsize=7.5)
    ax.set_xlabel("x₁ [λ_J]", fontsize=7)
    ax.set_ylabel("⟨ρ⟩", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

for j in range(len(results), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("x₂,x₃-averaged density profiles at t = 4.0 t_J", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig2_density_profiles.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig2_density_profiles.pdf", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig2")

# ── Fig 3: λ_frag and C_final vs β and M ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# λ_frag vs β
ax = axes[0, 0]
betas_c12  = [r["beta"]         for r in c12]
lfrags_c12 = [r["lambda_frag"]  for r in c12]
lerrs_c12  = [r["lambda_frag_std"] for r in c12]
ax.errorbar(betas_c12, lfrags_c12, yerr=lerrs_c12,
            fmt="ko-", ms=7, lw=2, capsize=5, label="Measured λ_frag")
ax.axhline(SEED_LAMBDA, color="C0", ls="--", lw=1.5, label=f"Seed λ={SEED_LAMBDA}")
ax.axhline(LAMBDA_J,    color="C2", ls=":",  lw=1.5, label="λ_J (thermal)")
ax.set_xlabel("Plasma β", fontsize=11); ax.set_ylabel("λ_frag  [λ_J]", fontsize=11)
ax.set_title("β-sweep: fragmentation wavelength  (M=3.0)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# λ_frag vs M
ax = axes[0, 1]
machs_c3   = [r["mach"]        for r in c3]
lfrags_c3  = [r["lambda_frag"] for r in c3]
lerrs_c3   = [r["lambda_frag_std"] for r in c3]
ax.errorbar(machs_c3, lfrags_c3, yerr=lerrs_c3,
            fmt="ko-", ms=7, lw=2, capsize=5, label="Measured λ_frag")
ax.axhline(SEED_LAMBDA, color="C0", ls="--", lw=1.5, label=f"Seed λ={SEED_LAMBDA}")
ax.axhline(LAMBDA_J,    color="C2", ls=":",  lw=1.5, label="λ_J (thermal)")
ax.set_xlabel("Mach number M", fontsize=11); ax.set_ylabel("λ_frag  [λ_J]", fontsize=11)
ax.set_title("M-sweep: fragmentation wavelength  (β=0.85)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# C_final vs β
ax = axes[1, 0]
Cfinal_c12 = [r["C_final"] for r in c12]
ax.semilogy(betas_c12, Cfinal_c12, "ko-", ms=7, lw=2)
ax.set_xlabel("Plasma β", fontsize=11); ax.set_ylabel("C(t=4 t_J)", fontsize=11)
ax.set_title("β-sweep: final density contrast  (M=3.0)", fontsize=11)
ax.grid(True, alpha=0.3)

# C_final vs M
ax = axes[1, 1]
Cfinal_c3  = [r["C_final"] for r in c3]
ax.semilogy(machs_c3, Cfinal_c3, "ko-", ms=7, lw=2)
ax.set_xlabel("Mach number M", fontsize=11); ax.set_ylabel("C(t=4 t_J)", fontsize=11)
ax.set_title("M-sweep: final density contrast  (β=0.85)", fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig3_params_sweep.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig3_params_sweep.pdf", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig3")

# ── Fig 4: N_cores heat map ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar(range(len(c12)), [r["n_peaks"] for r in c12],
       color=[plt.cm.plasma(i/(len(c12)-1)) for i in range(len(c12))])
ax.set_xticks(range(len(c12)))
ax.set_xticklabels([f"β={r['beta']:.2f}" for r in c12], rotation=45, fontsize=8)
ax.set_ylabel("N_peaks (density cores)", fontsize=10)
ax.set_title("β-sweep: core count at t=4 t_J  (M=3.0)", fontsize=11)
ax.axhline(8, color="C0", ls="--", lw=1.5, label="Expected (box/seed λ = 8)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.bar(range(len(c3)), [r["n_peaks"] for r in c3],
       color=[plt.cm.viridis(i/(len(c3)-1)) for i in range(len(c3))])
ax.set_xticks(range(len(c3)))
ax.set_xticklabels([f"M={r['mach']:.1f}" for r in c3], rotation=45, fontsize=8)
ax.set_ylabel("N_peaks (density cores)", fontsize=10)
ax.set_title("M-sweep: core count at t=4 t_J  (β=0.85)", fontsize=11)
ax.axhline(8, color="C0", ls="--", lw=1.5, label="Expected (box/seed λ = 8)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(f"{OUTDIR}/fig4_core_counts.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{OUTDIR}/fig4_core_counts.pdf", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig4")

print(f"\nAll done. Results in {OUTDIR}")
