"""
W3 Deep-Dive: L1=16λ_J, 256³, M=3, β=0.7-1.0
Glenn J. White (Open University), 2026-04-18
Target: Resolve multiple gravitational cores with measurable separations.

Seeding: n=8 modes at λ=2.0 λ_J (magneto-Jeans scale for β~0.7).
Expected: 8 cores per sim with ~2λ_J spacing (β-dependent shift).
"""

import subprocess, json, time, glob, sys, signal
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
ATHENA_BIN   = "/home/fetch-agi/athena/bin/athena"
WORK_DIR     = Path("/home/fetch-agi/w3_deepdive")
STATUS_FILE  = Path("/home/fetch-agi/w3_deepdive_status.json")
LOG_DIR      = WORK_DIR / "logs"
N_PROCS      = 32          # MPI procs per sim  (4×32=128 < 200 CPUs → all concurrent)
NX           = 256         # cells per dimension
L            = 16.0        # domain side = 16 λ_J (λ_J=1.0 code unit)
FOUR_PI_G    = 4.0 * np.pi**2   # 39.478418 → λ_J=1.0
TLIM         = 4.0         # code units ≈ 25 t_ff
DT_OUTPUT    = 0.4         # → 11 snapshots: t=0.0,0.4,...,4.0
N_MODES      = 8           # perturbation mode count → λ_seed=2.0 λ_J
LAM_SEED     = L / N_MODES # 2.0 λ_J exactly (fits periodic BC cleanly)
MACH         = 3.0

SIMS = [
    {'beta': 0.7, 'name': 'W3_M30_b07'},
    {'beta': 0.8, 'name': 'W3_M30_b08'},
    {'beta': 0.9, 'name': 'W3_M30_b09'},
    {'beta': 1.0, 'name': 'W3_M30_b10'},
]

# ─── Input file writer ────────────────────────────────────────────────────────
def write_input(sim):
    beta, name = sim['beta'], sim['name']
    sim_dir = WORK_DIR / name
    (sim_dir / 'outputs').mkdir(parents=True, exist_ok=True)

    content = f"""<comment>
problem   = W3 deep-dive: M={MACH:.1f}, beta={beta:.1f}, 256^3, L=16xLambdaJ
reference = Glenn J. White (Open University), 2026-04-18

<job>
problem_id = {name}

<time>
cfl_number  = 0.4
nlim        = -1
tlim        = {TLIM}
integrator  = vl2

<mesh>
nx1         = {NX}
x1min       = 0.0
x1max       = {L}
ix1_bc      = periodic
ox1_bc      = periodic

nx2         = {NX}
x2min       = 0.0
x2max       = {L}
ix2_bc      = periodic
ox2_bc      = periodic

nx3         = {NX}
x3min       = 0.0
x3max       = {L}
ix3_bc      = periodic
ox3_bc      = periodic

<meshblock>
nx1         = 32
nx2         = 32
nx3         = 32

<hydro>
iso_sound_speed = 1.0

<problem>
four_pi_G    = {FOUR_PI_G:.6f}
mach_number  = {MACH:.4f}
plasma_beta  = {beta:.4f}
wavelength   = {LAM_SEED:.6f}
perturb_ampl = 0.01

<output1>
file_type   = hdf5
variable    = prim
id          = out1
dt          = {DT_OUTPUT}
"""
    path = sim_dir / f"{name}.in"
    path.write_text(content)
    return path

# ─── Assemble 3D density field from Athena++ HDF5 ─────────────────────────────
def assemble_density(hdf5_path):
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        t    = float(f.attrs['Time'])
        prim = np.array(f['prim'], dtype=np.float32)   # (nvars, nblocks, nz, ny, nx)
        locs = np.array(f['LogicalLocations'])           # (nblocks, 3)
        nvars, nblocks, nz, ny, nx_blk = prim.shape
        max_lx = locs[:,0].max() + 1
        max_ly = locs[:,1].max() + 1
        max_lz = locs[:,2].max() + 1
        full_rho = np.zeros((max_lz*nz, max_ly*ny, max_lx*nx_blk), dtype=np.float32)
        for b in range(nblocks):
            lx, ly, lz = locs[b]
            full_rho[lz*nz:(lz+1)*nz, ly*ny:(ly+1)*ny, lx*nx_blk:(lx+1)*nx_blk] = prim[0,b]
    return t, full_rho

# ─── Quick density stats for progress reporting ───────────────────────────────
def quick_stats(sim_name):
    files = sorted((WORK_DIR / sim_name / 'outputs').glob('*.athdf'))
    if not files:
        return None
    try:
        t, rho = assemble_density(files[-1])
        return {
            't': round(t, 2),
            'n_outputs': len(files),
            'C': round(float(rho.max() / rho.mean()), 3),
            'rho_max': round(float(rho.max()), 3),
        }
    except:
        return {'n_outputs': len(files)}

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    WORK_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    print("=" * 68)
    print("  W3 DEEP-DIVE: L=16λ_J, 256³, M=3.0, β=0.7–1.0")
    print(f"  Seed: {N_MODES} modes at λ={LAM_SEED:.1f}λ_J  →  {N_MODES} expected cores")
    print(f"  tlim={TLIM}  dt_out={DT_OUTPUT}  {N_PROCS} MPI procs × {len(SIMS)} sims")
    print(f"  Started: {now_utc}")
    print("=" * 68)
    sys.stdout.flush()

    # Write input files
    for sim in SIMS:
        inp = write_input(sim)
        print(f"  Input written: {inp}")
    sys.stdout.flush()

    # Launch all sims simultaneously with mpirun
    procs = {}
    for sim in SIMS:
        name = sim['name']
        sim_dir = WORK_DIR / name
        log_path = LOG_DIR / f"{name}.log"
        cmd = [
            "mpirun", "--oversubscribe", f"-np", str(N_PROCS), "--bind-to", "none",
            ATHENA_BIN, "-i", str(sim_dir / f"{name}.in"), "-d", str(sim_dir / "outputs/")
        ]
        with open(log_path, 'w') as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        procs[name] = {'proc': p, 'start': time.time(), 'beta': sim['beta']}
        print(f"  Launched {name} (PID {p.pid})")
        sys.stdout.flush()

    print(f"\nAll {len(SIMS)} simulations running. Monitoring...\n")
    sys.stdout.flush()

    # Monitor loop
    results = {}
    while True:
        time.sleep(60)
        elapsed = time.time() - t0
        still_running = {n: d for n, d in procs.items() if n not in results}

        for name, d in list(still_running.items()):
            p = d['proc']
            rc = p.poll()
            if rc is not None:
                wall = time.time() - d['start']
                stats = quick_stats(name)
                results[name] = {
                    'beta': d['beta'], 'returncode': rc,
                    'elapsed_s': round(wall, 1),
                    'final_stats': stats,
                    'status': 'PASS' if rc == 0 else 'FAIL',
                }
                print(f"  [{elapsed/60:.1f}m] DONE {name} (β={d['beta']}) "
                      f"rc={rc} wall={wall:.0f}s | {stats}")
                sys.stdout.flush()

        # Progress report for still-running
        running_status = {}
        for name in list(still_running.keys()):
            if still_running[name]['proc'].poll() is None:
                stats = quick_stats(name)
                running_status[name] = stats

        if running_status:
            print(f"\n--- [{elapsed/60:.1f} min elapsed] Running: {len(running_status)} sims ---")
            for name, stats in running_status.items():
                if stats:
                    print(f"  {name}: t={stats.get('t','?')}  C={stats.get('C','?')}  "
                          f"n_out={stats.get('n_outputs','?')}")
                else:
                    print(f"  {name}: starting...")
            if results:
                print(f"  Completed: {list(results.keys())}")
            sys.stdout.flush()

        if not running_status and len(results) == len(SIMS):
            break

    total_wall = time.time() - t0
    print(f"\n{'='*68}")
    print(f"  ALL DONE  —  {len(results)}/{len(SIMS)} sims  —  "
          f"wall time {total_wall/60:.1f} min")
    for name, r in results.items():
        s = r.get('final_stats') or {}
        print(f"  {name}: β={r['beta']} | {r['status']} | "
              f"t={s.get('t','?')} C={s.get('C','?')} | {r['elapsed_s']:.0f}s")

    status = {
        'batch': 'DONE',
        'n_sims': len(SIMS),
        'n_done': len(results),
        'n_failed': sum(1 for r in results.values() if r['status']=='FAIL'),
        'elapsed_h': round(total_wall/3600, 3),
        'started': now_utc,
        'results': results,
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    print(f"\nStatus saved: {STATUS_FILE}")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
