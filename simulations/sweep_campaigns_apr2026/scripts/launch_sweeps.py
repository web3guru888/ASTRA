#!/usr/bin/env python3
"""
Launch all 12 filament_spacing sweep simulations on astra-climate.
Campaign 1+2: β-sweep at M=3.0  (β=0.22, 0.32, 0.5, 0.7, 1.0, 1.5, 2.0)
Campaign 3:   M-sweep at β=0.85 (M=1.0, 2.0, 3.0, 4.0, 5.0)
Grid: 256×64×64, 32 MPI procs per sim, runs first 6 sims concurrently
then remaining 6.
"""

import subprocess, sys, os

SSH_KEY  = "/shared/keys/astra-climate.key"
SSH_HOST = "fetch-agi@34.143.130.135"
SSH_OPTS = ["-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20"]

ATHENA   = "/home/fetch-agi/athena/bin/athena"
BASEDIR  = "/home/fetch-agi/filament_sweeps"
NPROC    = 32   # meshblocks = MPI ranks per sim


# ───────────────────────────────────────────────────────────────────
# Sim catalogue
# ───────────────────────────────────────────────────────────────────
sims = []

# C1+C2: β-sweep at M=3
for beta in [0.22, 0.32, 0.50, 0.70, 1.00, 1.50, 2.00]:
    btag = f"{beta:.2f}".replace(".", "p")
    sims.append(dict(
        campaign = "C1C2_beta_sweep",
        run_id   = f"SWEEP_M30_b{btag}",
        beta     = beta,
        mach     = 3.0,
    ))

# C3: M-sweep at β=0.85
for mach in [1.0, 2.0, 3.0, 4.0, 5.0]:
    mtag = f"{mach:.1f}".replace(".", "p")
    sims.append(dict(
        campaign = "C3_mach_sweep",
        run_id   = f"SWEEP_M{mtag}_b0p85",
        beta     = 0.85,
        mach     = mach,
    ))

assert len(sims) == 12, f"Expected 12 sims, got {len(sims)}"


# ───────────────────────────────────────────────────────────────────
# Build one big bash script that:
#  (a) writes all 12 input files
#  (b) launches batch-1 (6 sims), waits, batch-2 (6 sims), waits
# ───────────────────────────────────────────────────────────────────
def athinput(s):
    """Return the contents of the Athena++ input file for sim s."""
    return f"""\
<comment>
problem   = filament_spacing sweep
configure = --prob=filament_spacing -b --mpi --self_gravity=fft -fft

<job>
problem_id = {s['run_id']}

<output1>
file_type  = hdf5
variable   = cons
dt         = 0.2
id         = out

<time>
cfl_number = 0.3
nlim       = -1
tlim       = 4.0

<mesh>
nx1        = 256
x1min      = -8.0
x1max      =  8.0
ix1_bc     = periodic
ox1_bc     = periodic

nx2        = 64
x2min      = -2.0
x2max      =  2.0
ix2_bc     = periodic
ox2_bc     = periodic

nx3        = 64
x3min      = -2.0
x3max      =  2.0
ix3_bc     = periodic
ox3_bc     = periodic

<meshblock>
nx1        = 32
nx2        = 32
nx3        = 32

<hydro>
gamma           = 1.0
iso_sound_speed = 1.0

<gravity>
grav_mean_rho = 1.0

<problem>
four_pi_G    = 39.478418
mach_number  = {s['mach']}
plasma_beta  = {s['beta']}
wavelength   = 2.0
perturb_ampl = 0.01
"""


lines = ["#!/bin/bash", "set -e", f"ATHENA={ATHENA}", f"BASEDIR={BASEDIR}", ""]

# ── write input files ──────────────────────────────────────────────
lines.append("echo '=== Creating run directories and input files ==='")
for s in sims:
    rundir = f"$BASEDIR/{s['campaign']}/{s['run_id']}"
    lines.append(f"mkdir -p {rundir}")
    # use a uniquely-named heredoc sentinel to avoid collisions
    sentinel = f"EOF_{s['run_id']}"
    lines.append(f"cat > {rundir}/{s['run_id']}.in << '{sentinel}'")
    lines.append(athinput(s))
    lines.append(sentinel)
    lines.append(f"echo '  wrote {s['run_id']}.in'")
lines.append("")

# ── launch helper function ─────────────────────────────────────────
lines += [
    "launch_sim() {",
    "  local rid=$1 camp=$2 beta=$3 mach=$4",
    f"  local rundir=$BASEDIR/$camp/$rid",
    "  local logfile=$rundir/run.log",
    f"  mpirun -np {NPROC} $ATHENA -i $rundir/$rid.in -d $rundir > $logfile 2>&1 &",
    "  echo \"  Launched $rid  PID=$!\"",
    "}",
    "",
]

# ── batch 1: first 6 sims ─────────────────────────────────────────
lines.append("echo '=== Batch 1: launching first 6 sims ==='")
for s in sims[:6]:
    lines.append(
        f"launch_sim {s['run_id']} {s['campaign']} {s['beta']} {s['mach']}"
    )
lines += [
    "echo 'Batch 1 launched — waiting …'",
    "wait",
    "echo 'Batch 1 complete'",
    "",
]

# ── batch 2: remaining 6 sims ─────────────────────────────────────
lines.append("echo '=== Batch 2: launching remaining 6 sims ==='")
for s in sims[6:]:
    lines.append(
        f"launch_sim {s['run_id']} {s['campaign']} {s['beta']} {s['mach']}"
    )
lines += [
    "echo 'Batch 2 launched — waiting …'",
    "wait",
    "echo 'ALL 12 SIMS DONE'",
    "",
]

remote_script = "\n".join(lines) + "\n"

# ───────────────────────────────────────────────────────────────────
# Upload the bash script to astra-climate via stdin pipe
# ───────────────────────────────────────────────────────────────────
REMOTE_SCRIPT = "/home/fetch-agi/launch_sweeps.sh"

print("Uploading launch script to astra-climate …")
r = subprocess.run(
    ["ssh"] + SSH_OPTS + [SSH_HOST, f"cat > {REMOTE_SCRIPT} && chmod +x {REMOTE_SCRIPT} && echo UPLOAD_OK"],
    input=remote_script, capture_output=True, text=True, timeout=30
)
if "UPLOAD_OK" not in r.stdout:
    print(f"Upload failed!\nstdout: {r.stdout}\nstderr: {r.stderr}")
    sys.exit(1)
print("  Script uploaded ✓\n")

# ───────────────────────────────────────────────────────────────────
# Verify script looks right
# ───────────────────────────────────────────────────────────────────
r2 = subprocess.run(
    ["ssh"] + SSH_OPTS + [SSH_HOST, f"head -5 {REMOTE_SCRIPT} && echo '...' && wc -l {REMOTE_SCRIPT}"],
    capture_output=True, text=True, timeout=15
)
print("Remote script preview:")
print(r2.stdout)

# ───────────────────────────────────────────────────────────────────
# Launch with nohup so it survives SSH disconnect
# ───────────────────────────────────────────────────────────────────
MASTER_LOG = "/home/fetch-agi/launch_sweeps_master.log"
print("Launching with nohup …")
r3 = subprocess.run(
    ["ssh"] + SSH_OPTS + [SSH_HOST,
     f"nohup bash {REMOTE_SCRIPT} > {MASTER_LOG} 2>&1 & echo NOHUP_PID=$!"],
    capture_output=True, text=True, timeout=20
)
print(f"  {r3.stdout.strip()}")
if r3.stderr:
    print(f"  stderr: {r3.stderr.strip()}")

print()
print("All 12 sims queued successfully.")
print(f"Monitor with:")
print(f"  ssh -i /shared/keys/astra-climate.key {SSH_HOST} 'ps aux | grep mpirun | grep -v grep | wc -l'")
print(f"  ssh -i /shared/keys/astra-climate.key {SSH_HOST} 'tail -30 {MASTER_LOG}'")
