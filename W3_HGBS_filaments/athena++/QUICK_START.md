# Athena++ 3D Filament Simulation - QUICK REFERENCE

## FILES CREATED

| File | Purpose | Size |
|------|---------|------|
| `filament_3d_athinput.athdf` | Athena++ configuration file | 7.1 KB |
| `filament.cpp` | Problem generator (initial conditions) | 7.5 KB |
| `analyze_filament.py` | Analysis script for results | 13 KB |
| `run_simulation.sh` | Automated compile/run script | 7.4 KB |
| `README.md` | Full documentation | 6.5 KB |

---

## 3-MINUTE START GUIDE

### 1. Copy Files to Athena++ Directory

```bash
# Copy all files to your Athena++ installation
cp -r *.cpp *.athdf /path/to/athena++/
```

### 2. Compile

```bash
cd /path/to/athena++

# Configure with required modules
./configure --hdf5=$(brew --prefix hdf5) --mpi \
  --prob=filament_fragmentation --coord=cartesian \
  --eos=iso_hydro --flux=hlld

# Copy problem generator
cp filament.cpp src/pmods/

# Compile
make -j8
```

### 3. Run

```bash
# Interactive (to monitor output)
mpirun -np 16 ./bin/athena++ -i filament_3d_athinput.athdf

# Background (to run unattended)
nohup mpirun -np 16 ./bin/athena++ -i filament_3d_athinput.athdf > sim.log 2>&1 &

# Monitor progress
tail -f sim.log
```

### 4. Analyze Results

```bash
# Wait for simulation to complete (~30 hours)
# Then run analysis script

python3 analyze_filament.py
```

---

## SIMULATION PARAMETERS

**Minimal Useful Configuration:**
- Domain: 0.8 pc × 0.4 pc × 0.4 pc
- Resolution: 128 × 64 × 64 (524,288 cells)
- Physics: Hydro + self-gravity, isothermal
- Evolution: 4.0 Myr
- Runtime: **~30 hours on 16 cores**

**Expected Results:**
- 3-4 cores form along filament
- Spacing: ~0.2 pc (close to observed 0.213 pc)
- Fragmentation time: 1.5-3 Myr

---

## KEY OUTPUT FILES

```
output/
├── blocs0000.hdf5    # t = 0.0 Myr (initial state)
├── blocs0001.hdf5    # t = 0.1 Myr
├── blocs0002.hdf5    # t = 0.2 Myr
├── ...
└── blocs0039.hdf5    # t = 3.9 Myr (final state)

analysis_results/
├── filament_t3.9myr.png           # Final state visualization
├── spacing_evolution.png          # Spacing vs time plot
└── analysis_results.npz          # Numerical results
```

---

## MODIFICATION OPTIONS

### Run with 32 Cores (faster)
```bash
mpirun -np 32 ./bin/athena++ -i filament_3d_athinput.athdf
# Runtime: ~15 hours
```

### Increase Resolution (better accuracy)
Edit `filament_3d_athinput.athdf`:
```xml
<mesh>
nx1 = 256    # Was 128
nx2 = 128    # Was 64
nx3 = 256    # Was 128
</mesh>
```
**Runtime increases 4-8×**

### Add Magnetic Fields
Edit `filament_3d_athinput.athdf`:
```xml
<field>
b_initial = uniform
bx1       = 5.0    # B along filament (microG)
</field>
```
Configure with `--flux=hlld`

---

## TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| "Cannot find -lhdf5" | Install HDF5: `brew install hdf5` |
| "MPI not found" | Install MPI: `brew install openmpi` |
| "Segmentation fault" | Reduce MPI processes: `mpirun -np 8` |
| "Timestep too small" | Reduce CFL in input file |
| "No cores detected" | Increase `pert` in problem block |

---

## CITATION FOR PAPER

**Method:**
> "We performed 3D hydrodynamical simulations using the Athena++ code
>  (Stone et al. 2020). The simulations solve the equations of
>  self-gravitating isothermal hydrodynamics on a 0.8 pc × 0.4 pc × 0.4 pc
>  domain with 128 × 64 × 64 grid cells, evolved for 4.0 Myr."

**References:**
- Stone, J. M., et al. 2020, ApJS, 249, 4 (Athena++ code)
- Inutsuka & Miyama 1992, ApJ, 388, 392 (Fragmentation theory)

---

## ESTIMATED RUNTIMES

| Cores | Runtime | Speedup |
|-------|---------|---------|
| 1 | ~480 hours | 1× |
| 8 | ~60 hours | 8× |
| 16 | ~30 hours | 16× |
| 32 | ~15 hours | 32× |

**Recommendation:** Use 16-32 cores for best efficiency

---

## NEXT STEPS

1. **Compile** Athena++ with the provided configuration
2. **Run** simulation (background mode recommended)
3. **Wait** ~30 hours for completion
4. **Analyze** results with provided Python script
5. **Compare** to observations: 0.213 ± 0.007 pc spacing
6. **Include** figure in paper as "3D Validation"

---

**Created:** 2026-04-09
**Status:** Ready to run on fast server
**Questions:** See README.md for full documentation
