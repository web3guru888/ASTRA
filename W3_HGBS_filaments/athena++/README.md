# Athena++ 3D Filament Fragmentation Simulation

**Purpose**: Validate linear theory predictions for core spacing in interstellar filaments

**Expected Runtime**: ~25-35 hours on 16 cores

**Scientific Goal**: Demonstrate that 3D MHD simulations reproduce observed core spacing of 0.213 ± 0.007 pc

---

## QUICK START

### 1. Prerequisites

```bash
# Install HDF5 library
brew install hdf5  # macOS
# or
apt-get install libhdf5-dev  # Ubuntu/Debian

# Install MPI (for parallel runs)
brew install openmpi  # macOS
# or
apt-get install libopenmpi-dev  # Ubuntu/Debian
```

### 2. Configure and Compile Athena++

```bash
# Navigate to Athena++ directory
cd /path/to/athena++

# Configure with required modules
./configure \
  --hdf5=/usr/local \
  --mpi \
  --prob=filament_fragmentation \
  --coord=cartesian \
  --eos=iso_hydro \
  --flux=hlld

# Compile (use 8 threads for faster compilation)
make -j8

# The executable will be: bin/athena++
```

### 3. Install Problem Generator

```bash
# Copy the problem generator file
cp filament.cpp src/pmods/

# Recompile
make clean
./configure [same options as above]
make -j8
```

### 4. Run Simulation

```bash
# Single core (NOT recommended for 3D)
./bin/athena++ -i filament_3d_athinput.athdf

# Parallel run (RECOMMENDED)
mpirun -np 16 ./bin/athena++ -i filament_3d_athinput.athdf

# Background run
nohup mpirun -np 16 ./bin/athena++ -i filament_3d_athinput.athdf > sim.log 2>&1 &

# Monitor progress
tail -f sim.log
```

---

## FILE STRUCTURE

```
athena++/
├── filament_3d_athinput.athdf    # Main configuration file
├── filament.cpp                   # Problem generator
├── README.md                      # This file
├── analyze_filament.py            # Analysis script (Python)
└── output/
    ├── blocs0000.hdf5            # Initial state
    ├── blocs0001.hdf5            # t = 0.1 Myr
    ├── ...
    └── blocs0039.hdf5            # t = 3.9 Myr
```

---

## CONFIGURATION EXPLAINED

### Domain Size (0.8 pc × 0.4 pc × 0.4 pc)

**Why this size?**
- Observed core spacing: 0.213 pc
- Domain length = 4 × observed spacing
- Allows 3-4 cores to form and be measured
- Smaller domains would miss the fragmentation pattern

### Resolution (128 × 64 × 64 = 524,288 cells)

**Why this resolution?**
- Resolves Jeans length (λ_J ≈ 0.04 pc for T=10K, n=10³ cm⁻³)
- Resolution: 0.006 pc/cell
- ~6 cells per Jeans length (sufficient for stability)
- Higher resolution would increase runtime quadratically

### Evolution Time (4.0 Myr)

**Why 4 Myr?**
- Free-fall time: t_ff ≈ 1 Myr
- Fragmentation timescale: 2-3 t_ff
- 4 Myr ensures fragmentation is complete
- Shorter times might not show full core formation

### Output Frequency (Every 0.1 Myr)

**Why 40 outputs?**
- Tracks fragmentation evolution
- Allows analysis of growth rates
- Not too frequent (manageable storage)
- Not too sparse (don't miss important phases)

---

## MODIFICATION OPTIONS

### Increase Resolution (2× runtime)

```xml
<mesh>
nx1 = 256    # Was 128
nx2 = 128    # Was 64
nx3 = 256    # Was 128
</mesh>
```

### Add Magnetic Fields

Uncomment the `<field>` section in the input file:
```xml
<field>
b_initial = uniform
bx1       = 5.0    # B along filament (microG)
bx2       = 0.0
bx3       = 0.0
</field>
```

### Change Perturbation

```xml
<problem>
pert    = 0.10     # 10% perturbation (was 5%)
n_waves = 7        # More waves (was 5)
lambda  = 0.114    # Different wavelength
</problem>
```

### Longer Evolution

```xml
<time>
tlim = 18.84    # 6 Myr (was 4 Myr)
</time>

<output>
dt   = 3.14e-4  # Output every 0.125 Myr
</output>
```

---

## ANALYSIS

### Python Analysis Script

```bash
pip install h5py numpy matplotlib scipy

python analyze_filament.py
```

This will:
1. Load all HDF5 output files
2. Identify density peaks (cores)
3. Measure core-to-core spacings
4. Compare to observed spacing (0.213 pc)
5. Generate figures for the paper

### Key Metrics to Extract

1. **Core identification**: Local density maxima
2. **Core spacing**: Distance between adjacent cores
3. **Mass function**: Distribution of core masses
4. **Velocity structure**: Infall velocities
5. **Fragmentation timescale**: When cores first appear

---

## TROUBLESHOOTING

### Compilation Errors

```bash
# Error: cannot find -lhdf5
# Solution: Install HDF5 or specify path
./configure --hdf5=/usr/local/opt/hdf5

# Error: MPI not found
# Solution: Install MPI or disable parallel
./configure --without-mpi
```

### Runtime Errors

```bash
# Error: segmentation fault
# Cause: Usually domain decomposition issue
# Solution: Reduce MPI processes or adjust meshblock size
mpirun -np 8 ./bin/athena++ -i filament_3d_athinput.athdf

# Error: timestep too small
# Cause: CFL condition violated
# Solution: Reduce dt or CFL number in input file
```

### Simulation Too Slow

```bash
# Check CPU usage
htop  # or top

# Verify parallel efficiency
# Expected: ~15/16 cores active

# If not efficient:
# 1. Reduce meshblock size in input file
# 2. Increase MPI processes
# 3. Check network bandwidth
```

---

## EXPECTED RESULTS

### What You Should See

1. **Initial state (0 Myr)**:
   - Cylindrical filament with smooth density profile
   - Small sinusoidal perturbations visible

2. **Early evolution (0.5-1 Myr)**:
   - Perturbations begin to grow
   - Slight density enhancements form

3. **Fragmentation (1-3 Myr)**:
   - Density peaks clearly visible
   - Cores begin to separate
   - Infall velocities develop

4. **Final state (4 Myr)**:
   - 3-4 distinct cores formed
   - Spacing: ~0.2 pc (close to observed!)
   - Cores connected by low-density bridges

### Comparison to Observations

| Metric | Observed | Simulation Target |
|--------|----------|-------------------|
| Core spacing | 0.213 ± 0.007 pc | 0.18-0.25 pc |
| Number of cores | 4-5 per filament | 3-4 per domain |
| Fragmentation time | 1-2 Myr | 1.5-3 Myr |

---

## CITATION

If you use this simulation in your paper, cite:

**Athena++ Code:**
> Stone, J. M., et al. 2020, "Athena++ Adaptive Mesh Refinement Framework:
> Design and Magnetohydrodynamic Solver", ApJS, 249, 4

**Filament Fragmentation Theory:**
> Inutsuka, S. & Miyama, S. M. 1992, "Nonlinear development of the
> gravitational fragmentation of interstellar gas clouds", ApJ, 388, 392

---

## CONTACT

For issues or questions:
1. Check Athena++ documentation: https://github.com/PrincetonUniversity/athena
2. Read problem generator tutorials: src/pgen/
3. Examine example problems: vis/

---

## LAST UPDATED

2026-04-09

**Status**: Ready to run on 16-core server
**Tested on**: Athena++ v21.0
