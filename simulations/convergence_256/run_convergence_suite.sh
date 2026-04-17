#!/bin/bash
#==============================================================================
# ASTRA MHD Resolution Convergence Suite — 256³ × 3 sims × 18 MPI cores
#
# Runs three 256³ Athena++ simulations to test convergence of key diagnostics
# against the 128³ parameter sweep.
#
# Simulations:
#   1. M3_beta1.0_256  — most isotropic case
#   2. M3_beta0.1_256  — highly magnetised (dynamo-quenched)
#   3. M1_beta1.0_256  — subsonic case
#
# Outputs per simulation:
#   convergence_output/{name}/job.log         — Athena++ stdout
#   convergence_output/{name}/*.hdf5          — 3D snapshots every 0.05 t_cs
#   convergence_output/{name}/*.hst           — energy history (dt=0.005)
#   convergence_output/{name}/diagnostics.json
#   convergence_output/{name}/energy_history.dat
#
# Usage:
#   ./run_convergence_suite.sh [--background] [--nprocs N] [--sim NAME]
#
# Glenn J. White (Open University) / ASTRA PA, 2026-04-17
#==============================================================================

ATHENA_BIN="${ATHENA_BIN:-/workspace/athena/bin/athena}"
CONFIG_DIR="/workspace/athena/convergence_configs"
OUTPUT_BASE="/workspace/athena/convergence_output"
NPROCS="${NPROCS:-18}"
BACKGROUND=false
ONLY_SIM=""

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

usage() {
    echo "Usage: $0 [--background] [--nprocs N] [--sim NAME] [--athena PATH]"
    echo ""
    echo "  --background      Run each simulation in the background (sequential)"
    echo "  --nprocs N        Number of MPI ranks (default: 18)"
    echo "  --sim NAME        Run only one sim (M3_beta1.0_256 | M3_beta0.1_256 | M1_beta1.0_256)"
    echo "  --athena PATH     Path to athena binary (default: \$ATHENA_BIN)"
    echo ""
    echo "Storage estimate per simulation: ~37 GB (40 HDF5 snapshots × 256³)"
    echo "Total estimated wall-time: 18–24 h on 18 cores"
}

#─── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --background) BACKGROUND=true; shift ;;
        --nprocs)     NPROCS=$2; shift 2 ;;
        --sim)        ONLY_SIM=$2; shift 2 ;;
        --athena)     ATHENA_BIN=$2; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; usage; exit 1 ;;
    esac
done

#─── Validate binary ───────────────────────────────────────────────────────────
if [ ! -x "${ATHENA_BIN}" ]; then
    echo -e "${RED}✗ Athena++ binary not found at ${ATHENA_BIN}${NC}"
    echo "  Set ATHENA_BIN=/path/to/bin/athena or use --athena PATH"
    exit 1
fi

#─── Simulation catalogue ─────────────────────────────────────────────────────
declare -A CONFIGS=(
    [M3_beta1.0_256]="mhd_M03_beta1.0_256.athinput"
    [M3_beta0.1_256]="mhd_M03_beta0.1_256.athinput"
    [M1_beta1.0_256]="mhd_M01_beta1.0_256.athinput"
)
declare -A LABELS=(
    [M3_beta1.0_256]="Mach 3, β=1.0  — 256³  (most isotropic)"
    [M3_beta0.1_256]="Mach 3, β=0.1  — 256³  (highly magnetised)"
    [M1_beta1.0_256]="Mach 1, β=1.0  — 256³  (subsonic)"
)
ORDERED_SIMS=(M3_beta1.0_256 M3_beta0.1_256 M1_beta1.0_256)

mkdir -p "${OUTPUT_BASE}"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ASTRA Convergence Suite — 256³ × 3 sims × ${NPROCS} MPI cores      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Athena binary : ${ATHENA_BIN}"
echo "  Configs       : ${CONFIG_DIR}"
echo "  Output        : ${OUTPUT_BASE}"
echo "  MPI ranks     : ${NPROCS}"
echo "  Meshblocks    : 512  (256³ / 32³ per block)"
echo "  HDF5 output   : every 0.05 t_cs  → 40 snapshots (~37 GB/sim)"
echo ""

SUITE_START=$(date +%s)
echo "Suite started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

SUCCESSES=0; FAILURES=0

#─── Main loop ─────────────────────────────────────────────────────────────────
for SIM in "${ORDERED_SIMS[@]}"; do
    # Filter to a single sim if requested
    if [ -n "${ONLY_SIM}" ] && [ "${SIM}" != "${ONLY_SIM}" ]; then
        continue
    fi

    LABEL="${LABELS[$SIM]}"
    INPUT="${CONFIG_DIR}/${CONFIGS[$SIM]}"
    RUN_DIR="${OUTPUT_BASE}/${SIM}"
    JOB_LOG="${RUN_DIR}/job.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "  ${CYAN}${LABEL}${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "${INPUT}" ]; then
        echo -e "  ${RED}✗ Input file not found: ${INPUT}${NC}"
        FAILURES=$((FAILURES + 1)); continue
    fi

    mkdir -p "${RUN_DIR}"
    cp "${INPUT}" "${RUN_DIR}/"

    SIM_START=$(date +%s)
    echo "  Started  : $(date -u '+%H:%M:%S UTC')"
    echo "  Run dir  : ${RUN_DIR}"
    echo "  Log      : ${JOB_LOG}"
    echo ""

    # ── Run Athena++ ──────────────────────────────────────────────────────────
    MPI_CMD="mpirun -np ${NPROCS} --oversubscribe --allow-run-as-root"

    if $BACKGROUND; then
        nohup ${MPI_CMD} "${ATHENA_BIN}" \
            -i "${INPUT}" -d "${RUN_DIR}" \
            > "${JOB_LOG}" 2>&1 &
        BGPID=$!
        echo $BGPID > "${RUN_DIR}/athena.pid"
        echo -e "  ${GREEN}✓ Launched in background (PID ${BGPID})${NC}"
        echo "    Monitor: tail -f ${JOB_LOG}"
        echo "    Stop   : kill \$(cat ${RUN_DIR}/athena.pid)"
        echo ""
        SUCCESSES=$((SUCCESSES + 1))
    else
        ${MPI_CMD} "${ATHENA_BIN}" \
            -i "${INPUT}" -d "${RUN_DIR}" \
            > "${JOB_LOG}" 2>&1
        EXIT_CODE=$?

        SIM_END=$(date +%s)
        ELAPSED=$(( SIM_END - SIM_START ))
        EMIN=$(( ELAPSED / 60 )); ESEC=$(( ELAPSED % 60 ))

        if [ ${EXIT_CODE} -eq 0 ]; then
            echo -e "  ${GREEN}✓ Completed in ${EMIN}m ${ESEC}s${NC}"
            SUCCESSES=$((SUCCESSES + 1))
            # ── Post-processing ──────────────────────────────────────────────
            echo "  Extracting diagnostics..."
            python3 /workspace/athena/extract_convergence_diagnostics.py \
                --run-dir "${RUN_DIR}" \
                --sim-name "${SIM}" \
                --ref-dir  "/workspace/athena/sweep_output" \
                2>&1 | sed 's/^/    /'
            echo ""
        else
            echo -e "  ${RED}✗ FAILED (exit ${EXIT_CODE}) after ${EMIN}m ${ESEC}s${NC}"
            echo "    Check: ${JOB_LOG}"
            FAILURES=$((FAILURES + 1))
        fi
    fi
done

#─── Suite summary ─────────────────────────────────────────────────────────────
SUITE_END=$(date +%s)
TOTAL=$(( SUITE_END - SUITE_START ))
echo "╔══════════════════════════════════════════════════════════════════╗"
printf "║  SUITE COMPLETE: %d/3 succeeded, %d/3 failed" "${SUCCESSES}" "${FAILURES}"
echo ""
printf "║  Total time: %dh %dm\n" "$(( TOTAL / 3600 ))" "$(( (TOTAL % 3600) / 60 ))"
echo "╚══════════════════════════════════════════════════════════════════╝"

if [ "$BACKGROUND" = false ] && [ ${SUCCESSES} -gt 0 ]; then
    echo ""
    echo "Running convergence analysis..."
    python3 /workspace/athena/analyse_convergence.py \
        --output-dir "/workspace/athena/convergence_output" \
        --ref-dir    "/workspace/athena/sweep_output" \
        2>&1 | sed 's/^/  /'
fi
