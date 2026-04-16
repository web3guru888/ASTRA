#!/bin/bash
#==============================================================================
# ASTRA MHD β-Sweep V2 — Tension Model Calibration Sims
# Sim A: M3, β=0.75  (tension model predicts λ/W = 2.04 — TARGET RANGE)
# Sim B: M3, β=0.50  (tension model predicts λ/W = 1.79 — LOWER BRACKET)
# Purpose: Map λ/W vs β curve through the observed 2.1±0.1 value
# Est. runtime: ~12 hrs each, ~24 hrs total (12 MPI cores)
# Date: 2026-04-14
#==============================================================================

ATHENA_BIN="/workspace/athena/bin/athena_turb"
CONFIG_DIR="/workspace/athena/sweep_configs"
OUTPUT_BASE="/workspace/athena/sweep_output"
NPROCS=16   # FFT driver requires 16 for 128^3 / 32^3 meshblock decomposition
LOG_DIR="${OUTPUT_BASE}/logs"
RESULTS_FILE="${OUTPUT_BASE}/beta_sweep_v2_results.json"

mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

CONFIGS=("mhd_M03_beta0.75" "mhd_M03_beta0.5")
LABELS=("Sim A: Mach 3, β=0.75 (tension model target)" "Sim B: Mach 3, β=0.50 (lower bracket)")
BETAS=(0.75 0.5)

START_TIME=$(date +%s)
echo "${START_TIME}" > "${OUTPUT_BASE}/beta_sweep_v2_start_time.txt"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ASTRA MHD β-Sweep V2 — Tension Model Calibration                 ║"
echo "║  2 sims × 128³ × 12 MPI cores                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Goal: Map λ/W vs β curve through observed 2.1 ± 0.1"
echo "  Tension model: λ/W = 4/√(1 + 2/β)"
echo "    β=0.50 → λ/W = 1.79 (lower bracket)"
echo "    β=0.75 → λ/W = 2.04 (in target range)"
echo "    β=0.76 → λ/W = 2.10 (analytical exact match)"
echo "    β=1.00 → λ/W = 2.31 (existing sim)"
echo ""
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

SUCCESSES=0
FAILURES=0
declare -A SIM_RUNTIMES

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    label="${LABELS[$i]}"
    beta="${BETAS[$i]}"
    input="${CONFIG_DIR}/${config}.athinput"
    run_dir="${OUTPUT_BASE}/${config}"
    log_file="${LOG_DIR}/${config}.log"

    mkdir -p "${run_dir}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$(( i + 1 ))/${#CONFIGS[@]}] ${label}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Input:  ${input}"
    echo "  Output: ${run_dir}"
    echo "  Log:    ${log_file}"

    sim_start=$(date +%s)
    echo "  Started: $(date -u '+%H:%M:%S UTC')"
    echo ""

    mpirun -np ${NPROCS} --oversubscribe --allow-run-as-root \
        "${ATHENA_BIN}" -i "${input}" -d "${run_dir}" \
        > "${log_file}" 2>&1
    exit_code=$?

    sim_end=$(date +%s)
    sim_elapsed=$(( sim_end - sim_start ))
    sim_min=$(( sim_elapsed / 60 ))
    sim_sec=$(( sim_elapsed % 60 ))
    SIM_RUNTIMES[$config]=$sim_min

    if [ ${exit_code} -eq 0 ]; then
        echo "  ✓ COMPLETED in ${sim_min}m ${sim_sec}s"
        SUCCESSES=$((SUCCESSES + 1))
        echo "  → Now analysing results..."

        # Quick sanity check: count output lines in hst file
        hst_file="${run_dir}/M03_b${beta}.hst"
        if [ -f "${hst_file}" ]; then
            lines=$(wc -l < "${hst_file}")
            echo "  → History file: ${lines} time steps recorded"
        else
            # Try to find any .hst file
            hst_file=$(find "${run_dir}" -name "*.hst" 2>/dev/null | head -1)
            if [ -n "${hst_file}" ]; then
                lines=$(wc -l < "${hst_file}")
                echo "  → History file: ${hst_file##*/} (${lines} time steps)"
            fi
        fi
    else
        echo "  ✗ FAILED (exit code ${exit_code}) after ${sim_min}m ${sim_sec}s"
        echo "  → Last 10 lines of log:"
        tail -10 "${log_file}" | sed 's/^/    /'
        FAILURES=$((FAILURES + 1))
    fi
    echo ""
done

END_TIME=$(date +%s)
TOTAL=$(( END_TIME - START_TIME ))

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  SWEEP COMPLETE"
echo "║  ${SUCCESSES}/2 succeeded, ${FAILURES}/2 failed"
echo "║  Total wall time: $(( TOTAL / 3600 ))h $(( (TOTAL % 3600) / 60 ))m $(( TOTAL % 60 ))s"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Next step: run analyse_beta_sweep_v2.py to extract saturated"
echo "  field properties and compute λ/W for each β."
echo ""

# Write summary JSON
python3 -c "
import json, os
results = {
    'sweep': 'beta_sweep_v2',
    'date': '2026-04-14',
    'purpose': 'Map lambda/W vs beta through observed 2.1 +/- 0.1',
    'sims': [
        {'config': 'mhd_M03_beta0.75', 'beta': 0.75, 'mach_target': 3,
         'tension_prediction': 2.04, 'status': 'completed' if ${SUCCESSES} > 0 else 'failed'},
        {'config': 'mhd_M03_beta0.5',  'beta': 0.50, 'mach_target': 3,
         'tension_prediction': 1.79, 'status': 'completed' if ${SUCCESSES} > 1 else 'failed'},
    ],
    'successes': ${SUCCESSES},
    'failures': ${FAILURES},
    'total_runtime_min': $(( TOTAL / 60 ))
}
with open('${RESULTS_FILE}', 'w') as f:
    json.dump(results, f, indent=2)
print('Results summary written to ${RESULTS_FILE}')
"
