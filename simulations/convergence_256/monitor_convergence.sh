#!/bin/bash
#==============================================================================
# ASTRA Convergence Monitor
# Watches job.log files for the three 256³ convergence simulations and
# prints a live status summary every 60 seconds.
#
# Usage:
#   ./monitor_convergence.sh [--interval N]   (default interval: 60 s)
#   ./monitor_convergence.sh --once           (single snapshot, then exit)
#
# Glenn J. White (Open University) / ASTRA PA, 2026-04-17
#==============================================================================

OUTPUT_BASE="/workspace/athena/convergence_output"
INTERVAL=60
ONCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --interval) INTERVAL=$2; shift 2 ;;
        --once)     ONCE=true; shift ;;
        *) shift ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

SIMS=(M3_beta1.0_256 M3_beta0.1_256 M1_beta1.0_256)
LABELS=(
    "M3 β=1.0  256³  |  Isotropic"
    "M3 β=0.1  256³  |  Highly magnetised"
    "M1 β=1.0  256³  |  Subsonic"
)

# Expected saturation diagnostics for >50% deviation warning
declare -A EXP_MEz_RATIO=([M3_beta1.0_256]="6.5"   [M3_beta0.1_256]="75.0"  [M1_beta1.0_256]="35.0")
declare -A EXP_M_A=(      [M3_beta1.0_256]="1.0"    [M3_beta0.1_256]="0.6"   [M1_beta1.0_256]="0.8")
declare -A EXP_KE_SAT=(   [M3_beta1.0_256]="1.0"    [M3_beta0.1_256]="1.5"   [M1_beta1.0_256]="0.5")

parse_hst() {
    # Usage: parse_hst <hst_file>
    # Prints: time  KE  MEz  MEperp  M_A  cycle  MEz_ratio
    python3 - "$1" <<'PYEOF'
import sys, numpy as np
path = sys.argv[1]
cols = None; data = []
with open(path) as f:
    for line in f:
        if line.startswith('#') and '=' in line:
            parts = line.strip('#').strip().split()
            cols = [p.split('=')[1] for p in parts]
            continue
        if line.startswith('#'): continue
        try:
            data.append([float(x) for x in line.split()])
        except: pass
if not data:
    print("NO_DATA")
    sys.exit(0)
arr = np.array(data)
# Columns: 0=time,1=dt,2=mass,3=1-mom,4=2-mom,5=3-mom,
#          6=1-KE,7=2-KE,8=3-KE,9=1-ME,10=2-ME,11=3-ME
t   = arr[-1, 0]
KE  = arr[-1, 6] + arr[-1, 7] + arr[-1, 8]
MEx = arr[-1, 9]
MEy = arr[-1, 10]
MEz = arr[-1, 11]
ME_perp = MEx + MEy
cs = 1.0  # iso sound speed in code units
rho = arr[-1, 2]
# Alfvén mach: M_A = v_rms / v_A  ≈ sqrt(2*KE/rho) / sqrt(2*ME_perp/rho)
#   (using total ME as proxy for B^2/8pi)
ME_tot = MEx + MEy + MEz
if ME_tot > 0 and rho > 0:
    v_rms = np.sqrt(2.0 * KE / rho)
    v_A   = np.sqrt(2.0 * ME_tot / rho)
    M_A   = v_rms / v_A
else:
    M_A = float('nan')
ratio = MEz / ME_perp if ME_perp > 1e-30 else float('inf')
cycle = len(arr)
# Saturation check: compare last 100 points vs first 100
if len(arr) > 200:
    ke_late = np.mean(arr[-100:, 6:9].sum(axis=1))
    ke_early = np.mean(arr[50:150, 6:9].sum(axis=1))
    saturated = "YES" if ke_late < 1.5 * ke_early else "NO"
else:
    saturated = "?"
print(f"{t:.5f} {KE:.4e} {MEz:.4e} {ME_perp:.4e} {M_A:.3f} {cycle} {ratio:.1f} {saturated}")
PYEOF
}

print_status() {
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  ASTRA Convergence Monitor — $(date -u '+%Y-%m-%d %H:%M:%S UTC')${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    for i in "${!SIMS[@]}"; do
        SIM="${SIMS[$i]}"
        LABEL="${LABELS[$i]}"
        RUN_DIR="${OUTPUT_BASE}/${SIM}"
        JOB_LOG="${RUN_DIR}/job.log"

        echo ""
        echo -e "  ${CYAN}${LABEL}${NC}"

        # Check if running
        if [ -f "${RUN_DIR}/athena.pid" ]; then
            PID=$(cat "${RUN_DIR}/athena.pid")
            if kill -0 $PID 2>/dev/null; then
                STATUS="${GREEN}RUNNING (PID ${PID})${NC}"
            else
                STATUS="${YELLOW}STOPPED (PID ${PID})${NC}"
            fi
        elif [ -f "${JOB_LOG}" ]; then
            if grep -q "Terminating on time limit" "${JOB_LOG}" 2>/dev/null; then
                STATUS="${GREEN}COMPLETE${NC}"
            else
                STATUS="${YELLOW}UNKNOWN (no PID file)${NC}"
            fi
        else
            STATUS="${RED}NOT STARTED${NC}"
        fi
        echo -e "    Status  : ${STATUS}"

        # Parse history file
        HST=$(ls "${RUN_DIR}"/*.hst 2>/dev/null | head -1)
        if [ -n "${HST}" ]; then
            RESULT=$(parse_hst "${HST}" 2>/dev/null)
            if [ "${RESULT}" = "NO_DATA" ] || [ -z "${RESULT}" ]; then
                echo "    History : no data yet"
            else
                read -r TIME KE MEz MEperp M_A CYCLE RATIO SATURATED <<< "$RESULT"
                echo "    t        = ${TIME} / 2.0"
                echo "    Cycle    = ${CYCLE}"
                echo "    KE       = ${KE}"
                echo "    MEz      = ${MEz}  |  ME⊥ = ${MEperp}"
                echo "    MEz/ME⊥  = ${RATIO}   (expected: ~${EXP_MEz_RATIO[$SIM]})"
                echo "    M_A      = ${M_A}    (expected: ~${EXP_M_A[$SIM]})"
                echo "    Saturated: ${SATURATED}"

                # Deviation check
                EXP_R="${EXP_MEz_RATIO[$SIM]}"
                if python3 -c "r,e=float('${RATIO}'),float('${EXP_R}'); print('WARN' if abs(r-e)/e>0.5 else 'OK')" \
                   2>/dev/null | grep -q "WARN"; then
                    echo -e "    ${YELLOW}⚠ MEz/ME⊥ deviates >50% from expectation — check configuration${NC}"
                fi
            fi
        else
            echo "    History : not yet written"
        fi

        # Last few lines of log
        if [ -f "${JOB_LOG}" ]; then
            echo "    Last log entries:"
            grep "^cycle=" "${JOB_LOG}" 2>/dev/null | tail -3 | sed 's/^/      /'
        fi

        # HDF5 count
        N_HDF5=$(ls "${RUN_DIR}"/*.hdf5 2>/dev/null | wc -l)
        echo "    HDF5 snapshots: ${N_HDF5}/40"

        # Disk usage
        if [ -d "${RUN_DIR}" ]; then
            DU=$(du -sh "${RUN_DIR}" 2>/dev/null | cut -f1)
            echo "    Disk usage: ${DU}"
        fi
    done

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

if $ONCE; then
    print_status
else
    while true; do
        print_status
        echo ""
        echo "  (Refreshing every ${INTERVAL}s — Ctrl+C to quit)"
        sleep "${INTERVAL}"
    done
fi
