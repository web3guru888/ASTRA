#!/bin/bash
#
# Quick Start Script for Injection-Recovery Campaign
#
# Usage: ./run.sh
#

set -e  # Exit on error

echo "================================================================"
echo "INJECTION-RECOVERY CAMPAIGN QUICK START"
echo "================================================================"
echo ""

# Check if Ray is installed
if ! command -v ray &> /dev/null; then
    echo "ERROR: Ray not found. Install with: pip install ray"
    exit 1
fi

# Check if Ray is already running
if ray status &> /dev/null; then
    echo "Ray is already running. Using existing cluster."
    ray_status=$(ray status)
    echo "$ray_status"
else
    echo "Starting Ray cluster..."
    ray start --head --num-cpus=200 --num-gpus=0 --port=6379 &
    sleep 5
    echo "Ray cluster started."
fi

echo ""
echo "================================================================"
echo "Running Injection-Recovery Campaign"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  Total simulations: 320"
echo "  Expected wall time: 2-4 hours"
echo "  Output directory: injection_recovery_results/"
echo ""

# Run the campaign
python3 run_campaign.py

echo ""
echo "================================================================"
echo "Analyzing Results"
echo "================================================================"
echo ""

python3 analyze_results.py

echo ""
echo "================================================================"
echo "CAMPAIGN COMPLETE"
echo "================================================================"
echo ""
echo "Results saved to: injection_recovery_results/"
echo "  - bias_characterization.png/pdf (main figure)"
echo "  - bias_table.tex (LaTeX table)"
echo "  - paper_text_section.tex (text for paper)"
echo ""
echo "Next steps:"
echo "  1. Review the figures: bias_characterization.png"
echo "  2. Check the bias factor: overall bias = X.XX ± Y.YY"
echo "  3. Insert paper_text_section.tex into your paper"
echo "  4. Update abstract to mention bias first"
echo ""
