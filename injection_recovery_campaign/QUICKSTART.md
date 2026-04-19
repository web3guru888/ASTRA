# Injection-Recovery Campaign: Quick Start Guide

## What This Does

This campaign runs **320 synthetic filament simulations** to measure the actual bias in the pairwise median method used throughout your paper. This directly addresses Referee Concern #1.

## 3 Commands to Run

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Run the campaign (2-4 hours on 200 CPUs)
./run.sh

# Or manually:
#   ray start --head --num-cpus=200
#   python3 run_campaign.py
#   python3 analyze_results.py
#   ray stop
```

## What You Get

After completion, you'll have:

1. **bias_characterization.pdf** — Figure for paper showing:
   - Bias vs true spacing
   - Bias vs number of cores
   - Bias vs noise level
   - Bias vs background type
   - Overall bias distribution

2. **bias_table.tex** — LaTeX table for paper with statistics

3. **paper_text_section.tex** — Pre-written LaTeX subsection to insert into paper

4. **bias_analysis.json** — Raw bias factor measurements

## The Key Result

The script outputs the overall bias factor: **f = X.XX ± Y.YY**

This tells you exactly how much the pairwise median overestimates the true core spacing.

## How This Addresses the Referee

| Referee's Concern | How This Addresses It |
|-------------------|----------------------|
| "50% unknown bias" | **Measures actual bias** (not theoretical worst-case) |
| "Should be prerequisite" | **Delivers required test** before resubmission |
| "Elaborate machinery" | **Validates all comparisons** with corrected values |
| "Restructure narrative" | **Provides text to restructure** abstract & conclusions |

## Integration with Paper

### Step 1: Check the Result
```bash
# After campaign completes
cat injection_recovery_results/bias_analysis.json | python3 -m json.tool
```

Look for `"overall"` → `"bias_mean"` value.

### Step 2: Update Paper

**If bias < 1.15** (good news):
- Use paper_text_section.tex as-is
- Update abstract to start with bias correction
- All comparisons become stronger

**If bias 1.15-1.30** (manageable):
- Apply correction factor to all measurements
- Increase quoted systematic uncertainties
- Narrative: "After correcting for X% bias..."

**If bias > 1.30** (problematic):
- Refocus on statistical trends
- Emphasize robustness of λ/W ratio
- Narrative: "Absolute scale uncertain, but relative trends robust"

### Step 3: Insert Into Paper

Add to §5 (Systematic Uncertainties):
```latex
% Insert this after current §5.3

\subsection{Injection-Recovery Validation}
\label{sec:injection_recovery}

[Copy content from paper_text_section.tex]
```

Update abstract:
```latex
We quantify systematic uncertainties in core spacing measurements using
Monte Carlo injection-recovery tests, finding a pairwise median bias of
[X.XX ± Y.YY]. After correcting for this bias, HGBS filaments have...
```

## Timeline

- **Setup**: 5 minutes (install dependencies)
- **Run campaign**: 2-4 hours (200 CPUs, 320 sims)
- **Analysis**: 5 minutes
- **Paper integration**: 30 minutes

**Total: ~3-5 hours to address Referee Concern #1**

## Troubleshooting

### Ray won't start
```bash
# Check if running
ray status

# Stop if needed
ray stop

# Start fresh
ray start --head --num-cpus=200
```

### Out of memory
Edit `run_campaign.py`, line ~100:
```python
# Reduce concurrent tasks
results = run_campaign_parallel(tasks, batch_size=50)  # was 200
```

### Test with small sample first
Edit `run_campaign.py`, line ~45:
```python
'n_repeats': 2,  # was 20
```

This gives 32 sims instead of 320 (~15 min).

## Files Created

```
injection_recovery_campaign/
├── synthetic_filament_generator.py  # Generate synthetic maps
├── core_extractor.py                # Extract cores, measure spacings
├── run_campaign.py                  # Ray parallel runner
├── analyze_results.py               # Create figures/tables
├── run.sh                           # Quick start script
├── requirements.txt                 # Python dependencies
└── README.md                        # Detailed documentation

injection_recovery_results/          # Created during run
├── campaign_config.json             # Configuration used
├── all_results.json                 # All simulation results
├── bias_analysis.json               # Statistical analysis
├── bias_characterization.png        # Main results figure
├── bias_characterization.pdf        # Publication version
├── bias_table.tex                   # LaTeX table
└── paper_text_section.tex           # Text for paper
```

## Expected Output

The main figure (`bias_characterization.pdf`) shows 5 panels:
- **A**: Bias vs true spacing (1.5-4.0 W)
- **B**: Bias vs number of cores (5-13)
- **C**: Bias vs noise level (0.5-2.0×)
- **D**: Bias vs background type (flat/gradient/clumpy)
- **E**: Overall bias distribution (histogram)

## Next Steps After Running

1. **Review the bias factor** — Is it <1.15 (good), 1.15-1.30 (OK), or >1.30 (problematic)?

2. **Integrate into paper** — Use the generated LaTeX text and figures

3. **Update abstract** — Put bias correction FIRST

4. **Refine comparisons** — Apply correction to all HGBS measurements

5. **Revise conclusions** — Acknowledge the bias and how you've corrected for it

---

**This directly addresses the referee's primary concern and makes your paper much stronger for resubmission.**
