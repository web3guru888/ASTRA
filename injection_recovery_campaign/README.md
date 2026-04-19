# Injection-Recovery Campaign for Pairwise Median Bias

**Objective**: Quantify the systematic bias in the pairwise median method for measuring core spacings in HGBS filaments through Monte Carlo injection-recovery tests.

**Status**: Ready to run on 200 vCPU machine

---

## Quick Start

### 1. Start Ray Cluster
```bash
# Start Ray head node (use all CPUs)
ray start --head --num-cpus=200 --num-gpus=0

# Or start with custom resources
ray start --head --num-cpus=200 --num-gpus=0 --port=6379
```

### 2. Run Campaign
```bash
# Run full campaign (320 simulations, ~2-4 hours)
python3 run_campaign.py

# Or test with smaller batch first
# Edit run_campaign.py: set batch_size=10 for testing
```

### 3. Analyze Results
```bash
# Generate figures, tables, and paper text
python3 analyze_results.py
```

### 4. Stop Ray
```bash
ray stop
```

---

## Campaign Design

### Total Simulations: 320

| Campaign | Parameter | Values | Fixed | N_repeats | Total |
|----------|-----------|--------|-------|-----------|-------|
| 1 | spacing_true | 1.5, 2.0, 2.5, 3.0, 4.0 W | n_cores=7 | 20 | 100 |
| 2 | n_cores | 5, 7, 9, 11, 13 | spacing=2.0W | 20 | 100 |
| 3a | noise_level | 0.5, 1.0, 2.0× | spacing=2.0W | 20 | 60 |
| 3b | background_type | flat, gradient, clumpy | spacing=2.0W | 20 | 60 |

### Computational Cost

| Metric | Value |
|--------|-------|
| Total simulations | 320 |
| Wall time (200 CPUs) | ~2-4 hours |
| CPU hours | ~400-600 |
| Disk space | ~2 GB for results |
| Memory per worker | ~500 MB |

---

## Expected Results

### Best Case: Bias is modest (f ≈ 1.1 ± 0.1)
- Pairwise median is reliable with small correction
- Can proceed with confidence in theoretical comparisons
- Paper strengthened

### Medium Case: Bias is moderate (f ≈ 1.25 ± 0.15)
- Pairwise median requires correction but remains usable
- Comparisons valid with appropriately larger uncertainties
- Paper reframes with emphasis on uncertainties

### Worst Case: Bias is large (f ≈ 1.5 ± 0.3)
- Pairwise median significantly overestimates true spacing
- Paper must focus on statistical trends, not precise values
- Narrative restructure required (as referee suggested)

---

## Files

### Core Modules
- `synthetic_filament_generator.py` - Generate synthetic Herschel-like maps
- `core_extractor.py` - Extract cores and measure spacings
- `run_campaign.py` - Ray-based parallel campaign runner
- `analyze_results.py` - Create figures, tables, and paper text

### Configuration
- `CAMPAIGN_CONFIG` in `run_campaign.py` - Edit to customize campaign

### Output
- `injection_recovery_results/` - All results and analysis
  - `bias_characterization.png/pdf` - Main results figure
  - `bias_table.tex` - LaTeX table for paper
  - `paper_text_section.tex` - Pre-formatted text to insert
  - `all_results.json` - Raw results
  - `bias_analysis.json` - Statistical analysis

---

## Customization

### Change Map Parameters
Edit `CAMPAIGN_CONFIG['map_config']` in `run_campaign.py`:
```python
'map_config': {
    'map_size': 256,           # Pixels
    'pixel_scale': 2.0,        # arcsec/pixel
    'distance_pc': 1.95,       # Distance (pc)
    'beam_size_fwhm': 18.0     # Beam (arcsec)
}
```

### Change Detection Parameters
Edit `CAMPAIGN_CONFIG['extraction_config']`:
```python
'extraction_config': {
    'threshold_sigma': 3.0,    # Detection threshold
    'min_pixels': 5,           # Min core size
    'min_separation': 10       # Min core separation (pixels)
}
```

### Add More Spacings/N_Cores
Edit campaign definitions in `CAMPAIGN_CONFIG`:
```python
'campaign_1': {
    'values': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # Add more
    ...
}
```

---

## Integration with Paper

### 1. Run Campaign
```bash
python3 run_campaign.py
```

### 2. Generate Paper Materials
```bash
python3 analyze_results.py
```

### 3. Insert Results into Paper
The `analyze_results.py` script generates:
- **bias_characterization.pdf** → Figure for paper (Figure 5 or 6)
- **bias_table.tex** → Table for paper (Table X)
- **paper_text_section.tex** → Pre-formatted LaTeX subsection

### 4. Update Abstract
Insert bias at the beginning:
```latex
We quantify systematic uncertainties in core spacing measurements using
Monte Carlo injection-recovery tests, finding a pairwise median bias of
X.XX ± Y.YY. After correcting for this bias, HGBS filaments have...
```

---

## Testing

### Test Individual Components
```bash
# Test synthetic filament generator
python3 synthetic_filament_generator.py

# Test core extractor
python3 core_extractor.py
```

### Test Small Campaign
Edit `run_campaign.py` to reduce simulations:
```python
# Change n_repeats from 20 to 2 for quick test
'n_repeats': 2,
```

---

## Troubleshooting

### Ray won't start
```bash
# Check if Ray is already running
ray status

# Kill existing Ray
ray stop

# Start fresh
ray start --head --num-cpus=200
```

### Out of memory errors
- Reduce batch_size in `run_campaign.py`
- Reduce map_size in CAMPAIGN_CONFIG

### Files not found
- Ensure all Python files are in same directory
- Check that output directory exists

---

## Contact

Questions? Check the ASTRA documentation or create an issue on GitHub.

---

**Ready to address Referee Concern #1** ✓
