#!/usr/bin/env python3
"""
Ray-Based Injection-Recovery Campaign Runner

Parallel execution of synthetic filament tests on a 200 vCPU machine.

Usage:
    # Start Ray cluster
    ray start --head --num-cpus=200 --num-gpus=0

    # Run campaign
    python3 run_campaign.py

Author: ASTRA Agent System
Date: 2026-04-19
"""

import ray
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import h5py
from datetime import datetime
import time

# Import our modules
from synthetic_filament_generator import SyntheticFilamentGenerator
from core_extractor import CoreExtractor


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# Campaign Configuration
# =============================================================================

CAMPAIGN_CONFIG = {
    # Output directory
    'output_dir': 'injection_recovery_results',

    # Synthetic map parameters
    'map_config': {
        'map_size': 256,
        'pixel_scale': 2.0,  # arcsec/pixel
        'distance_pc': 1950.0,  # W3 = 1.95 kpc = 1950 pc (code uses pc)
        'beam_size_fwhm': 18.0  # arcsec (Herschel PACS 70um ~ 5", SPIRE 250um ~ 18")
    },

    # Core extraction parameters
    # min_separation=6 handles the tightest campaign spacing of 1.5W = 7.93 px
    # (would be blocked at 10 px); the prominence filter in core_extractor.py
    # prevents spurious shoulder detections at wide spacings.
    'extraction_config': {
        'threshold_sigma': 3.0,
        'min_pixels': 5,
        'min_separation': 6
    },

    # Campaign 1: Baseline bias characterization
    'campaign_1': {
        'name': 'baseline_bias',
        'description': 'Bias vs true spacing',
        'parameter': 'spacing_true',
        'values': [1.5, 2.0, 2.5, 3.0, 4.0],
        'fixed_params': {
            'n_cores': 7,
            'contrast': 10.0,
            'noise_level': 1.0,
            'background_type': 'flat'
        },
        'n_repeats': 20,
        'total_sims': 100  # 5 spacings × 20 repeats
    },

    # Campaign 2: Core number dependence
    'campaign_2': {
        'name': 'core_number_dependence',
        'description': 'Bias vs number of cores',
        'parameter': 'n_cores',
        'values': [5, 7, 9, 11, 13],
        'fixed_params': {
            'spacing_true': 2.0,
            'contrast': 10.0,
            'noise_level': 1.0,
            'background_type': 'flat'
        },
        'n_repeats': 20,
        'total_sims': 100  # 5 N_cores × 20 repeats
    },

    # Campaign 3: Noise robustness
    'campaign_3a': {
        'name': 'noise_robustness',
        'description': 'Bias vs noise level',
        'parameter': 'noise_level',
        'values': [0.5, 1.0, 2.0],
        'fixed_params': {
            'spacing_true': 2.0,
            'n_cores': 7,
            'contrast': 10.0,
            'background_type': 'flat'
        },
        'n_repeats': 20,
        'total_sims': 60  # 3 noise × 20 repeats
    },

    # Campaign 3b: Background robustness
    'campaign_3b': {
        'name': 'background_robustness',
        'description': 'Bias vs background type',
        'parameter': 'background_type',
        'values': ['flat', 'gradient', 'clumpy'],
        'fixed_params': {
            'spacing_true': 2.0,
            'n_cores': 7,
            'contrast': 10.0,
            'noise_level': 1.0
        },
        'n_repeats': 20,
        'total_sims': 60  # 3 backgrounds × 20 repeats
    },

    # Total simulations
    'total_simulations': 320
}


# =============================================================================
# Ray Remote Functions
# =============================================================================

@ray.remote
def generate_and_process_synthetic_filament(
    sim_id: int,
    campaign_name: str,
    params: Dict,
    map_config: Dict,
    extraction_config: Dict,
    output_dir: str
) -> Dict:
    """
    Generate and process a single synthetic filament.

    This function runs on a Ray worker.

    Parameters
    ----------
    sim_id : int
        Unique simulation ID
    campaign_name : str
        Name of the campaign this simulation belongs to
    params : dict
        Parameters for this specific simulation
    map_config : dict
        Configuration for map generation
    extraction_config : dict
        Configuration for core extraction
    output_dir : str
        Output directory for results

    Returns
    -------
    result : dict
        Dictionary containing all results from this simulation
    """
    try:
        # Create output path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate synthetic map
        generator = SyntheticFilamentGenerator(map_config)
        column_density, metadata = generator.generate_filament(
            spacing_true=params['spacing_true'],
            n_cores=params['n_cores'],
            contrast=params['contrast'],
            width_pc=params.get('width_pc', None),
            noise_level=params['noise_level'],
            background_type=params['background_type'],
            seed=params.get('seed', sim_id)
        )

        # Save synthetic map
        filename = output_path / f"{campaign_name}_sim_{sim_id:04d}.h5"
        generator.save_to_hdf5(column_density, metadata, str(filename))

        # Extract cores and measure spacing
        extractor = CoreExtractor(extraction_config)
        results = extractor.process_synthetic_map(str(filename))

        # Add simulation metadata
        results['sim_id'] = sim_id
        results['campaign_name'] = campaign_name
        results['params'] = params

        return results

    except Exception as e:
        return {
            'sim_id': sim_id,
            'campaign_name': campaign_name,
            'error': str(e),
            'params': params,
            'success': False
        }


def create_simulation_tasks(campaign_config: Dict) -> List[Dict]:
    """
    Create list of all simulation tasks.

    Parameters
    ----------
    campaign_config : dict
        Full campaign configuration

    Returns
    -------
    tasks : list of dict
        List of all simulation tasks to run
    """
    tasks = []

    # Process each campaign
    for campaign_key in ['campaign_1', 'campaign_2', 'campaign_3a', 'campaign_3b']:
        campaign = campaign_config[campaign_key]

        param_name = campaign['parameter']
        param_values = campaign['values']
        fixed_params = campaign['fixed_params']
        n_repeats = campaign['n_repeats']

        # Create tasks for each parameter value and repeat
        sim_id = 0
        for value in param_values:
            for repeat in range(n_repeats):
                # Create parameter dict for this simulation
                params = fixed_params.copy()
                params[param_name] = value
                params['repeat'] = repeat

                # Create unique seed for each simulation
                params['seed'] = 10000 + sim_id

                task = {
                    'sim_id': sim_id,
                    'campaign_name': campaign['name'],
                    'params': params,
                    'map_config': campaign_config['map_config'],
                    'extraction_config': campaign_config['extraction_config'],
                    'output_dir': campaign_config['output_dir']
                }

                tasks.append(task)
                sim_id += 1

    return tasks


def run_campaign_parallel(tasks: List[Dict], batch_size: int = 10):
    """
    Run all simulation tasks in parallel using Ray.

    Parameters
    ----------
    tasks : list of dict
        List of simulation tasks
    batch_size : int
        Number of concurrent tasks (default: 10)
    """
    print(f"\n{'='*70}")
    print(f"INJECTION-RECOVERY CAMPAIGN")
    print(f"{'='*70}")
    print(f"Total simulations: {len(tasks)}")
    print(f"Concurrent workers: {batch_size}")
    print(f"Output directory: {tasks[0]['output_dir']}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path(tasks[0]['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save campaign configuration
    config_file = output_dir / 'campaign_config.json'
    with open(config_file, 'w') as f:
        json.dump(CAMPAIGN_CONFIG, f, indent=2)

    # Submit all tasks to Ray
    print(f"Submitting {len(tasks)} tasks to Ray...")
    start_time = time.time()

    # Create list of remote futures
    futures = []
    for task in tasks:
        future = generate_and_process_synthetic_filament.remote(
            task['sim_id'],
            task['campaign_name'],
            task['params'],
            task['map_config'],
            task['extraction_config'],
            task['output_dir']
        )
        futures.append(future)

    print(f"All tasks submitted. Waiting for completion...")

    # Collect results with progress tracking
    results = []
    n_completed = 0
    n_total = len(futures)

    # Use ray.wait() to process results as they complete
    remaining_futures = futures
    while remaining_futures:
        ready_futures, remaining_futures = ray.wait(remaining_futures, num_returns=min(batch_size, len(remaining_futures)))

        for future in ready_futures:
            try:
                result = ray.get(future)
                results.append(result)
                n_completed += 1

                # Progress update
                if n_completed % 10 == 0 or n_completed == n_total:
                    elapsed = time.time() - start_time
                    rate = n_completed / elapsed
                    eta = (n_total - n_completed) / rate
                    print(f"Progress: {n_completed}/{n_total} ({100*n_completed/n_total:.1f}%) | "
                          f"Rate: {rate:.2f} sims/sec | ETA: {eta/60:.1f} min")

            except Exception as e:
                print(f"Error getting result: {e}")

    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETED")
    print(f"{'='*70}")
    print(f"Total simulations: {n_total}")
    print(f"Successfully processed: {len(results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {n_total/total_time:.2f} simulations/second")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")

    return results


def analyze_results(results: List[Dict], output_dir: str):
    """
    Analyze results from injection-recovery campaign.

    Parameters
    ----------
    results : list of dict
        Results from all simulations
    output_dir : str
        Output directory for analysis products
    """
    print("Analyzing results...")

    output_path = Path(output_dir)

    # Filter successful results
    successful_results = [r for r in results if 'error' not in r and r.get('recovery_success', False)]
    failed_results = [r for r in results if 'error' in r or not r.get('recovery_success', False)]

    print(f"Successful: {len(successful_results)}/{len(results)}")
    print(f"Failed: {len(failed_results)}/{len(results)}")

    if len(successful_results) == 0:
        print("No successful results to analyze!")
        return

    # Organize by campaign
    campaigns = {}
    for result in successful_results:
        campaign_name = result['campaign_name']
        if campaign_name not in campaigns:
            campaigns[campaign_name] = []
        campaigns[campaign_name].append(result)

    # Analyze each campaign
    analysis = {}

    for campaign_name, campaign_results in campaigns.items():
        print(f"\nAnalyzing {campaign_name}...")

        # Extract bias factors
        bias_factors = [r['bias_factor'] for r in campaign_results if not np.isnan(r['bias_factor'])]

        if len(bias_factors) == 0:
            print(f"No valid bias factors for {campaign_name}")
            continue

        # Statistics
        bias_mean = np.mean(bias_factors)
        bias_std = np.std(bias_factors)
        bias_median = np.median(bias_factors)
        bias_percentiles = np.percentile(bias_factors, [16, 50, 84])

        campaign_analysis = {
            'n_sims': len(campaign_results),
            'n_valid_bias': len(bias_factors),
            'bias_mean': float(bias_mean),
            'bias_std': float(bias_std),
            'bias_median': float(bias_median),
            'bias_16th': float(bias_percentiles[0]),
            'bias_84th': float(bias_percentiles[2]),
            'bias_factors': bias_factors
        }

        analysis[campaign_name] = campaign_analysis

        print(f"  Bias factor: {bias_mean:.3f} ± {bias_std:.3f} (median: {bias_median:.3f})")

    # Combined analysis (all campaigns)
    all_bias_factors = []
    for campaign_analysis in analysis.values():
        all_bias_factors.extend(campaign_analysis['bias_factors'])

    if len(all_bias_factors) > 0:
        overall_mean = np.mean(all_bias_factors)
        overall_std = np.std(all_bias_factors)
        overall_median = np.median(all_bias_factors)

        print(f"\n{'='*70}")
        print(f"OVERALL BIAS FACTOR")
        print(f"{'='*70}")
        print(f"Mean:     {overall_mean:.3f}")
        print(f"Std dev:  {overall_std:.3f}")
        print(f"Median:   {overall_median:.3f}")
        print(f"{'='*70}\n")

        overall_p16, overall_p84 = np.percentile(all_bias_factors, [16, 84])
        analysis['overall'] = {
            'n_total': len(all_bias_factors),
            'bias_mean': float(overall_mean),
            'bias_std': float(overall_std),
            'bias_median': float(overall_median),
            'bias_16th': float(overall_p16),
            'bias_84th': float(overall_p84)
        }

    # Save analysis results
    analysis_file = output_path / 'bias_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)

    # Save all results
    results_file = output_path / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(successful_results, f, indent=2, cls=NumpyEncoder)

    print(f"Analysis saved to {analysis_file}")
    print(f"All results saved to {results_file}")

    return analysis


def main():
    """Main execution function."""
    # Initialize Ray
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True)

    # Check available resources
    resources = ray.available_resources()
    print(f"Available CPUs: {resources.get('CPU', 'unknown')}")

    # Create simulation tasks
    tasks = create_simulation_tasks(CAMPAIGN_CONFIG)
    print(f"Created {len(tasks)} simulation tasks")

    # Run campaign in parallel
    results = run_campaign_parallel(tasks, batch_size=200)

    # Analyze results
    analyze_results(results, CAMPAIGN_CONFIG['output_dir'])

    # Shutdown Ray
    ray.shutdown()

    print("\nCampaign complete!")


if __name__ == '__main__':
    main()
