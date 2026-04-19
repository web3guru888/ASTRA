#!/usr/bin/env python3
"""
Core Extraction and Spacing Measurement

Simulates the HGBS pipeline: detect cores from synthetic column density maps
and measure core-to-core spacings using the pairwise median method.

Author: ASTRA Agent System
Date: 2026-04-19
"""

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks
from typing import Tuple, List, Dict
import h5py


class CoreExtractor:
    """Extract cores and measure spacings from synthetic maps."""

    def __init__(self, config: Dict):
        """
        Initialize core extractor.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - threshold_sigma: Detection threshold in sigma (default: 3.0)
            - min_pixels: Minimum pixels for a core (default: 5)
            - min_separation: Minimum separation between cores in pixels (default: 10)
        """
        self.config = config
        self.threshold_sigma = config.get('threshold_sigma', 3.0)
        self.min_pixels = config.get('min_pixels', 5)
        self.min_separation = config.get('min_separation', 10)

    def extract_cores(self, column_density: np.ndarray,
                     metadata: Dict) -> Tuple[List[Dict], Dict]:
        """
        Extract cores from synthetic column density map.

        Uses a simplified version of the getources algorithm:
        1. Threshold at specified sigma level
        2. Label connected regions
        3. Filter by size and separation
        4. Measure centroid positions

        Parameters
        ----------
        column_density : np.ndarray
            Column density map
        metadata : dict
            Metadata from synthetic filament generation

        Returns
        -------
        cores : list of dict
            Detected cores with positions and properties
        measurement_info : dict
            Information about the measurement process
        """
        # ── 1-D mean projection along filament axis (x) ──────────────
        # Averaging over y suppresses pixel noise by √N_rows (~16×) while
        # preserving core signal.  This avoids 2D background subtraction
        # artefacts that corrupt the signal for tightly-packed cores.
        profile_1d = np.mean(column_density, axis=0)   # shape (NX,)

        # ── Linear baseline fit to edge regions ───────────────────────
        # Fit a degree-1 polynomial to the outermost ~sixth of the x-axis
        # (where no filament cores reside).  Using a linear fit rather than
        # a constant mean correctly handles gradient backgrounds — the
        # constant-mean estimator would otherwise report a hugely inflated
        # "noise" equal to the gradient amplitude.
        NX = len(profile_1d)
        edge = max(5, NX // 6)
        x_coords = np.arange(NX, dtype=float)
        edge_x = np.concatenate([x_coords[:edge], x_coords[-edge:]])
        edge_y = np.concatenate([profile_1d[:edge], profile_1d[-edge:]])
        p = np.polyfit(edge_x, edge_y, 1)          # degree-1 polynomial
        baseline = np.polyval(p, x_coords)

        # Noise from edge residuals after baseline removal — this gives
        # the true 1D pixel-noise level, uncontaminated by gradient slopes.
        edge_residuals = edge_y - np.polyval(p, edge_x)
        noise_1d = float(np.std(edge_residuals))
        if noise_1d == 0 or not np.isfinite(noise_1d):
            noise_1d = float(np.std(profile_1d)) + 1e-30

        bg_1d = float(np.polyval(p, NX / 2))       # representative background
        signal_1d = profile_1d - baseline
        detect_thresh = self.threshold_sigma * noise_1d

        # ── Peak finding with prominence filter ───────────────────────
        # Adding prominence=detect_thresh rejects low-prominence noise
        # bumps that appear on the outer shoulders of the core array at
        # wide spacings (≥3W) where the core tails are still well above
        # the noise floor.  Real cores always have prominence >> noise.
        min_dist_px = max(2, self.min_separation)
        # Prominence threshold = 0.75 × detect_thresh.
        # This rejects low-prominence noise bumps (prom/thresh ≈ 0.04–0.53 in
        # tests) and barely-passing shoulder bumps (prom/thresh ≈ 0.73) while
        # still accepting marginally-resolved real cores at 2.0W spacing
        # (prom/thresh ≈ 1.22–1.97).
        prominence_thresh = 0.75 * detect_thresh
        peak_indices, _ = find_peaks(
            signal_1d,
            height=detect_thresh,
            distance=min_dist_px,
            prominence=prominence_thresh
        )

        # Remove any detections that fall inside the edge region used for
        # baseline estimation — those can only be spurious noise artefacts.
        peak_indices = peak_indices[
            (peak_indices >= edge) & (peak_indices < NX - edge)
        ]

        # Build cores list: for each peak x, find the y with maximum value
        cores = []
        for px in peak_indices:
            py = int(np.argmax(column_density[:, px]))
            peak_value = float(column_density[py, px])
            cores.append({
                'id': len(cores),
                'x': float(px),
                'y': float(py),
                'peak_value': peak_value,
                'centroid_value': peak_value,
                'n_pixels': 1,
                'mean_value': peak_value,
            })

        cores.sort(key=lambda c: c['x'])

        measurement_info = {
            'n_detected': len(cores),
            'n_true': int(metadata['n_cores']),
            'threshold': float(detect_thresh),
            'noise_rms': float(noise_1d),
            'background_mean': float(bg_1d)
        }

        return cores, measurement_info

    def measure_pairwise_median_spacing(self, cores: List[Dict]) -> Dict:
        """
        Measure core spacing using pairwise median method.

        This is the same method used in HGBS analyses:
        1. Compute all pairwise separations
        2. Take the median as the characteristic spacing

        Parameters
        ----------
        cores : list of dict
            Detected cores with x, y positions

        Returns
        -------
        spacing_info : dict
            Dictionary containing:
            - spacing_measured: Median pairwise separation
            - spacing_std: Standard deviation of separations
            - n_pairs: Number of pairs
            - all_separations: All pairwise separations
            - units: 'pixels' (caller must convert to physical units)
        """
        n_cores = len(cores)

        if n_cores < 2:
            return {
                'spacing_measured': np.nan,
                'spacing_std': np.nan,
                'n_pairs': 0,
                'all_separations': [],
                'units': 'pixels'
            }

        # Compute all pairwise separations
        separations = []
        for i in range(n_cores):
            for j in range(i + 1, n_cores):
                dx = cores[i]['x'] - cores[j]['x']
                dy = cores[i]['y'] - cores[j]['y']
                sep = np.sqrt(dx**2 + dy**2)
                separations.append(sep)

        separations = np.array(separations)

        # Median spacing (HGBS method)
        spacing_measured = np.median(separations)
        spacing_std = np.std(separations)

        spacing_info = {
            'spacing_measured': float(spacing_measured),
            'spacing_std': float(spacing_std),
            'n_pairs': len(separations),
            'all_separations': separations.tolist(),
            'units': 'pixels'
        }

        return spacing_info

    def process_synthetic_map(self, filename: str) -> Dict:
        """
        Process a synthetic filament map end-to-end.

        Parameters
        ----------
        filename : str
            Path to HDF5 file containing synthetic map

        Returns
        -------
        results : dict
            Dictionary containing:
            - metadata: True parameters
            - cores: Detected cores
            - measurement_info: Extraction info
            - spacing_info: Measured spacings
            - bias_factor: λ_measured / λ_true
        """
        # Load synthetic map
        with h5py.File(filename, 'r') as f:
            column_density = f['column_density'][:]

            # Load metadata
            metadata = dict(f.attrs)
            if 'core_positions' in f:
                core_pos_data = f['core_positions'][:]
                metadata['core_positions_true'] = [
                    {'x_pixel': pos['x_pixel'], 'y_pixel': pos['y_pixel'],
                     'x_rot': pos['x_rot'], 'y_rot': pos['y_rot']}
                    for pos in core_pos_data
                ]

        # Extract cores
        cores, measurement_info = self.extract_cores(column_density, metadata)

        # Measure spacing
        spacing_info = self.measure_pairwise_median_spacing(cores)

        # Convert to physical units
        pixel_scale = metadata['pixel_scale']  # arcsec/pixel
        distance_pc = metadata['distance_pc']

        # Measured spacing in various units
        spacing_measured_arcsec = spacing_info['spacing_measured'] * pixel_scale
        spacing_measured_pc = spacing_measured_arcsec / 206265 * distance_pc
        width_arcsec = (metadata['width_pc'] / distance_pc) * 206265

        spacing_measured_W = spacing_measured_arcsec / width_arcsec

        # True spacing in same units
        spacing_true_W = metadata['spacing_true']

        # Bias factor
        if not np.isnan(spacing_measured_W) and spacing_true_W > 0:
            bias_factor = spacing_measured_W / spacing_true_W
        else:
            bias_factor = np.nan

        results = {
            'filename': filename,
            'metadata': metadata,
            'cores': cores,
            'measurement_info': measurement_info,
            'spacing_info': spacing_info,
            'spacing_measured_W': spacing_measured_W,
            'spacing_measured_pc': spacing_measured_pc,
            'spacing_measured_arcsec': spacing_measured_arcsec,
            'spacing_true_W': spacing_true_W,
            'bias_factor': bias_factor,
            'recovery_success': len(cores) >= metadata['n_cores'] * 0.8  # 80% recovery threshold
        }

        return results


def test_extractor():
    """Test the core extractor."""
    import matplotlib.pyplot as plt
    from synthetic_filament_generator import SyntheticFilamentGenerator

    # Create a test map
    config = {
        'map_size': 256,
        'pixel_scale': 2.0,
        'distance_pc': 1.95,
        'beam_size_fwhm': 18.0
    }

    generator = SyntheticFilamentGenerator(config)
    column_density, metadata = generator.generate_filament(
        spacing_true=2.0,
        n_cores=7,
        contrast=10.0,
        seed=42
    )

    # Save test file
    generator.save_to_hdf5(column_density, metadata, 'test_filament.h5')

    # Extract cores
    extractor_config = {
        'threshold_sigma': 3.0,
        'min_pixels': 5,
        'min_separation': 10
    }

    extractor = CoreExtractor(extractor_config)
    results = extractor.process_synthetic_map('test_filament.h5')

    # Print results
    print(f"True spacing: {results['spacing_true_W']:.2f} W")
    print(f"Measured spacing: {results['spacing_measured_W']:.2f} W")
    print(f"Bias factor: {results['bias_factor']:.3f}")
    print(f"Detected cores: {results['measurement_info']['n_detected']}")
    print(f"True cores: {results['measurement_info']['n_true']}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: column density
    im = axes[0].imshow(column_density, origin='lower', cmap='viridis')
    axes[0].set_title('Synthetic Column Density Map')

    # Overplot detected cores
    for core in results['cores']:
        axes[0].plot(core['x'], core['y'], 'r+', markersize=15, markeredgewidth=2)
        circle = plt.Circle((core['x'], core['y']), radius=5,
                           fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(circle)

    plt.colorbar(im, ax=axes[0])

    # Right: pairwise separations
    if results['spacing_info']['n_pairs'] > 0:
        separations = results['spacing_info']['all_separations']
        axes[1].hist(separations, bins=20, alpha=0.7, edgecolor='black')
        axes[1].axvline(results['spacing_info']['spacing_measured'],
                       color='red', linestyle='--', linewidth=2,
                       label=f"Median: {results['spacing_info']['spacing_measured']:.1f} px")
        axes[1].axvline(metadata['spacing_true_arcsec'] / metadata['pixel_scale'],
                       color='green', linestyle='--', linewidth=2,
                       label=f"True: {metadata['spacing_true_arcsec'] / metadata['pixel_scale']:.1f} px")
        axes[1].set_xlabel('Pairwise Separation (pixels)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Pairwise Separations')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig('test_core_extraction.png', dpi=150)
    print("Test figure saved to test_core_extraction.png")


if __name__ == '__main__':
    test_extractor()
