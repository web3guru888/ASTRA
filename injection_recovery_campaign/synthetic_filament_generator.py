#!/usr/bin/env python3
"""
Synthetic Filament Generator for Injection-Recovery Tests

Generates realistic synthetic Herschel-like column density maps of filaments
with known core spacings for bias quantification.

Author: ASTRA Agent System
Date: 2026-04-19
"""

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from typing import Tuple, Dict, Optional
import h5py


class SyntheticFilamentGenerator:
    """Generate synthetic filament observations with known properties."""

    def __init__(self, config: Dict):
        """
        Initialize generator with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - map_size: Size of output map in pixels (default: 512)
            - pixel_scale: Pixel scale in arcsec/pixel (default: 2.0)
            - distance_pc: Distance to filament in pc (default: 1.95)
            - beam_size_fwhm: Instrument beam FWHM in arcsec (default: 18.0)
        """
        self.config = config
        self.map_size = config.get('map_size', 512)
        self.pixel_scale = config.get('pixel_scale', 2.0)  # arcsec/pixel
        self.distance_pc = config.get('distance_pc', 1.95)
        self.beam_size_fwhm = config.get('beam_size_fwhm', 18.0)  # arcsec

        # Convert beam to pixels
        self.beam_sigma = self.beam_size_fwhm / 2.355 / self.pixel_scale

        # Characteristic filament width in HGBS
        self.width_pc = 0.10  # pc

    def generate_filament(
        self,
        spacing_true: float,
        n_cores: int,
        contrast: float = 10.0,
        width_pc: Optional[float] = None,
        noise_level: float = 1.0,
        background_type: str = 'flat',
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a synthetic filament map.

        Parameters
        ----------
        spacing_true : float
            True core spacing in units of filament width W
        n_cores : int
            Number of cores along the filament
        contrast : float
            Peak core contrast relative to background (default: 10.0)
        width_pc : float, optional
            Filament width in pc (default: use HGBS value 0.10 pc)
        noise_level : float
            Noise level relative to Herschel nominal (default: 1.0)
        background_type : str
            Type of background: 'flat', 'gradient', 'clumpy' (default: 'flat')
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        column_density : np.ndarray
            Synthetic column density map (N_h2 in cm^-2)
        metadata : dict
            Metadata containing true parameters and core positions
        """
        if seed is not None:
            np.random.seed(seed)

        # Use default width if not specified
        if width_pc is None:
            width_pc = self.width_pc

        # Convert width to pixels at W3 distance
        # At 1.95 kpc: 0.10 pc = 0.10 / 1.95 rad = 0.0513 rad = 2.94 deg = 10590 arcsec
        width_arcsec = (width_pc / self.distance_pc) * 206265
        width_pixels = width_arcsec / self.pixel_scale

        # Core separation in pixels
        spacing_arcsec = spacing_true * width_arcsec
        spacing_pixels = spacing_arcsec / self.pixel_scale

        # Create coordinate grid
        y, x = np.mgrid[0:self.map_size, 0:self.map_size]
        center_y, center_x = self.map_size // 2, self.map_size // 2

        # Initialize column density map
        column_density = np.zeros((self.map_size, self.map_size))

        # Base background level (typical HGBS column density)
        background_level = 1e21  # cm^-2

        # Add background
        if background_type == 'flat':
            column_density[:] = background_level
        elif background_type == 'gradient':
            # Linear gradient across map
            gradient = np.linspace(0.5, 1.5, self.map_size)
            column_density[:] = background_level * gradient[np.newaxis, :]
        elif background_type == 'clumpy':
            # Clumpy background from Gaussian blobs
            n_clumps = np.random.poisson(10)
            for _ in range(n_clumps):
                cy = np.random.randint(0, self.map_size)
                cx = np.random.randint(0, self.map_size)
                sigma = np.random.uniform(20, 50)
                amplitude = np.random.uniform(0.1, 0.3) * background_level
                blob = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                column_density += blob
            column_density += background_level

        # Add filament backbone
        # Filament orientation (slight angle to be realistic)
        angle_deg = np.random.uniform(-5, 5)
        angle_rad = np.radians(angle_deg)

        # Create rotated coordinate system along filament
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_rot = (x - center_x) * cos_a + (y - center_y) * sin_a
        y_rot = -(x - center_x) * sin_a + (y - center_y) * cos_a

        # Add Gaussian filament profile
        filament_profile = np.exp(-y_rot**2 / (2 * width_pixels**2))
        column_density += background_level * 0.5 * filament_profile

        # Add cores along filament
        core_positions = []

        # Center the cores on the map
        total_length = (n_cores - 1) * spacing_pixels
        start_x = center_x - total_length / 2

        for i in range(n_cores):
            core_x = start_x + i * spacing_pixels
            core_y = center_y

            # Add core as Gaussian enhancement
            core_sigma = width_pixels / 2.0  # Cores are narrower than filament

            # Rotate core position back to original coordinates
            cos_a_inv, sin_a_inv = np.cos(-angle_rad), np.sin(-angle_rad)
            x_core = (core_x - center_x) * cos_a_inv + (core_y - center_y) * sin_a_inv + center_x
            y_core = -(core_x - center_x) * sin_a_inv + (core_y - center_y) * cos_a_inv + center_y

            # Core profile
            dx = x - x_core
            dy = y - y_core
            core_profile = contrast * background_level * np.exp(-(dx**2 + dy**2) / (2 * core_sigma**2))

            column_density += core_profile

            # Store true core position (in rotated coordinates)
            core_positions.append({
                'x_pixel': x_core,
                'y_pixel': y_core,
                'x_rot': core_x,
                'y_rot': core_y
            })

        # Apply instrumental beam smoothing
        column_density = gaussian_filter(column_density, sigma=self.beam_sigma)

        # Add realistic noise
        # Herschel SPIRE 250 micron noise level ~ 1-2 MJy/sr
        # Convert to column density uncertainty
        noise_rms = noise_level * 0.05 * background_level
        noise = np.random.normal(0, noise_rms, column_density.shape)
        column_density += noise

        # Ensure non-negative
        column_density = np.maximum(column_density, 0)

        # Metadata
        metadata = {
            'spacing_true': spacing_true,  # in units of W
            'spacing_true_pc': spacing_true * width_pc,  # in pc
            'spacing_true_arcsec': spacing_true * width_arcsec,  # in arcsec
            'n_cores': n_cores,
            'contrast': contrast,
            'width_pc': width_pc,
            'noise_level': noise_level,
            'background_type': background_type,
            'core_positions': core_positions,
            'beam_size_fwhm': self.beam_size_fwhm,
            'pixel_scale': self.pixel_scale,
            'distance_pc': self.distance_pc,
            'seed': seed
        }

        return column_density, metadata

    def save_to_hdf5(self, column_density: np.ndarray, metadata: Dict,
                     filename: str):
        """Save synthetic map and metadata to HDF5 file."""
        with h5py.File(filename, 'w') as f:
            f.create_dataset('column_density', data=column_density)

            # Save metadata as attributes
            for key, value in metadata.items():
                if key == 'core_positions':
                    # Save core positions as dataset
                    if len(value) > 0:
                        dt = np.dtype([('x_pixel', 'f8'), ('y_pixel', 'f8'),
                                     ('x_rot', 'f8'), ('y_rot', 'f8')])
                        arr = np.zeros(len(value), dtype=dt)
                        for i, pos in enumerate(value):
                            arr[i] = (pos['x_pixel'], pos['y_pixel'],
                                     pos['x_rot'], pos['y_rot'])
                        f.create_dataset('core_positions', data=arr)
                else:
                    f.attrs[key] = value


def test_generator():
    """Test the filament generator with visualization."""
    import matplotlib.pyplot as plt

    config = {
        'map_size': 256,
        'pixel_scale': 2.0,  # arcsec/pixel
        'distance_pc': 1.95,
        'beam_size_fwhm': 18.0  # arcsec
    }

    generator = SyntheticFilamentGenerator(config)

    # Test different spacings
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    spacings = [1.5, 2.0, 2.5, 3.0, 4.0, 2.0]
    contrasts = [10, 10, 10, 10, 10, 5]
    titles = ['λ=1.5W', 'λ=2.0W', 'λ=2.5W', 'λ=3.0W', 'λ=4.0W', 'λ=2.0W, low contrast']

    for i, (spacing, contrast, title) in enumerate(zip(spacings, contrasts, titles)):
        column_density, metadata = generator.generate_filament(
            spacing_true=spacing,
            n_cores=7,
            contrast=contrast,
            seed=42+i
        )

        im = axes[i].imshow(column_density, origin='lower', cmap='viridis')
        axes[i].set_title(f'{title}\nTrue: {spacing:.1f}W = {metadata["spacing_true_pc"]:.2f} pc')
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig('test_synthetic_filaments.png', dpi=150)
    print("Test figure saved to test_synthetic_filaments.png")


if __name__ == '__main__':
    test_generator()
