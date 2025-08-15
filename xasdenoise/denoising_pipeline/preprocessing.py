"""
Data Preprocessing Module

This module handles the initial data preprocessing steps including:
- Baseline estimation and removal
- Noise estimation
- Data scaling and masking
- Dimension validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple, Dict, Any

from xasdenoise.utils import baseline_estimation
from xasdenoise.denoising_methods import denoising_utils


class DataPreprocessor:
    """
    Handles data preprocessing steps for XAS denoising pipeline.
    
    This class manages baseline removal, noise estimation, and data scaling
    operations that prepare the data for warping and denoising.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.verbose = config.get('verbose', 0)
        
        # State variables
        self.baseline: Optional[np.ndarray] = None
        self.noise: Optional[np.ndarray] = None
        self.original_shape: Optional[Tuple] = None
        
    def process(self, x: np.ndarray, y: np.ndarray, spectrum_obj=None, 
                data_mask: Optional[np.ndarray] = None, 
                noise: Optional[np.ndarray] = None,
                y_weights: Optional[np.ndarray] = None,
                y_reference: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Apply preprocessing steps to the input data.
        
        Args:
            x: Energy array
            y: Spectrum data
            spectrum_obj: Original spectrum object
            data_mask: Optional data mask
            noise: Optional pre-computed noise estimate
            y_weights: Optional data weights
            y_reference: Optional reference data
            
        Returns:
            Dictionary containing processed data and metadata
        """
        if self.verbose > 0:
            print('--------------- Data Preprocessing ---------------')
        
        # Store original shape for restoration
        self.original_shape = y.shape
        
        # Validate and reshape dimensions
        x, y, data_mask, noise, y_weights, y_reference = self._validate_and_reshape(
            x, y, data_mask, noise, y_weights, y_reference
        )
        
        # Remove baseline
        y, baseline, edge_energy = self._remove_baseline(x, y, spectrum_obj)
        self.baseline = baseline
        
        # Estimate noise if not provided
        if noise is None:
            noise = self._estimate_noise(x, y)
        self.noise = noise
        
        # Apply data scaling
        y, noise, y_reference = self._apply_data_scaling(y, noise, y_weights, y_reference)
        
        return {
            'x': x,
            'y': y,
            'noise': noise,
            'baseline': baseline,
            'edge_energy': edge_energy,
            'data_mask': data_mask,
            'y_reference': y_reference,
            'y_weights': y_weights
        }
    
    def _validate_and_reshape(self, x, y, data_mask=None, noise=None, 
                             y_weights=None, y_reference=None):
        """Validate input dimensions and reshape arrays consistently."""
        
        # Validate basic requirements
        if x.ndim != 1:
            raise ValueError("x should be a 1D array")
        
        # Ensure y is 2D
        if y.ndim != 2:
            y = y[:, None]
        
        # Validate and reshape other arrays
        if y_reference is not None and y_reference.ndim != 2:
            y_reference = y_reference[:, None]
        
        if noise is not None and noise.ndim != 2:
            noise = noise[:, None]
        
        if data_mask is not None and data_mask.ndim != 1:
            data_mask = data_mask[:, 0]
        
        # Validate dimensions match
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y dimensions do not match: {x.shape[0]} != {y.shape[0]}")
        
        if y_reference is not None and y_reference.shape[0] != y.shape[0]:
            raise ValueError("y_reference dimensions do not match y")
        
        if noise is not None and noise.shape != y.shape:
            raise ValueError("noise dimensions do not match y")
        
        if data_mask is not None and data_mask.shape[0] != x.shape[0]:
            raise ValueError("data_mask dimensions do not match x")
        
        return x, y, data_mask, noise, y_weights, y_reference
    
    def _remove_baseline(self, x: np.ndarray, y: np.ndarray, spectrum_obj=None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Remove baseline from the data.
        
        Args:
            x: Energy array
            y: Spectrum data
            spectrum_obj: Original spectrum object
            
        Returns:
            Tuple of (processed_y, baseline, edge_energy)
        """
        baseline_method = self.config.get('baseline_removal')
        
        if baseline_method is None:
            # No baseline removal - determine edge energy for other purposes
            edge_energy = self._get_edge_energy(x, y, spectrum_obj)
            baseline = np.zeros_like(y)
            return y, baseline, edge_energy
        
        if self.verbose > 0:
            print(f'Estimating baseline using method: {baseline_method}')
        
        # Estimate baseline
        baseline = self._estimate_baseline(x, y, baseline_method, spectrum_obj)
        
        # Remove baseline
        y_processed = y - baseline
        
        # Determine edge energy from baseline
        edge_energy = self._get_edge_energy_from_baseline(x, baseline)
        
        if self.verbose >= 2:
            self._plot_baseline_results(x, y, baseline, y_processed)
        
        return y_processed, baseline, edge_energy
    
    def _estimate_baseline(self, x: np.ndarray, y: np.ndarray, method: str, spectrum_obj=None) -> np.ndarray:
        """Estimate baseline using specified method."""
        
        if method == 'step':
            # Use time-averaged data for baseline estimation
            y_avg = np.mean(y, axis=1)
            edge = self._get_edge_energy(x, y, spectrum_obj)
            baseline = baseline_estimation.fit_edge_step(x, y_avg, edge)[:, None]
            
        elif method == 'step_not_normalized':
            y_avg = np.mean(y, axis=1)
            edge = self._get_edge_energy(x, y, spectrum_obj)
            pre_edge_idx = self._get_pre_edge_indices(spectrum_obj)
            post_edge_idx = self._get_post_edge_indices(spectrum_obj)
            baseline_funcs = self.config.get('baseline_fitting_funcs', ['1', '1'])
            
            baseline = baseline_estimation.fit_edge_step_not_normalized(
                x, y_avg, edge, pre_edge_idx, post_edge_idx, baseline_funcs
            )[:, None]
            
        elif method == 'poly':
            baseline = baseline_estimation.estimate_polynomial_baseline(x, y)[:, None]
            
        elif method == 'smooth':
            baseline = baseline_estimation.estimate_smooth_baseline(x, y)[:, None]
            
        else:
            raise ValueError(f"Unknown baseline method: {method}")
        
        return baseline
    
    def _get_edge_energy(self, x: np.ndarray, y: np.ndarray, spectrum_obj=None) -> float:
        """Get edge energy from spectrum object or estimate from data."""
        if spectrum_obj is not None and hasattr(spectrum_obj, 'edge') and spectrum_obj.edge is not None:
            return spectrum_obj.edge
        
        # Estimate from data - find maximum derivative
        dx = np.diff(x)
        dy = np.diff(np.mean(y, axis=1))
        edge_idx = np.argmax(dy / dx)
        return x[edge_idx]
    
    def _get_edge_energy_from_baseline(self, x: np.ndarray, baseline: np.ndarray) -> float:
        """Get edge energy from baseline midpoint."""
        baseline_avg = np.mean(baseline, axis=1)
        midpoint_value = (baseline_avg.max() + baseline_avg.min()) / 2
        edge_idx = np.argmin(np.abs(baseline_avg - midpoint_value))
        return x[edge_idx]
    
    def _get_pre_edge_indices(self, spectrum_obj):
        """Get pre-edge region indices."""
        if spectrum_obj is not None and hasattr(spectrum_obj, 'pre_edge_region_indices'):
            return spectrum_obj.pre_edge_region_indices
        return np.array([])
    
    def _get_post_edge_indices(self, spectrum_obj):
        """Get post-edge region indices."""
        if spectrum_obj is not None and hasattr(spectrum_obj, 'post_edge_region_indices'):
            return spectrum_obj.post_edge_region_indices
        return np.array([])
    
    def _estimate_noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Estimate noise in the data.
        
        Args:
            x: Energy array
            y: Spectrum data
            
        Returns:
            Estimated noise array
        """
        if self.verbose > 0:
            print('Estimating noise level')
        
        mode = self.config.get('noise_estimation_mode', 1)
        
        if mode == 0:
            # Constant noise from specific region
            crop_region = self.config.get('noise_crop_region', np.arange(0, 100))
            noise_val = denoising_utils.estimate_noise(
                x[crop_region], y[crop_region, :]
            )
            noise_estimate = np.ones_like(y) * noise_val
            
        elif mode == 1:
            # Varying noise using sliding window
            noise_window = self.config.get('noise_window', 3)
            win_points = self._ev_to_points(x, noise_window)
            
            noise_estimate = denoising_utils.estimate_noise_std_sliding_window(x, y, win_points)
            
            # Apply smoothing if requested
            if self.config.get('noise_estimation_smoothing', True):
                smooth_win = len(x) // 5
                noise_estimate = uniform_filter1d(
                    noise_estimate**2, smooth_win, mode="nearest", axis=0
                )**0.5
        else:
            raise ValueError(f"Unknown noise estimation mode: {mode}")
        
        if self.verbose >= 2:
            self._plot_noise_results(x, noise_estimate)
        
        return noise_estimate
    
    def _apply_data_scaling(self, y: np.ndarray, noise: np.ndarray, 
                           y_weights: Optional[np.ndarray], 
                           y_reference: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply data scaling using weights."""
        
        if y_weights is None:
            return y, noise, y_reference
        
        if self.verbose > 0:
            print('Applying data scaling')
        
        # Scale data and noise
        y_scaled = y * y_weights
        noise_scaled = noise * y_weights
        
        # Scale reference if provided
        y_reference_scaled = None
        if y_reference is not None:
            y_reference_scaled = y_reference * y_weights
        
        return y_scaled, noise_scaled, y_reference_scaled
    
    def _ev_to_points(self, x: np.ndarray, win_eV: float) -> int:
        """Convert energy window to number of points."""
        win = np.argmin(np.cumsum(abs(np.diff(x))) <= win_eV)
        win = win // 2 * 2 + 1  # Make odd
        win = np.maximum(win, 3)  # At least 3 points
        return win
    
    def _plot_baseline_results(self, x: np.ndarray, y: np.ndarray, 
                              baseline: np.ndarray, y_processed: np.ndarray):
        """Plot baseline estimation results."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, y[:, 0], '*', label='Original Signal', alpha=0.7)
        plt.plot(x, baseline[:, 0], label='Baseline', linewidth=2)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Absorption")
        plt.legend()
        plt.title("Baseline Estimation")
        
        plt.subplot(1, 2, 2)
        plt.plot(x, y_processed[:, 0], label='Signal - Baseline')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Absorption")
        plt.legend()
        plt.title("After Baseline Removal")
        
        plt.tight_layout()
        plt.show()
    
    def _plot_noise_results(self, x: np.ndarray, noise: np.ndarray):
        """Plot noise estimation results."""
        plt.figure()
        plt.plot(x, noise[:, 0], label='Estimated Noise (σ)')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Noise Level')
        plt.title('Noise Estimation')
        plt.legend()
        plt.show()
    
    def restore_baseline_and_scaling(self, y_denoised: np.ndarray, 
                                   y_error: np.ndarray, 
                                   y_noise: np.ndarray,
                                   y_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Restore baseline and undo scaling in post-processing.
        
        Args:
            y_denoised: Denoised data
            y_error: Error estimates
            y_noise: Noise estimates
            y_weights: Data weights used in scaling
            
        Returns:
            Tuple of (restored_y, restored_error, restored_noise)
        """
        # Undo data scaling
        if y_weights is not None:
            y_denoised = y_denoised / (y_weights + 1e-6)
            y_error = y_error / (y_weights + 1e-6)
            y_noise = y_noise / (y_weights + 1e-6)
        
        # Add back baseline
        if self.baseline is not None:
            y_denoised = y_denoised + self.baseline
        
        return y_denoised, y_error, y_noise
