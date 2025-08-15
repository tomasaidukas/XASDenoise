"""
Smoothness Estimation Module

This module handles smoothness estimation of XAS signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Union
from scipy.ndimage import uniform_filter1d

from xasdenoise.utils.smoothness import estimate_smoothness as smoothness_estimation
from xasdenoise.denoising_methods import denoising_utils


class SmoothnessEstimator:
    """
    Handles smoothness estimation operations for XAS denoising pipeline.
    
    This class provides various methods for estimating the smoothness of spectral data,
    including k-space methods and iterative refinement using denoising algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the smoothness estimator.
        
        Args:
            config: Configuration dictionary containing smoothness estimation parameters
        """
        self.config = config
        self.verbose = config.get('verbose', 0)
        
    def estimate_smoothness(self, x: np.ndarray, y: np.ndarray, noise: np.ndarray,
                           warping_energy: float, denoiser=None) -> Optional[np.ndarray]:
        """
        Estimate data smoothness using various methods.
        
        Args:
            x: Energy array
            y: Spectrum data  
            noise: Noise estimates
            warping_energy: Energy for warping operations
            denoiser: Denoiser object for iterative estimation
            
        Returns:
            Smoothness array or None if not needed
        """
        if not self.is_smoothness_needed():
            return None
        
        if self.verbose > 0:
            print('Estimating data smoothness')
        
        # Use time-averaged data for smoothness estimation
        y_avg = np.mean(y, axis=1)[:, None]
        
        # Choose estimation method
        downsampling_method = self.config.get('smoothness_downsampling_method')
        if downsampling_method is None:
            downsampling_method = self.config.get('data_downsampling_method')
        if downsampling_method == 'kspace':
            return self.estimate_kspace_smoothness(x, warping_energy)
        else:
            return self.estimate_iterative_smoothness(x, y_avg, noise, warping_energy, denoiser)
    
    def is_smoothness_needed(self) -> bool:
        """Check if smoothness estimation is required."""
        smoothness_methods = ['smoothness', 'kspace_exafs_smoothness_xanes']
        downsampling_method = self.config.get('smoothness_downsampling_method')
        if downsampling_method is None:
            downsampling_method = self.config.get('data_downsampling_method')
        warping_method = self.config.get('input_warping_method')
        
        return (downsampling_method in smoothness_methods or 
                warping_method in smoothness_methods)
    
    def estimate_kspace_smoothness(self, x: np.ndarray, edge_energy: float) -> np.ndarray:
        """Estimate smoothness using k-space factors."""
        smoothness = denoising_utils.kspace_downsampling_factors(x, edge_energy)
        
        if self.verbose >= 2:
            self.plot_smoothness_results(x, smoothness, 'K-space smoothness estimation')
        
        return smoothness[:, None]
    
    def estimate_iterative_smoothness(self, x: np.ndarray, y_avg: np.ndarray, 
                                     noise: np.ndarray, edge_energy: float, 
                                     denoiser=None) -> np.ndarray:
        """Estimate smoothness using iterative refinement."""
        
        if denoiser is None:
            raise ValueError("Denoiser is required for iterative smoothness estimation")
        
        # Initial preprocessing
        y_preprocessed = self.preprocess_for_smoothness(x, y_avg)
        
        # Get window sizes
        windows = self.get_smoothness_windows(x)
        
        # Initial smoothness estimation
        _, smoothness = smoothness_estimation(
            x, y_preprocessed, windows['initial_polyfit'], windows['initial_smooth']
        )
        
        # Iterative refinement
        num_iterations = self.config.get('smoothness_est_iter', 5)
        for iteration in range(num_iterations):
            if self.verbose >= 1:
                print(f"Smoothness refinement iteration {iteration + 1}/{num_iterations}")
            
            smoothness = self.refine_smoothness_iteration(
                x, y_avg, noise, smoothness, edge_energy, windows, denoiser
            )
        
        if self.verbose >= 2:
            self.plot_smoothness_results(x, smoothness, 'Iterative smoothness estimation')
        
        return smoothness[:, None]
    
    def preprocess_for_smoothness(self, x: np.ndarray, y_avg: np.ndarray) -> np.ndarray:
        """Preprocess data for smoothness estimation."""
        
        if not self.config.get('smoothness_denoise', True):
            return y_avg
        
        denoise_window = self.config.get('smoothness_denoise_window', 1)
        win_points = self.ev_to_points(x, denoise_window)
        y_denoised = uniform_filter1d(y_avg, win_points, axis=0)
        
        if self.verbose >= 2:
            plt.figure()
            plt.title("Smoothness: initial data denoising")
            plt.plot(x, y_avg[:, 0], label='initial')
            plt.plot(x, y_denoised[:, 0], label='denoised')
            plt.legend()
            plt.show()
        
        return y_denoised
    
    def get_smoothness_windows(self, x: np.ndarray) -> Dict[str, int]:
        """Get window sizes for smoothness estimation in points."""
        return {
            'initial_polyfit': self.ev_to_points(x, self.config.get('smoothness_polyfit_window0', 3)),
            'initial_smooth': self.ev_to_points(x, self.config.get('smoothness_smooth_window0', 11)),
            'refined_polyfit': self.ev_to_points(x, self.config.get('smoothness_polyfit_window', 0.1)),
            'refined_smooth': self.ev_to_points(x, self.config.get('smoothness_smooth_window', 11))
        }
    
    def refine_smoothness_iteration(self, x: np.ndarray, y_avg: np.ndarray, 
                                   noise: np.ndarray, current_smoothness: np.ndarray,
                                   edge_energy: float, windows: Dict[str, int], 
                                   denoiser) -> np.ndarray:
        """Perform one iteration of smoothness refinement."""
        
        # Setup denoiser state
        denoiser_state = self.setup_denoiser_for_iteration(denoiser)
        
        try:
            # Create temporary warper for iteration
            temp_warper = self.create_temp_warper_for_iteration()
            
            # Process data through warping and denoising
            y_denoised = self.process_temp_iteration(
                temp_warper, x, y_avg, noise, current_smoothness, edge_energy, denoiser
            )
            
            # Estimate new smoothness from denoised data
            _, new_smoothness = smoothness_estimation(
                x, y_denoised, windows['refined_polyfit'], windows['refined_smooth']
            )
            return new_smoothness
            
        finally:
            # Restore denoiser state
            self.restore_denoiser_state(denoiser, denoiser_state)
    
    def setup_denoiser_for_iteration(self, denoiser):
        """Setup denoiser for smoothness iteration."""
        try:
            state = denoiser.save_state() if hasattr(denoiser, 'save_state') else None
            
            # Configure denoiser for iteration
            if hasattr(denoiser, 'refine_noise_estimate'):
                denoiser.refine_noise_estimate = False
            if hasattr(denoiser, 'refine_downsampled_estimate'):
                denoiser.refine_downsampled_estimate = False
            if hasattr(denoiser, 'optimize_params_find_tv_lambda'):
                denoiser.optimize_params_find_tv_lambda = False
            if hasattr(denoiser, 'verbose'):
                denoiser.verbose = 0
                
            return state
        except:
            return None
    
    def restore_denoiser_state(self, denoiser, state):
        """Restore denoiser state after iteration."""
        if state is not None and hasattr(denoiser, 'restore_state'):
            try:
                denoiser.restore_state(state)
            except:
                pass
    
    def create_temp_warper_for_iteration(self):
        """Create temporary warper configuration for iteration."""
        # Import here to avoid circular imports
        from .stationarity_warping import DataWarper
        
        temp_config = self.config.copy()
        
        # Modify configuration for iteration
        if temp_config.get('warping_interpolation_method') in ['upsample', 'same']:
            temp_config['warping_interpolation_method'] = 'downsample'
        
        return DataWarper(temp_config)
    
    def process_temp_iteration(self, temp_warper, x: np.ndarray, y_avg: np.ndarray,
                               noise: np.ndarray, smoothness: np.ndarray, 
                               edge_energy: float, denoiser) -> np.ndarray:
        """Process data through temporary warping and denoising."""
        
        # Setup temporary data
        noise_avg = np.mean(noise, axis=1)[:, None]
        
        # Setup data mask for iteration
        data_mask = self.get_smoothness_mask(x, edge_energy)
        
        # Store temp data and apply warping
        temp_warper._store_original_data(x, y_avg, noise_avg, data_mask, None)
        temp_warper.smoothness = smoothness.copy()
        
        # Apply warping
        if self.config.get('input_warping_method') is not None:
            x_warped, y_warped, noise_warped, smoothness_warped, data_mask_warped, _ = temp_warper.apply_input_warping(
                x, y_avg, noise_avg, edge_energy, smoothness, data_mask, None
            )
        else:
            x_warped, y_warped, noise_warped, smoothness_warped, data_mask_warped = x, y_avg, noise_avg, smoothness, data_mask

        # Apply mask if needed
        if data_mask_warped is not None and not np.all(data_mask_warped):
            x_masked = x_warped[data_mask_warped]
            y_masked = y_warped[data_mask_warped, :]
            noise_masked = noise_warped[data_mask_warped, :]
            smoothness_masked = smoothness_warped[data_mask_warped] if smoothness is not None else None
        else:
            x_masked, y_masked, noise_masked = x_warped, y_warped, noise_warped
            smoothness_masked = smoothness_warped
        
        # Initialize denoiser and denoise
        denoiser.initialize_denoiser(
            x=x_masked, y=y_masked, noise=noise_masked,
            x_predict=x_warped, noise_predict=noise_warped
        )
        
        downsampling_method = self.config.get('smoothness_downsampling_method')
        downsampling_pts=self.config.get('smoothness_downsampling_pts')
        if downsampling_method is None:
            downsampling_pts=self.config.get('downsampling_pts', 1000)
            downsampling_method = self.config.get('data_downsampling_method')
        y_denoised, _, _ = denoiser.denoise_with_downsampling(
            downsampling_pts=self.config.get('smoothness_downsampling_pts', 1000),
            downsampling_method=downsampling_method,
            smoothness=smoothness_masked
        )
        
        # Handle masked regions
        if (self.config.get('input_warping_method') == 'kspace_exafs_smoothness_xanes' and 
            data_mask_warped is not None):
            y_denoised[~data_mask_warped] = 0
        
        # Unwarp if needed
        if self.config.get('input_warping_method') is not None:
            _, y_denoised, _, _ = temp_warper.undo_warping(y_denoised)

        return y_denoised
    
    def get_smoothness_mask(self, x: np.ndarray, edge_energy: float) -> Optional[np.ndarray]:
        """Get data mask for smoothness estimation."""
        
        if self.config.get('input_warping_method') == 'kspace_exafs_smoothness_xanes':
            # Mask out EXAFS region for XANES-only smoothness estimation
            mask = np.ones_like(x, dtype=bool)
            edge_idx = np.argmin(np.abs(x - edge_energy))
            mask[edge_idx:] = False
            return mask
        
        return np.ones_like(x, dtype=bool)
    
    def ev_to_points(self, x: np.ndarray, win_eV: float) -> int:
        """Convert energy window to number of points."""
        win = np.argmin(np.cumsum(abs(np.diff(x))) <= win_eV)
        win = win // 2 * 2 + 1  # Make odd
        win = np.maximum(win, 3)  # At least 3 points
        return win
    
    def plot_smoothness_results(self, x: np.ndarray, smoothness: np.ndarray, title: str):
        """Plot smoothness estimation results."""
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, smoothness, label='Smoothness')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Smoothness')
        plt.title(title)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        downsampling_factors = denoising_utils.get_downsampling_factors(smoothness)
        plt.plot(x, downsampling_factors)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Downsampling Probability')
        plt.title('Downsampling Factors')
        
        plt.tight_layout()
        plt.show()
