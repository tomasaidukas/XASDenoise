"""
Data Warping Module

This module performs input warping as well as signal interpolation onto a uniform grid.
In doing so both (x,y) data pairs can be transformed into a warped domain.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Optional, Tuple, Dict, Any, Union
from scipy.ndimage import uniform_filter1d

from xasdenoise.utils.smoothness import estimate_smoothness as smoothness_estimation
from xasdenoise.denoising_methods import denoising_utils
from .input_warping import InputWarper
from .interpolation import DataInterpolator
from .smoothness_estimation import SmoothnessEstimator


class DataWarper:
    """
    Handles data warping operations for XAS denoising pipeline.

    This class manages input warping and interpolation operations 
    that transform data for more effective denoising.
    Uses modular InputWarper, DataInterpolator, SmoothnessEstimator classes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data warper.
        
        Args:
            config: Configuration dictionary containing warping parameters
        """
        self.config = config
        self.verbose = config.get('verbose', 0)
        
        # Initialize modular components
        self.input_warper = InputWarper(
            method=config.get('input_warping_method'),
            verbose=self.verbose
        )
        
        self.interpolator = DataInterpolator(
            method=config.get('warping_interpolation_method'),
            interpolation_kind=config.get('output_warping_interpolation', 'linear'),
            num_points=config.get('warping_interpolation_num_points'),
            verbose=self.verbose
        )
        
        self.smoothness_estimator = SmoothnessEstimator(config)
        
        # State variables for unwarping
        self.original_data = {}
        self.smoothness = None
        
    def process(self, x: np.ndarray, y: np.ndarray, noise: np.ndarray,
                edge_energy: float, data_mask: Optional[np.ndarray] = None,
                y_reference: Optional[np.ndarray] = None,
                denoiser=None) -> Dict[str, Union[np.ndarray, Any]]:
        """
        Apply warping operations to the input data.
        
        Args:
            x: Energy array
            y: Spectrum data
            noise: Noise estimates
            edge_energy: Edge energy for warping
            data_mask: Optional data mask
            y_reference: Optional reference data
            denoiser: Denoiser object (needed for smoothness estimation)
            
        Returns:
            Dictionary containing warped data and metadata
        """
        if self.verbose > 0:
            print('--------------- Data Warping ---------------')
        
        # Store original data for unwarping
        self._store_original_data(x, y, noise, data_mask, y_reference)
        
        # Get warping energy
        warping_energy = self.config.get('warping_energy')
        if warping_energy is None:
            warping_energy = edge_energy
        
        # Estimate smoothness if needed
        smoothness = self.smoothness_estimator.estimate_smoothness(x, y, noise, warping_energy, denoiser)
        self.smoothness = smoothness
        
        # Apply warping if requested
        if self.config.get('input_warping_method') is not None:
            x_warped, y_warped, noise_warped, smoothness_warped, data_mask_warped, y_reference_warped = self.apply_input_warping(
                x, y, noise, warping_energy, smoothness, data_mask, y_reference
            )
        else:
            x_warped, y_warped = x, y
            noise_warped = noise
            data_mask_warped = data_mask
            y_reference_warped = y_reference
            smoothness_warped = smoothness
            
        return {
            'x': x_warped,
            'y': y_warped,
            'noise': noise_warped,
            'smoothness': smoothness_warped,
            'data_mask': data_mask_warped,
            'y_reference': y_reference_warped,
            'warping_energy': warping_energy
        }
    
    def apply_input_warping(self, x: np.ndarray, y: np.ndarray, noise: np.ndarray,
                      warping_energy: float, smoothness: Optional[np.ndarray],
                      data_mask: Optional[np.ndarray], 
                      y_reference: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Apply warping transformation to the data using modular components."""
        
        if self.verbose > 0:
            print(f"Applying {self.config.get('input_warping_method')} warping")
        
        # Step 1: Initialize and apply input warping (coordinate transformation)
        self.input_warper.initialize(x, warping_energy, smoothness)
        x_warped = self.input_warper.get_warped_coordinates()
        
        # Visualize warping results if verbosity is high enough
        if self.verbose >= 2:
            self.visualize_warping(x, y, x_warped, y, 
                                 title=f"Stationarity Warping: {self.config.get('input_warping_method')}")
        
        # Step 2: Initialize interpolation onto uniform grid (if needed)
        self.interpolator.initialize(x_warped)
        x_uniform = self.interpolator.get_uniform_grid()
        
        # Step 3: Interpolate all data arrays onto uniform grid
        y_warped = self.interpolator.interpolate(y)
        
        # Interpolate auxiliary data with reduced verbosity
        original_verbose = self.interpolator.verbose
        self.interpolator.verbose = 0
        
        noise_warped = self.interpolator.interpolate(noise)
        smoothness_warped = self.interpolator.interpolate(smoothness)
        data_mask_warped = self.interpolator.interpolate(data_mask)
        y_reference_warped = self.interpolator.interpolate(y_reference)
        
        self.interpolator.verbose = original_verbose
        
        if self.verbose > 0:
            print(f"Warped grid: {len(x)} -> {len(x_uniform)} points")
        
        return x_uniform, y_warped, noise_warped, smoothness_warped, data_mask_warped, y_reference_warped
    
    def undo_warping(self, y: np.ndarray, y_error: Optional[np.ndarray] = None,
                   noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """
        Unwarp denoised data back to original domain using modular components.
        
        Args:
            y: Denoised data in warped domain
            y_error: Error estimates in warped domain
            y_noise: Noise estimates in warped domain
            
        Returns:
            Tuple of unwarped data
        """
        if self.config.get('input_warping_method') is None:
            # No warping was applied
            x_original = self.original_data['x']
            return x_original, y, y_error, noise

        if self.verbose > 0:
            print("Undoing warping transformation")

        # Step 1: Uninterpolate from uniform grid back to warped coordinates
        y_warped = self.interpolator.uninterpolate(y)
        
        # Uninterpolate auxiliary data with reduced verbosity
        original_verbose = self.interpolator.verbose
        self.interpolator.verbose = 0
        
        y_error_warped = None
        noise_warped = None
        
        if y_error is not None:
            y_error_warped = self.interpolator.uninterpolate(y_error)
        if noise is not None:
            noise_warped = self.interpolator.uninterpolate(noise)

        self.interpolator.verbose = original_verbose
        
        # Step 2: Get original coordinates (no transformation needed for coordinates)
        x_original = self.input_warper.get_original_coordinates()
        
        # For data, we simply return the uninterpolated data mapped to original coordinates
        # since the input warping only transforms coordinates, not the data values
        
        # Visualize unwarping results if verbosity is high enough
        if self.verbose >= 2:
            # Get the uniform grid for comparison
            x_uniform = self.interpolator.get_uniform_grid()
            self.visualize_warping(x_uniform, y, x_original, y_warped, 
                                 title=f"Unwarping Results: {self.config.get('input_warping_method')}")
        
        return x_original, y_warped, y_error_warped, noise_warped
        
    def _store_original_data(self, x: np.ndarray, y: np.ndarray, noise: np.ndarray,
                            data_mask: Optional[np.ndarray], y_reference: Optional[np.ndarray]):
        """Store original data for unwarping operations."""
        self.original_data = {
            'x': x.copy(),
            'y': y.copy(),
            'noise': noise.copy() if noise is not None else None,
            'data_mask': data_mask.copy() if data_mask is not None else None,
            'y_reference': y_reference.copy() if y_reference is not None else None
        }
    
    def visualize_warping(self, x1, y1, x2, y2, title="Warping Results"):
        """
        Visualize warping results with dual plots.
        
        Args:
            x1 (array): First domain x-coordinates.
            y1 (array): First domain y-coordinates.
            x2 (array): Second domain x-coordinates.
            y2 (array): Second domain y-coordinates.
            title (str): Plot title.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First plot: Original domain data
        try:
            ax1.plot(x1, y1[:, 0], 'r-', label='Original domain data')
        except:
            ax1.plot(x1, y1, 'r-', label='Original domain data')
        ax1.set_title('Before Warping')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        
        # Second plot: Warped domain data
        try:
            ax2.plot(x2, y2[:, 0], 'g-', label='Warped domain data')
        except:
            ax2.plot(x2, y2, 'g-', label='Warped domain data')
        ax2.set_title('After Warping')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')   
        ax2.legend()     
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
