"""
Pipeline Configuration

This module defines the configuration class for the denoising pipeline.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any


@dataclass
class PipelineConfig:
    """
    Configuration class for the XAS denoising pipeline.
    
    This class encapsulates all parameters needed for the various stages
    of the denoising pipeline, organized by processing stage.
    """
    
    # ============ Data Preprocessing Parameters ============
    # Baseline removal settings
    data_baseline_removal: Optional[str] = None  # None, 'step', 'step_not_normalized', 'poly', 'smooth'
    baseline_fitting_funcs: List[str] = field(default_factory=lambda: ['1', '1'])  # fitting functions for pre/post edge when using data_baseline_removal='step_not_normalized'
    baseline_step_func: str = 'tanh'  # 'tanh', 'asymmetric_tanh' # step function when using data_baseline_removal='step'
    
    # Noise estimation settings
    noise_estimation_mode: int = 1
    noise_crop_region: np.ndarray = field(default_factory=lambda: np.arange(0, 100))  # eV
    noise_window: float = 3  # eV
    noise_estimation_smoothing: bool = True
    noise_polyfit_degree: int = 1  # Degree of polynomial for noise estimation within each sliding window
    
    # Data scaling settings
    apply_data_mask: Union[str, bool] = False
    
    # ============ Warping Parameters ============
    # Input warping method
    input_warping_method: Optional[str] = None  # None, 'kspace', 'smoothness', 'kspace_exafs_smoothness_xanes'
    warping_energy: Optional[float] = None  # energy to use for warping
    
    # Interpolation settings
    output_warping_interpolation: Optional[str] = 'linear'  # 'linear', 'nearest', 'cubic'
    warping_interpolation_method: Optional[str] = 'downsample'  # None, 'downsample', 'upsample', 'same'
    warping_interpolation_num_points: Optional[int] = None  # override interpolation points
    
    # ============ Smoothness Estimation Parameters ============
    smoothness_est_iter: int = 5
    smoothness_polyfit_window0: float = 3  # eV
    smoothness_polyfit_window: float = 0.1  # eV
    smoothness_smooth_window0: float = 11  # eV
    smoothness_smooth_window: float = 11  # eV
    
    smoothness_denoise: bool = True
    smoothness_denoise_window: float = 1  # eV
    smoothness_downsampling_pts: int = 1000
    smoothness_downsampling_method: Optional[str] = None  # 'uniform', 'smoothness', 'kspace'
    smoothness_use_simple_denoiser: bool = False # instead of the current denoiser method (e.g., GP) use a simpler moving average approach
    
    # ============ Denoising Parameters ============
    # Downsampling settings
    downsampling_pts: int = 1000
    data_downsampling_method: Optional[str] = None  # None, 'uniform', 'smoothness'
    
    # Region splitting
    split_xanes_exafs: bool = False
    split_xanes_exafs_energy: Optional[float] = None
    
    # ============ General Parameters ============
    verbose: int = 0
    values_in_ev: bool = True  # if values for windows are in eV, then convert to points (if false, then assume values are in number of points)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters."""
        # Validate baseline removal method
        valid_baseline_methods = [None, 'step', 'poly', 'smooth', 'step_not_normalized']
        if self.data_baseline_removal not in valid_baseline_methods:
            raise ValueError(f"data_baseline_removal must be one of {valid_baseline_methods}")
        
        # Validate warping method
        valid_warping_methods = [None, 'kspace', 'smoothness', 'kspace_exafs_smoothness_xanes', 'log']
        if self.input_warping_method not in valid_warping_methods:
            raise ValueError(f"input_warping_method must be one of {valid_warping_methods}")
        
        # Validate interpolation method
        valid_interp_methods = [None, 'downsample', 'upsample', 'same']
        if self.warping_interpolation_method not in valid_interp_methods:
            raise ValueError(f"warping_interpolation_method must be one of {valid_interp_methods}")
        
        # Validate numeric parameters
        if self.noise_window <= 0:
            raise ValueError("noise_window must be positive")
        
        if self.smoothness_est_iter < 0:
            raise ValueError("smoothness_est_iter must be non-negative")
        
        if self.downsampling_pts <= 0:
            raise ValueError("downsampling_pts must be positive")
    
    def copy(self):
        """Create a deep copy of the configuration."""
        import copy
        return copy.deepcopy(self)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Re-validate after updates
        self._validate_parameters()
    
    def get_preprocessing_config(self):
        """Get configuration subset for preprocessing stage."""
        return {
            'baseline_removal': self.data_baseline_removal,
            'baseline_fitting_funcs': self.baseline_fitting_funcs,
            'baseline_step_func': self.baseline_step_func,
            'noise_estimation_mode': self.noise_estimation_mode,
            'noise_crop_region': self.noise_crop_region,
            'noise_window': self.noise_window,
            'noise_estimation_smoothing': self.noise_estimation_smoothing,
            'apply_data_mask': self.apply_data_mask,
            'values_in_ev': self.values_in_ev,
            'verbose': self.verbose
        }
    
    def get_warping_config(self):
        """Get configuration subset for warping stage."""
        return {
            'input_warping_method': self.input_warping_method,
            'warping_energy': self.warping_energy,
            'output_warping_interpolation': self.output_warping_interpolation,
            'warping_interpolation_method': self.warping_interpolation_method,
            'warping_interpolation_num_points': self.warping_interpolation_num_points,
            'smoothness_est_iter': self.smoothness_est_iter,
            'smoothness_polyfit_window0': self.smoothness_polyfit_window0,
            'smoothness_polyfit_window': self.smoothness_polyfit_window,
            'smoothness_smooth_window0': self.smoothness_smooth_window0,
            'smoothness_smooth_window': self.smoothness_smooth_window,
            'smoothness_denoise': self.smoothness_denoise,
            'smoothness_denoise_window': self.smoothness_denoise_window,
            'smoothness_downsampling_pts': self.smoothness_downsampling_pts,
            'smoothness_downsampling_method': self.smoothness_downsampling_method,
            'smoothness_use_simple_denoiser': self.smoothness_use_simple_denoiser,
            'data_downsampling_method': self.data_downsampling_method,
            'values_in_ev': self.values_in_ev,
            'verbose': self.verbose
        }
    
    def get_denoising_config(self):
        """Get configuration subset for denoising stage."""
        return {
            'downsampling_pts': self.downsampling_pts,
            'data_downsampling_method': self.data_downsampling_method,
            'split_xanes_exafs': self.split_xanes_exafs,
            'split_xanes_exafs_energy': self.split_xanes_exafs_energy,
            'apply_data_mask': self.apply_data_mask,
            'verbose': self.verbose
        }

    def get_postprocessing_config(self):
        """Get configuration subset for postprocessing stage."""
        return {
            'input_warping_method': self.input_warping_method,
            'warping_energy': self.warping_energy,
            'output_warping_interpolation': self.output_warping_interpolation,
            'warping_interpolation_method': self.warping_interpolation_method,
            'warping_interpolation_num_points': self.warping_interpolation_num_points,
            'baseline_removal': self.data_baseline_removal,
            'baseline_fitting_funcs': self.baseline_fitting_funcs,
            'noise_estimation_mode': self.noise_estimation_mode,
            'noise_crop_region': self.noise_crop_region,
            'noise_window': self.noise_window,
            'noise_estimation_smoothing': self.noise_estimation_smoothing,
            'apply_data_mask': self.apply_data_mask,
            'verbose': self.verbose
        }