"""
Main Denoising Pipeline

This module provides the main DenoisingPipeline class that calls
all the modular processing steps for XAS data denoising.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Optional, Tuple, Dict, Any, Union

from .config import PipelineConfig
from .preprocessing import DataPreprocessor
from .stationarity_warping import DataWarper
from .denoising import DenoisingProcessor
from .postprocessing import DataPostprocessor
from xasdenoise.denoising_methods.denoisers import RegularDenoiser
from xasdenoise.utils import baseline_estimation
from xasdenoise.denoising_methods import denoising_utils


class DenoisingPipeline:
    """
    Main pipeline class for XAS data denoising.
    
    This class calls the complete denoising workflow:
    1. Data preprocessing (baseline removal, noise estimation)
    2. Data warping (smoothness estimation, input warping, interpolation onto uniform grids)
    3. Denoising (various methods with downsampling/masking)
    4. Post-processing (unwarping, baseline addition)
    
    Example:
        config = PipelineConfig(data_baseline_removal='step')
        pipeline = DenoisingPipeline(config)
        result = pipeline.process(spectrum, denoiser)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, denoiser=None):
        """
        Initialize the denoising pipeline.
        
        Args:
            config: Pipeline configuration object
            denoiser: Denoiser object to use (if None, defaults to GPDenoiser)
        """
        self.config = config or PipelineConfig()
        self.denoiser = denoiser
        
        # Initialize processing modules
        self.preprocessor = DataPreprocessor(self.config.get_preprocessing_config())
        self.warper = DataWarper(self.config.get_warping_config())
        self.denoising_processor = DenoisingProcessor(self.config.get_denoising_config())
        self.postprocessor = DataPostprocessor(self.config.get_postprocessing_config())

        # State variables for tracking data through pipeline
        self.spectrum_obj = None
        self.original_energy = None
        self.original_spectrum = None
        self.processing_metadata = {}
        
        # Results
        self.results = None
    
    def load_data(self, spectrum_obj, data_mask: Optional[np.ndarray] = None, 
                  noise: Optional[np.ndarray] = None, y_weights: Optional[np.ndarray] = None,
                  y_reference: Optional[np.ndarray] = None):
        """
        Load data from a Spectrum object.
        
        Args:
            spectrum_obj: Spectrum object containing energy and spectrum data
            data_mask: Optional data mask for selective processing
            noise: Optional pre-computed noise estimates
            y_weights: Optional data weights for scaling
            y_reference: Optional reference data for comparison
            
        Returns:
            Self for method chaining
        """
        self.spectrum_obj = spectrum_obj
        self.original_energy = spectrum_obj.energy.copy()
        self.original_spectrum = spectrum_obj.spectrum.copy()
        
        # Store additional data
        self.processing_metadata = {
            'data_mask': data_mask,
            'noise': noise,
            'y_weights': y_weights,
            'y_reference': y_reference
        }
        
        return self
    
    def process(self, denoiser=None) -> 'DenoisingPipeline':
        """
        Run the complete denoising pipeline.
        
        Args:
            denoiser: Optional denoiser to use (overrides constructor)
            
        Returns:
            Self for method chaining
        """
        # Set denoiser
        if denoiser is not None:
            self.denoiser = denoiser
        elif self.denoiser is None:
            self.denoiser = RegularDenoiser()
        
        # Validate that data has been loaded
        if self.spectrum_obj is None:
            raise ValueError("Data must be loaded before processing. Call load_data() first.")
        
        if self.config.verbose > 0:
            print('============= Starting XAS Denoising Pipeline =============')
        
        try:
            # Step 1: Preprocessing
            preprocessed_data = self._run_preprocessing()
            
            # Step 2: Warping
            warped_data = self._run_warping(preprocessed_data)
            
            # Step 3: Denoising
            denoised_data = self._run_denoising(warped_data, preprocessed_data)
            
            # Step 4: Post-processing
            final_results = self._run_postprocessing(denoised_data)
            
            # Store results
            self.results = final_results
            
            if self.config.verbose > 0:
                print('============= Pipeline Complete =============')
            
        except Exception as e:
            if self.config.verbose > 0:
                print(f'Pipeline failed: {str(e)}')
            raise
        
        return self
    
    def _run_preprocessing(self) -> Dict[str, Any]:
        """Run the preprocessing step."""
        
        return self.preprocessor.process(
            x=self.original_energy,
            y=self.original_spectrum,
            spectrum_obj=self.spectrum_obj,
            data_mask=self.processing_metadata.get('data_mask'),
            noise=self.processing_metadata.get('noise'),
            y_weights=self.processing_metadata.get('y_weights'),
            y_reference=self.processing_metadata.get('y_reference')
        )
    
    def _run_warping(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the warping step."""
        
        return self.warper.process(
            x=preprocessed_data['x'],
            y=preprocessed_data['y'],
            noise=preprocessed_data['noise'],
            edge_energy=preprocessed_data['edge_energy'],
            data_mask=preprocessed_data['data_mask'],
            y_reference=preprocessed_data['y_reference'],
            denoiser=self.denoiser
        )
    
    def _run_denoising(self, warped_data: Dict[str, Any], 
                      preprocessed_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the denoising step."""
        
        return self.denoising_processor.process(
            x=warped_data['x'],
            y=warped_data['y'],
            noise=warped_data['noise'],
            denoiser=self.denoiser,
            smoothness=warped_data['smoothness'],
            data_mask=warped_data['data_mask'],
            y_reference=warped_data['y_reference'],
            edge_energy=preprocessed_data['edge_energy'],
            x_original=preprocessed_data['x']
        )
    
        
    def _run_postprocessing(self, denoised_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run the post-processing step."""
        
        y_denoised, y_error, y_noise = denoised_data
        
        # Apply post-processing
        y_final, y_error_final, y_noise_final = self.postprocessor.process(
            y_denoised=y_denoised,
            y_error=y_error,
            y_noise=y_noise,
            preprocessor=self.preprocessor,
            warper=self.warper,
            y_weights=self.processing_metadata.get('y_weights')
        )
        
        # Package results
        return self.postprocessor.package_results(
            y_denoised=y_final,
            y_error=y_error_final,
            y_noise=y_noise_final,
            x_final=self.original_energy
        )
    
    def get_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the denoising results.
        
        Returns:
            Tuple of (denoised_spectrum, error_estimates, noise_estimates)
        """
        if self.results is None:
            raise ValueError("Processing must be completed before accessing results. Call process() first.")
        
        return self.results['denoised'], self.results['error'], self.results['noise']
    
    def plot_results(self, fig_size: Tuple[int, int] = (12, 8), title: Optional[str] = None, 
                     full_range: bool = True, plot_kspace: bool = False, 
                     time_instance: Optional[int] = None, show_error: bool = True):
        """
        Plot denoising results comparing original and denoised data.
        
        Args:
            fig_size: Size of the figure
            title: Optional title for the plot
            full_range: Whether to use the full energy range for plotting
            plot_kspace: Whether to include k-space plot
            time_instance: Specific time instance to plot (None for all)
            show_error: Whether to show error bands
        """
        if self.results is None:
            raise ValueError("Processing must be completed before plotting. Call process() first.")
        
        # Extract data for plotting
        x_data = self.original_energy
        y_original = self.original_spectrum
        y_denoised = self.results['denoised']
        y_error = self.results['error'] if show_error else None
        
        # Get edge energy for k-space plotting
        edge_energy = None
        if plot_kspace and hasattr(self.spectrum_obj, 'edge'):
            edge_energy = self.spectrum_obj.edge
        elif plot_kspace:
            # Estimate edge energy from data
            dy = np.diff(np.mean(y_original, axis=1) if y_original.ndim > 1 else y_original)
            dx = np.diff(x_data)
            edge_idx = np.argmax(dy / dx)
            edge_energy = x_data[edge_idx]
        
        # Handle multi-dimensional data
        if y_original.ndim == 2 and y_original.shape[1] > 1:
            n_time_instances = y_original.shape[1]
            if time_instance is not None:
                if time_instance >= n_time_instances:
                    raise ValueError(f"time_instance {time_instance} exceeds available instances ({n_time_instances})")
                y_original = y_original[:, time_instance:time_instance+1]
                y_denoised = y_denoised[:, time_instance:time_instance+1]
                if y_error is not None:
                    y_error = y_error[:, time_instance:time_instance+1]
                plot_single = True
            else:
                plot_single = False
        else:
            plot_single = True
            if y_original.ndim == 1:
                y_original = y_original[:, np.newaxis]
            if y_denoised.ndim == 1:
                y_denoised = y_denoised[:, np.newaxis]
            if y_error is not None and y_error.ndim == 1:
                y_error = y_error[:, np.newaxis]
            n_time_instances = 1
        
        # Create plots
        plt.figure(figsize=fig_size)
        if title:
            plt.suptitle(title)
        
        # Determine subplot layout
        n_plots = 1 if not plot_kspace else 2
        
        # Plot 1: Energy domain
        plt.subplot(n_plots, 1, 1)
        
        if plot_single:
            # Single time instance
            plt.plot(x_data, y_original[:, 0], linestyle="None", marker="o", markersize=1,
                    color="tab:blue", alpha=0.3, label="Original")
            plt.plot(x_data, y_denoised[:, 0], label='Denoised', color="tab:orange", linewidth=1.5)
            
            # Add error bands if available
            if y_error is not None and np.sum(y_error) > 0:
                plt.fill_between(x_data, y_denoised[:, 0] - y_error[:, 0]*2, 
                               y_denoised[:, 0] + y_error[:, 0]*2,
                               color="tab:orange", alpha=0.3, label=r"95% confidence interval")
        else:
            # Multiple time instances
            alpha_val = max(0.1, 0.5 / n_time_instances)
            
            plt.plot(x_data, y_original, linestyle="None", marker="o", markersize=1,
                    color="tab:blue", alpha=alpha_val, label="Original")
            plt.plot(x_data, y_denoised, color="tab:orange", linewidth=1,
                    alpha=alpha_val, label="Denoised")
            
            # Add error bands for first time instance if available
            if y_error is not None and np.sum(y_error) > 0:
                plt.fill_between(x_data, y_denoised[:, 0] - y_error[:, 0]*2,
                               y_denoised[:, 0] + y_error[:, 0]*2,
                               color="tab:orange", alpha=alpha_val*0.5, label=r"95% confidence interval")
        
        plt.legend(loc='upper right')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Absorption")
        # plt.title("Energy Domain Comparison")
        
        # Plot 2: K-space domain (if requested)
        if plot_kspace and edge_energy is not None:
            plt.subplot(n_plots, 1, 2)
            
            # Get k-space weights
            try:
                weights = self.spectrum_obj.ksquared_weights()[:, np.newaxis]
            except:
                # Fallback k-space calculation
                k_points = denoising_utils.energy_to_wavenumber(x_data, edge_energy)
                weights = k_points[:, np.newaxis]**2
            
            # Get baseline for k-space plotting
            try:
                if hasattr(self.spectrum_obj, 'background') and self.spectrum_obj.background is not None:
                    baseline = self.spectrum_obj.background
                    if baseline.ndim == 1:
                        baseline = baseline[:, np.newaxis]
                else:
                    # Estimate baseline
                    baseline = baseline_estimation.fit_edge_step(
                        x_data, np.mean(y_denoised, axis=1), edge_energy
                    )[:, np.newaxis]
            except:
                baseline = np.zeros_like(y_denoised[:, 0:1])
            
            if full_range:
                k_data = denoising_utils.energy_to_wavenumber(x_data, edge_energy)
                ky_original = (y_original - baseline) * weights
                ky_denoised = (y_denoised - baseline) * weights
                ky_error = y_error * weights if y_error is not None else None
            else:
                # Crop to EXAFS region only
                exafs_mask = x_data > edge_energy
                x_exafs = x_data[exafs_mask]
                y_orig_exafs = y_original[exafs_mask, :]
                y_den_exafs = y_denoised[exafs_mask, :]
                baseline_exafs = baseline[exafs_mask, :]
                weights_exafs = weights[exafs_mask, :]
                
                k_data = denoising_utils.energy_to_wavenumber(x_exafs, edge_energy)
                ky_original = (y_orig_exafs - baseline_exafs) * weights_exafs
                ky_denoised = (y_den_exafs - baseline_exafs) * weights_exafs
                ky_error = y_error[exafs_mask, :] * weights_exafs if y_error is not None else None
            
            if plot_single:
                plt.plot(k_data, ky_original[:, 0], linestyle="None", marker="o", markersize=1,
                        color="tab:blue", alpha=0.3, label="Original")
                plt.plot(k_data, ky_denoised[:, 0], label='Denoised', color="tab:orange", linewidth=1.5)
                
                if ky_error is not None and np.sum(ky_error) > 0:
                    plt.fill_between(k_data, ky_denoised[:, 0] - ky_error[:, 0]*2,
                                   ky_denoised[:, 0] + ky_error[:, 0]*2,
                                   color="tab:orange", alpha=0.3, label=r"95% confidence interval")
            else:
                alpha_val = max(0.1, 0.5 / n_time_instances)
                plt.plot(k_data, ky_original, linestyle="None", marker="o", markersize=1,
                        color="tab:blue", alpha=alpha_val, label="Original")
                plt.plot(k_data, ky_denoised, color="tab:orange", linewidth=1,
                        alpha=alpha_val, label="Denoised")
                
                if ky_error is not None and np.sum(ky_error) > 0:
                    plt.fill_between(k_data, ky_denoised[:, 0] - ky_error[:, 0]*2,
                                   ky_denoised[:, 0] + ky_error[:, 0]*2,
                                   color="tab:orange", alpha=alpha_val*0.5, label=r"95% confidence interval")
            
            plt.legend(loc='upper right')
            plt.xlabel("Wavenumber (k)")
            plt.ylabel("k²-weighted absorption")
            # plt.title("K-space Comparison")
        
        plt.tight_layout()
        plt.show()
        
        return self