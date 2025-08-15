"""
Denoising Module

This module perfoms signal denoising including:
- Data masking and preparation
- Denoising with downsampling
- Region-based denoising (XANES/EXAFS splitting)
- Standard denoising operations
"""

import numpy as np
import copy
from typing import Optional, Dict, Any, Tuple, Union

from xasdenoise.denoising_methods import denoising_utils


class DenoisingProcessor:
    """
    Handles the core denoising operations for XAS data.
    
    This class manages the application of various denoising strategies
    including masking, downsampling, and region-based approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the denoising processor.
        
        Args:
            config: Configuration dictionary containing denoising parameters
        """
        self.config = config
        self.verbose = config.get('verbose', 0)
    
    def process(self, x: np.ndarray, y: np.ndarray, noise: np.ndarray,
                denoiser, smoothness: Optional[np.ndarray] = None,
                data_mask: Optional[np.ndarray] = None,
                y_reference: Optional[np.ndarray] = None,
                edge_energy: Optional[float] = None,
                x_original: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply denoising to the input data.
        
        Args:
            x: Energy array (potentially warped)
            y: Spectrum data (potentially warped)
            noise: Noise estimates
            denoiser: Denoiser object to use
            smoothness: Optional smoothness estimates
            data_mask: Optional data mask
            y_reference: Optional reference data
            edge_energy: Edge energy for region splitting
            x_original: Original energy array (for region splitting)
            
        Returns:
            Tuple of (denoised_data, error_estimates, noise_estimates)
        """
        if self.verbose > 0:
            print('--------------- Data Denoising ---------------')
        
        # Validate inputs
        if denoiser is None:
            raise ValueError("Denoiser must be provided")
        
        # Prepare prediction arrays (full resolution)
        x_predict = x.copy()
        noise_predict = noise.copy()
        
        # Prepare masked data for denoising
        masked_data = self._prepare_masked_data(x, y, noise, smoothness, y_reference, data_mask)
        
        # Apply the appropriate denoising strategy
        y_denoised, y_error, y_noise = self._apply_denoising_strategy(
            masked_data, x_predict, noise_predict, denoiser, edge_energy, x_original
        )
        
        # Convert to float32 for memory efficiency
        y_denoised = self._convert_to_float32(y_denoised)
        y_error = self._convert_to_float32(y_error)
        y_noise = self._convert_to_float32(y_noise)
        
        return y_denoised, y_error, y_noise
    
    def _prepare_masked_data(self, x: np.ndarray, y: np.ndarray, noise: np.ndarray,
                            smoothness: Optional[np.ndarray], y_reference: Optional[np.ndarray],
                            data_mask: Optional[np.ndarray]) -> Dict[str, Union[np.ndarray, None]]:
        """Prepare masked data for denoising while preserving original state."""
        
        if not self.config.get('apply_data_mask', False):
            return {
                'x': x,
                'y': y,
                'noise': noise,
                'smoothness': smoothness,
                'y_reference': y_reference
            }
        
        # Apply mask
        if data_mask is None:
            data_mask = np.ones_like(x, dtype=bool)
        
        masked_data = {
            'x': x[data_mask],
            'y': y[data_mask, :],
            'noise': noise[data_mask, :],
            'smoothness': smoothness[data_mask, :] if smoothness is not None else None,
            'y_reference': y_reference[data_mask, :] if y_reference is not None else None
        }
        
        return masked_data
    
    def _apply_denoising_strategy(self, masked_data: Dict, x_predict: np.ndarray, 
                                 noise_predict: np.ndarray, denoiser,
                                 edge_energy: Optional[float] = None,
                                 x_original: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the appropriate denoising strategy based on configuration."""
        
        # Determine strategy based on configuration
        downsampling_method = self.config.get('data_downsampling_method')
        split_regions = self.config.get('split_xanes_exafs', False)
        
        if downsampling_method is not None:
            return self._denoise_with_downsampling(masked_data, x_predict, noise_predict, denoiser)
        elif split_regions:
            return self._denoise_split_regions(
                masked_data, x_predict, noise_predict, denoiser, edge_energy, x_original
            )
        else:
            return self._denoise_standard(masked_data, x_predict, noise_predict, denoiser)
    
    def _denoise_with_downsampling(self, masked_data: Dict, x_predict: np.ndarray,
                                  noise_predict: np.ndarray, denoiser) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply denoising with downsampling."""
        
        if self.verbose > 0:
            downsampling_pts = self.config.get('downsampling_pts', 1000)
            downsampling_method = self.config.get('data_downsampling_method')
            print(f'Denoising with downsampling: {downsampling_pts} points using {downsampling_method} method')
        
        # Initialize denoiser
        denoiser.initialize_denoiser(
            x=masked_data['x'],
            y=masked_data['y'],
            noise=masked_data['noise'],
            x_predict=x_predict,
            noise_predict=noise_predict,
            y_reference=masked_data['y_reference']
        )
        
        # Apply denoising with downsampling
        return denoiser.denoise_with_downsampling(
            downsampling_pts=self.config.get('downsampling_pts', 1000),
            downsampling_method=self.config.get('data_downsampling_method'),
            smoothness=masked_data['smoothness']
        )
    
    def _denoise_split_regions(self, masked_data: Dict, x_predict: np.ndarray,
                              noise_predict: np.ndarray, denoiser,
                              edge_energy: Optional[float], 
                              x_original: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply denoising by splitting XANES and EXAFS regions."""
        
        if self.verbose > 0:
            print('Denoising with XANES/EXAFS region splitting')
        
        # Determine threshold energy
        if edge_energy is None:
            raise ValueError("Edge energy is required for region splitting")
        
        split_energy = self.config.get('split_xanes_exafs_energy')
        if split_energy is None:
            split_energy = edge_energy
        
        # Use original energy array for splitting if available
        x_for_splitting = x_original if x_original is not None else x_predict
        
        # Define energy regions
        xanes_region = np.where(x_for_splitting < split_energy)[0]
        exafs_region = np.where(x_for_splitting >= split_energy)[0]
        regions = [xanes_region, exafs_region]
        
        # Define masked regions if data mask is applied
        if self.config.get('apply_data_mask', False):
            regions_masked = self._get_masked_regions(split_energy, masked_data['x'])
        else:
            regions_masked = regions
        
        # Initialize output arrays
        y_shape = (len(x_predict), masked_data['y'].shape[1])
        y_denoised = np.zeros(y_shape)
        y_error = np.zeros(y_shape)
        y_noise = np.zeros(y_shape)
        
        # Process each region
        region_names = ['XANES', 'EXAFS']
        for i, (region, region_masked, name) in enumerate(zip(regions, regions_masked, region_names)):
            if len(region_masked) == 0:
                if self.verbose > 0:
                    print(f'Skipping empty {name} region')
                continue
            
            if self.verbose > 0:
                print(f'Processing {name} region: {len(region_masked)} points')
            
            try:
                # Initialize denoiser for this region
                denoiser.initialize_denoiser(
                    x=masked_data['x'][region_masked],
                    y=masked_data['y'][region_masked, :],
                    noise=masked_data['noise'][region_masked, :],
                    x_predict=x_predict[region],
                    noise_predict=noise_predict[region, :],
                    y_reference=masked_data['y_reference'][region_masked, :] if masked_data['y_reference'] is not None else None
                )
                
                # Denoise this region
                y_denoised[region, :], y_error[region, :], y_noise[region, :] = denoiser.denoise()
                
            except Exception as e:
                if self.verbose > 0:
                    print(f'Error in {name} region denoising: {e}')
                    print('Falling back to standard denoising')
                return self._denoise_standard(masked_data, x_predict, noise_predict, denoiser)
        
        return y_denoised, y_error, y_noise
    
    def _denoise_standard(self, masked_data: Dict, x_predict: np.ndarray,
                         noise_predict: np.ndarray, denoiser) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply standard denoising without special processing."""
        
        if self.verbose > 0:
            print('Applying standard denoising')
        
        # Initialize denoiser
        denoiser.initialize_denoiser(
            x=masked_data['x'],
            y=masked_data['y'],
            noise=masked_data['noise'],
            x_predict=x_predict,
            noise_predict=noise_predict,
            y_reference=masked_data['y_reference']
        )
        
        # Apply denoising
        return denoiser.denoise()
    
    def _get_masked_regions(self, split_energy: float, x_masked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get energy regions after applying data mask."""
        
        xanes_region_masked = np.where(x_masked < split_energy)[0]
        exafs_region_masked = np.where(x_masked >= split_energy)[0]
        
        return xanes_region_masked, exafs_region_masked
    
    def _convert_to_float32(self, array: np.ndarray) -> np.ndarray:
        """Convert array to float32 if it's float64 for memory efficiency."""
        if array.dtype == np.float64:
            return array.astype(np.float32)
        return array
