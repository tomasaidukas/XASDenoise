"""
Post-processing Module

This module handles the final post-processing steps including:
- Data unwarping back to original domain
- Baseline restoration
- Scaling restoration
- Results packaging and validation
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class DataPostprocessor:
    """
    Handles post-processing operations for denoised XAS data.
    
    This class manages the restoration of original data properties
    including unwarping, baseline addition, and scaling restoration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data postprocessor.
        
        Args:
            config: Configuration dictionary containing post-processing parameters
        """
        self.config = config
        self.verbose = config.get('verbose', 0)
    
    def process(self, y_denoised: np.ndarray, y_error: np.ndarray, y_noise: np.ndarray,
                preprocessor, warper, 
                y_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply post-processing steps to denoised data.
        
        Args:
            y_denoised: Denoised spectrum data
            y_error: Error estimates
            y_noise: Noise estimates
            preprocessor: DataPreprocessor instance with stored state
            warper: DataWarper instance with stored state
            y_weights: Optional data weights for scaling restoration
            
        Returns:
            Tuple of (final_denoised, final_error, final_noise)
        """
        if self.verbose > 0:
            print('--------------- Post-processing ---------------')
        
        # Step 1: Unwarp data back to original domain
        x_final, y_denoised_unwarped, y_error_unwarped, y_noise_unwarped = self.undo_warping(
            y_denoised, y_error, y_noise, warper
        )
        
        # Step 2: Restore baseline and scaling
        y_final, y_error_final, y_noise_final = self._restore_baseline_and_scaling(
            y_denoised_unwarped, y_error_unwarped, y_noise_unwarped, preprocessor, y_weights
        )
        
        # Step 3: Validate results
        self._validate_results(y_final, y_error_final, y_noise_final)
        
        return y_final, y_error_final, y_noise_final
    
    def undo_warping(self, y_denoised: np.ndarray, y_error: np.ndarray, y_noise: np.ndarray,
                    warper) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unwarp data back to original energy domain."""
        
        
        if self.config.get('input_warping_method') is None:
            # No warping was applied - return original energy array
            if hasattr(warper, 'original_data') and 'x' in warper.original_data:
                x_original = warper.original_data['x']
            else:
                # Fallback - assume no change needed
                x_original = None
            return x_original, y_denoised, y_error, y_noise
        
        if self.verbose > 0:
            print('Unwarping data back to original domain')
        
        # Use warper to unwarp the data
        x_unwarped, y_denoised_unwarped, y_error_unwarped, y_noise_unwarped = warper.undo_warping(
            y_denoised, y_error, y_noise
        )
        
        return x_unwarped, y_denoised_unwarped, y_error_unwarped, y_noise_unwarped
    
    def _restore_baseline_and_scaling(self, y_denoised: np.ndarray, y_error: np.ndarray, 
                                     y_noise: np.ndarray, preprocessor,
                                     y_weights: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Restore baseline and undo scaling transformations."""
        
        if self.verbose > 0:
            print('Restoring baseline and scaling')
        
        # Use preprocessor to restore baseline and scaling
        y_restored, y_error_restored, y_noise_restored = preprocessor.restore_baseline_and_scaling(
            y_denoised, y_error, y_noise, y_weights
        )
        
        return y_restored, y_error_restored, y_noise_restored
    
    def _validate_results(self, y_denoised: np.ndarray, y_error: np.ndarray, y_noise: np.ndarray):
        """Validate final results for consistency and quality."""
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(y_denoised)):
            raise ValueError("Final denoised data contains NaN or infinite values")
        
        if np.any(~np.isfinite(y_error)):
            if self.verbose > 0:
                print("Warning: Error estimates contain NaN or infinite values")
        
        if np.any(~np.isfinite(y_noise)):
            if self.verbose > 0:
                print("Warning: Noise estimates contain NaN or infinite values")
        
        # Check shapes are consistent
        if y_denoised.shape != y_error.shape:
            raise ValueError(f"Shape mismatch: denoised {y_denoised.shape} vs error {y_error.shape}")
        
        if y_denoised.shape != y_noise.shape:
            raise ValueError(f"Shape mismatch: denoised {y_denoised.shape} vs noise {y_noise.shape}")
        
        if self.verbose > 0:
            print(f'Post-processing complete.')
    
    def package_results(self, y_denoised: np.ndarray, y_error: np.ndarray, 
                       y_noise: np.ndarray, x_final: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Package final results into a dictionary.
        
        Args:
            y_denoised: Final denoised data
            y_error: Final error estimates
            y_noise: Final noise estimates
            x_final: Final energy array
            
        Returns:
            Dictionary containing all results
        """
        results = {
            'denoised': y_denoised,
            'error': y_error,
            'noise': y_noise
        }
        
        if x_final is not None:
            results['energy'] = x_final
        
        return results
