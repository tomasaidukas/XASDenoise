"""
Input Warping Module

This module handles the transformation of input coordinates (energy grid) 
from original domain to warped domain.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union


class InputWarper:
    """
    Handles input coordinate warping for XAS data processing.
    
    This class transforms energy grids using various warping methods
    (k-space, smoothness-based) to create more stationary representations
    for improved denoising performance.
    """
    
    def __init__(self, method: str = 'kspace', verbose: int = 0):
        """
        Initialize the input warper.
        
        Args:
            method: Warping method ('kspace', 'smoothness', 'kspace_exafs_smoothness_xanes', etc.)
            verbose: Verbosity level (0=quiet, 1=basic info, 2=detailed with plots)
        """
        self.method = method
        self.verbose = verbose
        
        # State variables
        self.x_original = None
        self.x_warped = None
        self.edge_energy = None
        self.smoothness = None
        self.initialized = False
        
    def initialize(self, x: np.ndarray, edge_energy: Optional[float] = None, 
                   smoothness: Optional[np.ndarray] = None) -> 'InputWarper':
        """
        Initialize warping with input coordinates and parameters.
        
        Args:
            x: Original x-coordinates (energy grid)
            edge_energy: Edge energy for k-space warping
            smoothness: Smoothness values for smoothness-based warping
            
        Returns:
            Self for method chaining
        """
        self.x_original = x.copy()
        self.edge_energy = edge_energy
        self.smoothness = smoothness
        
        # Apply the warping transformation
        self.x_warped = self._apply_warping(x, edge_energy, smoothness)
        self.initialized = True
        
        
        if self.verbose > 0:
            self.visualize_warping() 
            print(f"Input warping initialized using '{self.method}' method")
            print(f"Original x range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"Warped x range: [{self.x_warped.min():.4f}, {self.x_warped.max():.4f}]")
            
        return self
    
    def get_warped_coordinates(self) -> np.ndarray:
        """Get the warped coordinates."""
        if not self.initialized:
            raise ValueError("Must call initialize() before getting warped coordinates")
        return self.x_warped.copy()
    
    def get_original_coordinates(self) -> np.ndarray:
        """Get the original coordinates."""
        if not self.initialized:
            raise ValueError("Must call initialize() before getting original coordinates")
        return self.x_original.copy()
    
    def _apply_warping(self, x: np.ndarray, edge_energy: Optional[float], 
                      smoothness: Optional[np.ndarray]) -> np.ndarray:
        """
        Apply the selected warping method to coordinates.
        
        Args:
            x: Original coordinates
            edge_energy: Edge energy for k-space methods
            smoothness: Smoothness values for smoothness methods
            
        Returns:
            Warped coordinates
        """
        if self.method == 'kspace':
            if edge_energy is None:
                raise ValueError("Edge energy must be provided for k-space warping")
            return self._energy_to_wavenumber(x, edge_energy)
        
        elif self.method == 'kspace_exafs':
            if edge_energy is None:
                raise ValueError("Edge energy must be provided for k-space warping")
            return self._energy_to_wavenumber(x, edge_energy, mode='linear_pre_edge')
        
        elif self.method == 'smoothness':
            if smoothness is None:
                raise ValueError("Smoothness values must be provided for smoothness warping")
            # Ensure smoothness is properly shaped
            smoothness_1d = np.asarray(smoothness).squeeze()
            if smoothness_1d.ndim != 1:
                raise ValueError(f"Smoothness must be 1D after squeezing, got shape {smoothness_1d.shape}")
            return self._smoothness_warping(x, smoothness_1d)
        
        elif self.method == 'kspace_exafs_smoothness_xanes':
            if edge_energy is None or smoothness is None:
                raise ValueError("Both edge energy and smoothness must be provided")
            # Ensure smoothness is properly shaped
            smoothness_1d = np.asarray(smoothness).squeeze()
            if smoothness_1d.ndim != 1:
                raise ValueError(f"Smoothness must be 1D after squeezing, got shape {smoothness_1d.shape}")
            return self._combined_warping(x, edge_energy, smoothness_1d)
        
        elif self.method is None or self.method == 'none':
            return x.copy()
        
        else:
            raise ValueError(f"Unknown warping method: {self.method}")
    
    def _energy_to_wavenumber(self, energy: np.ndarray, edge: float, 
                             mode: Optional[str] = None) -> np.ndarray:
        """
        Transform energy values to wavenumber (k-space) values.
        
        Args:
            energy: Array of energy values
            edge: Edge energy
            mode: Special handling modes ('linear_pre_edge')
            
        Returns:
            K-space values
        """
        # Calculate energy difference relative to the edge
        dE = energy - edge
        
        # Apply standard k-space transformation
        k = np.sign(dE) * (np.sqrt(0.2625 * np.abs(dE)))
        
        # Apply special handling for different modes if specified
        if mode == 'linear_pre_edge':
            # Apply linear transformation to the pre-edge region
            idx_pre = np.where(energy < edge)[0]
            if len(idx_pre) > 0:
                k[idx_pre] = np.linspace(k[idx_pre].min(), k[idx_pre].max(), len(idx_pre))
        
        return k
    
    def _smoothness_warping(self, x: np.ndarray, smoothness: np.ndarray) -> np.ndarray:
        """
        Warp coordinates based on smoothness values.
        
        Args:
            x: Original x-coordinates
            smoothness: Smoothness values
                        
        Returns:
            Warped coordinates
        """
        # Ensure smoothness is 1D and has the right length
        smoothness = np.asarray(smoothness).squeeze()
        if smoothness.ndim != 1:
            raise ValueError(f"Smoothness must be 1D after squeezing, got shape {smoothness.shape}")
        
        # Get warping factors from smoothness
        factors = self._get_smoothness_compression_factors(smoothness)
        
        # Apply warping using compression factors
        return self._warp_with_compression_factors(x, factors)
    
    def _combined_warping(self, x: np.ndarray, edge_energy: float, 
                         smoothness: np.ndarray) -> np.ndarray:
        """
        Apply combined k-space and smoothness warping.
        
        Args:
            x: Original coordinates
            edge_energy: Edge energy
            smoothness: Smoothness values
            
        Returns:
            Combined warped coordinates
        """
        # Start with k-space warping
        x_kspace = self._energy_to_wavenumber(x, edge_energy)
        
        # For pre-edge/XANES region, use smoothness warping
        # For EXAFS region, use k-space
        pre_edge_mask = x < edge_energy
        
        if np.any(pre_edge_mask):
            x_smooth = self._smoothness_warping(
                x_kspace[pre_edge_mask], 
                smoothness[pre_edge_mask]
            )
            x_warped = x_kspace.copy()
            x_warped[pre_edge_mask] = x_smooth
        else:
            x_warped = x_kspace
        
        return x_warped

    def _warp_with_compression_factors(self, x: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """
        Get a warping function from the compression factors.
        
        Args:
            x: Original x-coordinates
            factors: Grid compression factors
            
        Returns:
            Warped coordinates
        """
        # Compute intervals between points
        dX = np.diff(x, prepend=x[0])
        
        # The coordinate transformation function is given by the integral
        # of the inverse of the compression factors
        intervals_warped = dX / factors
        
        # Compute cumulative sum to get the new grid
        x_warped = np.cumsum(intervals_warped)
        
        # Normalize to preserve original range
        x_warped = (x_warped - x_warped[0]) / (x_warped[-1] - x_warped[0])
        x_warped = x_warped * (x[-1] - x[0]) + x[0]
        
        return x_warped

    def _get_smoothness_compression_factors(self, smoothness: np.ndarray) -> np.ndarray:
        """
        Calculate compression factors from smoothness values.

        Args:
            smoothness: Smoothness values for each point
            
        Returns:
            Warping factors derived from smoothness
        """
        # Apply square root to smoothness (Hessian) to get the compression factors
        factors = smoothness**0.5
        
        # Normalize
        # factors = factors / np.max(factors)
        # factors = factors / np.mean(factors)        
        
        # Compression factors are inverse of the Hessian
        factors = 1 / (factors+1e-6)

        factors = factors / np.max(factors)

        return factors
    
    def visualize_warping(self, title: str = "Input Warping Results"):
        """
        Visualize the warping transformation.
        
        Args:
            title: Plot title
        """
        if not self.initialized:
            raise ValueError("Must call initialize() before visualizing")
            
        plt.figure(figsize=(6, 4))
        
        plt.plot(self.x_original, self.x_warped, 'o-', markersize=2)
        plt.xlabel('Original coordinates')
        plt.ylabel('Warped coordinates')
        plt.title('Warping Function')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
