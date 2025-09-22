"""
This module provides functions for creating energy grids and resampling XAS spectra 
onto a new common grid.
"""

import numpy as np
from xasdenoise.xas_database import processing_utils


def create_energy_grid(grid_type, E_pre_edge, E_post_edge, N_pts):
    """
    Create a desired energy grid for XAS data based on user parameters.
    
    The energy grids are centered around zero which defines the absorption edge.
    
    Args:
        grid_type (str): Type of energy grid ('linear_in_energy' or 'linear_in_wavenumber')
        E_pre_edge (float): Minimum energy relative to edge (negative for pre-edge)
        E_post_edge (float): Maximum energy relative to edge (positive for post-edge)
        N_pts (int): Number of points in the grid
    
    Returns:
        np.ndarray: Energy grid centered around the edge (relative energies)
    
    Raises:
        ValueError: If grid_type is not supported
    """
    if grid_type == 'linear_in_energy':
        return _create_linear_energy_grid(E_pre_edge, E_post_edge, N_pts)
    
    elif grid_type == 'linear_in_wavenumber':
        return _create_linear_k_grid(E_pre_edge, E_post_edge, N_pts)
    
    else:
        raise ValueError(f"Unsupported grid_type: {grid_type}. "
                        f"Use 'linear_in_energy' or 'linear_in_wavenumber'")

def _create_linear_energy_grid(E_pre_edge, E_post_edge, N_pts):
    """Create a linear energy grid."""
    return np.linspace(E_pre_edge, E_post_edge, N_pts)

def _create_linear_k_grid(E_pre_edge, E_post_edge, N_pts):
    """Create a linear k-space grid with enhanced XANES sampling."""

    # Create temporary uniform k-space grid to calculate step sizes
    temp_energy_grid = np.linspace(E_pre_edge, E_post_edge, N_pts)
    temp_kspace_grid = np.sign(temp_energy_grid) * np.sqrt(0.2625 * abs(temp_energy_grid))
    kspace_grid = np.linspace(temp_kspace_grid[0], temp_kspace_grid[-1], len(temp_kspace_grid))
    energy_grid = np.sign(kspace_grid) * kspace_grid**2 / 0.2625
    
    return energy_grid


def resample_spectra_onto_a_new_energy_grid(spectrum_list, energy_grid=None, 
                                           resampling_method='downsample',
                                           interpolation_method='linear',
                                           **kwargs):
    """
    Resample a single spectrum list onto a new energy grid using interpolation,
    binning, or downsampling.
    
    Args:
        spectrum_list (list): List of Spectrum objects to resample
        energy_grid (np.ndarray, optional): New energy grid (relative to edge).
                                          If None, will be created from kwargs.
        resampling_method (str): Data resampling method ('interpolate', 'downsample', 'bin')
        interpolation_method (str): Interpolation method to use (if applicable)
        **kwargs: Parameters for grid creation or resampling:
            - grid_type (str): 'linear_in_energy' or 'linear_in_wavenumber'
            - E_pre_edge, E_post_edge, N_pts: Grid parameters
            - E_post_edge, E_post_edge, N_pts: Grid parameters
            - bin_size, bin_factor: Binning parameters
            - downsample_size, downsample_factor: Downsampling parameters
    
    Returns:
        list: List of resampled Spectrum objects
    """
    # Make copies
    spectrum_list = processing_utils.copy_spectra(spectrum_list)
    
    # Create energy grid if not provided
    if energy_grid is None and 'E_pre_edge' in kwargs:
        grid_type = kwargs.get('grid_type', 'linear_in_energy')
        E_pre_edge = kwargs['E_pre_edge']
        E_post_edge = kwargs['E_post_edge'] 
        N_pts = kwargs['N_pts']
        energy_grid = create_energy_grid(grid_type, E_pre_edge, E_post_edge, N_pts, **kwargs)
    
    # Process each spectrum
    for spectrum in spectrum_list:
        if resampling_method == 'interpolate':
            _apply_interpolation(spectrum, energy_grid, interpolation_method, **kwargs)

        # Apply binning if requested
        if resampling_method == 'bin':
            _apply_binning(spectrum, energy_grid, **kwargs)
        
        # Apply downsampling if requested
        if resampling_method == 'downsample':
            _apply_downsampling(spectrum, energy_grid, **kwargs)
    
    return spectrum_list

def _apply_interpolation(spectrum, energy_grid, interpolation_method='linear', **kwargs):
    """Apply interpolation to a spectrum."""
    # Store original energy range for mask creation
    energy_old = spectrum.energy.copy()
    
    # Interpolate onto new grid (shift grid by edge energy)
    spectrum.interpolate_spectrum(energy_grid + spectrum.edge, method=interpolation_method)
    
    # Create mask to exclude extrapolated regions
    spectrum.data_mask = ~((spectrum.energy > energy_old.max()) | 
                            (spectrum.energy < energy_old.min()))

def _apply_binning(spectrum, energy_grid, **kwargs):
    """Apply binning to a spectrum."""
    bin_size = kwargs.get('bin_size')
    bin_factor = kwargs.get('bin_factor')
    
    if energy_grid is not None:
        energy_grid_shifted = energy_grid + spectrum.edge
        spectrum.bin_spectrum_onto_grid(energy_grid_shifted)
    else:
        grid_type = kwargs.get('grid_type', 'linear_in_energy')
        spectrum.bin_spectrum(size=bin_size, factor=bin_factor, 
                            energy_grid_type=grid_type)

def _apply_downsampling(spectrum, energy_grid, **kwargs):
    """Apply downsampling to a spectrum."""
    downsample_size = kwargs.get('downsample_size')
    downsample_factor = kwargs.get('downsample_factor')
    
    energy_grid_shifted = energy_grid + spectrum.edge if energy_grid is not None else None
    grid_type = kwargs.get('grid_type', 'linear_in_energy')
    
    spectrum.downsample_spectrum(size=downsample_size, 
                               factor=downsample_factor,
                               energy_grid_type=grid_type,
                               new_energy_grid=energy_grid_shifted)
