"""
XAS Data Curation Module

This module provides functions for creating energy grids, resampling spectra,
preprocessing XAS data, and managing HDF5 databases of XAS spectra.
"""

import h5py
import numpy as np

from xasdenoise.xas_data import preprocess_spectrum, preprocess_spectrum_list, data_io


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
                                           method='downsample',
                                           interpolation_method='linear',
                                           **kwargs):
    """
    Resample a single spectrum list onto a new energy grid using interpolation,
    binning, or downsampling.
    
    Args:
        spectrum_list (list): List of Spectrum objects to resample
        energy_grid (np.ndarray, optional): New energy grid (relative to edge).
                                          If None, will be created from kwargs.
        method (str): Data resampling method ('interpolate', 'downsample', 'bin')
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
    spectrum_list = preprocess_spectrum_list.copy_spectra(spectrum_list)
    
    # Create energy grid if not provided
    if energy_grid is None and 'E_pre_edge' in kwargs:
        grid_type = kwargs.get('grid_type', 'linear_in_energy')
        E_pre_edge = kwargs['E_pre_edge']
        E_post_edge = kwargs['E_post_edge'] 
        N_pts = kwargs['N_pts']
        energy_grid = create_energy_grid(grid_type, E_pre_edge, E_post_edge, N_pts, **kwargs)
    
    # Process each spectrum
    for spectrum in spectrum_list:
        if method == 'interpolate':
            _apply_interpolation(spectrum, energy_grid, interpolation_method, **kwargs)

        # Apply binning if requested
        if method == 'bin':
            _apply_binning(spectrum, energy_grid, **kwargs)
        
        # Apply downsampling if requested
        if method == 'downsample':
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

def remove_spectrum_attributes(spectrum_list, attributes):
    """
    Remove specified attributes from every spectra.

    For example arrays I0, I1 might be unwanted to minimize data storage size.

    Args:
        spectrum_list (list): List of Spectrum objects to process
        attributes (list): List of attribute names to remove
    """
    for spectrum in spectrum_list:
        for attr in attributes:
            setattr(spectrum, attr, None)
    
def unify_time_axis(spectrum_list_train, spectrum_list_target):
    """
    If the "training" spectra contain many noisy time instances and the 
    "target" spectrum time instances were time-averaged, we want to pair 
    "training" and "target" time instances which is done by repeating the 
    averaged time instances.
    
    Args:
        spectrum_list_train (list): List of training Spectrum objects
        spectrum_list_target (list): List of target Spectrum objects
    """
    for train, target in zip(spectrum_list_train, spectrum_list_target):
        target.spectrum = np.repeat(target.spectrum, train.spectrum.shape[1], axis=1)

def process_xas_spectra(spectrum_list, 
                        remove_nans=False, remove_other_absorption_edges=False,
                        remove_bad_time_instances=False, remove_duplicate_compounds=False,
                        time_pts_keep=None, time_bin_size=None, time_average=False,
                        crop_energy=None, normalize_spectrum=False, center_edges=False,
                        estimate_background=False, remove_glitches=False):
    """
    Preprocess a list of XAS spectra with cropping, normalization, and edge centering.
    
    This function performs all the common preprocessing steps for XAS spectra:
    - Time manipulation (binning, cropping, averaging of time instances)
    - Energy cropping (crop spectrum energy to the desired pre/post edge values)
    - Normalization
    - Edge centering (align absorption edges to the middle of the rising absorption edge)
    - Background estimation
    
    Args:
        spectrum_list (list): List of Spectrum objects to process
        remove_nans (bool): Whether to remove NaN values
        remove_other_absorption_edges (bool): Whether to remove other absorption edges
        remove_bad_time_instances (bool): Whether to remove bad time instances
        remove_duplicate_compounds (bool): Whether to remove duplicate compounds
        time_pts_keep (int): Number of time points to keep (centered)
        time_bin_size (int): Time bin size 
        time_average (bool): Whether to average time points
        crop_energy (tuple): Energy range (E_pre_edge, E_post_edge) relative to edge
        normalize_spectrum (bool): Whether to normalize spectra
        center_edges (bool): Whether to center absorption edges
        estimate_background (bool): Whether to estimate background functions
    
    Returns:
        list: List of processed Spectrum objects
    """
    # Make copies
    spectrum_list = preprocess_spectrum_list.copy_spectra(spectrum_list)
    
    # Initialize masks if not present
    for spectrum in spectrum_list:        
        if spectrum.glitch_mask is None:
            spectrum.glitch_mask = np.zeros_like(spectrum.energy, dtype=bool)
        if spectrum.data_mask is None:
            spectrum.data_mask = np.ones_like(spectrum.energy, dtype=bool)

    # Remove nans
    if remove_nans:
        preprocess_spectrum_list.remove_nans(spectrum_list)

    # Crop other energy edges
    if remove_other_absorption_edges:
        preprocess_spectrum_list.crop_other_edges(spectrum_list)

    # Remove bad time instances
    if remove_bad_time_instances:
        preprocess_spectrum_list.remove_bad_time_instances(spectrum_list)

    # Find duplicate compounds
    if remove_duplicate_compounds:
        preprocess_spectrum_list.find_duplicate_compounds(spectrum_list, delete=False)
    
    # Crop time instances
    if time_pts_keep is not None:
        for spectrum in spectrum_list:
            n_times = spectrum.spectrum.shape[1]
            start_idx = n_times // 2 - time_pts_keep // 2
            end_idx = start_idx + time_pts_keep
            spectrum.spectrum = spectrum.spectrum[:, start_idx:end_idx]

    # Bin time instances
    if time_bin_size is not None:
        preprocess_spectrum_list.time_average_spectra(spectrum_list, size=time_bin_size)

    # Average time instances
    if time_average:
        preprocess_spectrum_list.time_average_spectra(spectrum_list)

    # Center edges if requested
    if center_edges:
        preprocess_spectrum_list.center_edges(spectrum_list)

    # Crop energy range
    if crop_energy is not None:
        preprocess_spectrum_list.crop_spectra(spectrum_list, pre_edge=crop_energy[0], post_edge=crop_energy[1])
        
    # Normalize spectra
    if normalize_spectrum:
        preprocess_spectrum_list.normalize_spectrum(spectrum_list, downsample=10, fit_individual=True)

    # Estimate background
    if estimate_background:
        for spectrum in spectrum_list:
            preprocess_spectrum.estimate_background(spectrum)
    
    # Remove glitches
    if remove_glitches:
        for spectrum in spectrum_list:    
            glitch_mask = spectrum.glitch_mask
            if (glitch_mask is not None) and (np.sum(glitch_mask) > 0):
                print(f'Processing spectrum {spectrum.compound}')
                preprocess_spectrum.remove_glitches(spectrum, glitch_mask, glitch_fill='interp_avg', crop_edges=True)
                # preprocess_spectrum.remove_glitches(spectrum, glitch_mask, glitch_fill='delete', crop_edges=True)

    return spectrum_list


def load_xas_database(filepath):
    """
    Load an existing HDF5 dataset of XAS data.
    
    Args:
        filepath (str): Path to the HDF5 file
    
    Returns:
        tuple: (spectrum_list, metadata_dict)
            - spectrum_list: List of loaded Spectrum objects
            - metadata_dict: Dictionary containing processing parameters
    """
    spectrum_list = data_io.load_spectra_from_h5(filepath)
    
    # Load processing metadata if available
    metadata_dict = {}
    try:
        with h5py.File(filepath, 'r') as f:
            if 'processing_metadata' in f:
                meta_group = f['processing_metadata']
                for key in meta_group.keys():
                    if isinstance(meta_group[key], h5py.Dataset):
                        metadata_dict[key] = meta_group[key][()]
                        # Convert bytes to string if needed
                        if isinstance(metadata_dict[key], bytes):
                            metadata_dict[key] = metadata_dict[key].decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not load processing metadata: {e}")
    
    return spectrum_list, metadata_dict

def save_xas_database(spectrum_list, filepath, processing_metadata):
    """
    Save processed spectrum list to HDF5 with processing metadata.
    
    Args:
        spectrum_list (list): List of processed Spectrum objects
        filepath (str): Output HDF5 file path
        processing_metadata: Processing parameter dictionary stored for reproducibility
    """
    # Save spectra using existing function
    data_io.save_spectra_to_h5(spectrum_list, filepath)
    
    # Add processing metadata
    with h5py.File(filepath, 'a') as f:
        if 'processing_metadata' in f:
            del f['processing_metadata']
        
        meta_group = f.create_group('processing_metadata')
        
        for key, value in processing_metadata.items():
            if value is not None:
                if isinstance(value, (list, tuple, np.ndarray)):
                    meta_group.create_dataset(key, data=np.array(value))
                elif isinstance(value, str):
                    meta_group.create_dataset(key, data=value.encode('utf-8'))
                else:
                    meta_group.create_dataset(key, data=value)


def append_to_xas_database(new_spectrum_list, database_filepath):
    """
    Append new spectra to an existing database using the same processing parameters.
    
    This function loads the existing database, applies the same processing steps
    to new spectra, and saves the combined dataset.
    
    Args:
        new_spectrum_list (list): List of new Spectrum objects to add
        database_filepath (str): Path to existing HDF5 database
    
    Returns:
        list: Combined spectrum list
    """
    # check if spectrum is a list, if not convert it to a list
    if not isinstance(new_spectrum_list, list):
        new_spectrum_list = [new_spectrum_list]
    
    # Load existing database and metadata
    existing_spectra, metadata = load_xas_database(database_filepath)
    
    if not metadata:
        raise ValueError("No processing metadata found in database. "
                        "Cannot reproduce processing steps.")
    
    # Process new spectra with same parameters
    processed_new_spectra = process_xas_spectra(
        new_spectrum_list,
        remove_nans=metadata.get('remove_nans', False),
        remove_bad_time_instances=metadata.get('remove_bad_time_instances', False),
        remove_other_absorption_edges=metadata.get('remove_other_absorption_edges', False),
        remove_duplicate_compounds=metadata.get('remove_duplicate_compounds', False),
        time_pts_keep=metadata.get('time_pts_keep'),
        time_bin_size=metadata.get('time_bin_size'),
        time_average=metadata.get('time_average', False),
        crop_energy=metadata.get('crop_energy'),
        normalize_spectrum=metadata.get('normalize_spectrum', False),
        center_edges=metadata.get('center_edges', False),
        estimate_background=metadata.get('estimate_background', False),
        remove_glitches=metadata.get('remove_glitches', False)
    )
    
    # Recreate energy grid
    new_energy_grid = metadata.get('new_energy_grid', None)
    if new_energy_grid is None:
        grid_type = metadata.get('grid_type', None)
        E_pre_edge = metadata.get('E_pre_edge', None)
        E_post_edge = metadata.get('E_post_edge', None)
        N_pts = metadata.get('N_pts', None)
        if all(v is not None for v in [grid_type, E_pre_edge, E_post_edge, N_pts]):
            # Recreate from parameters if grid not stored
            new_energy_grid = create_energy_grid(grid_type, E_pre_edge, E_post_edge, N_pts)

    # Resample new spectra onto same grid
    if new_energy_grid is not None:
        processed_new_spectra = resample_spectra_onto_a_new_energy_grid(
            processed_new_spectra,
            energy_grid=new_energy_grid,
            method=metadata.get('grid_resampling_method', 'interpolate'),
            interpolation_method=metadata.get('interpolation_method', 'linear'),
        )
    
    # Combine with existing spectra
    combined_spectra = existing_spectra + processed_new_spectra
    
    # Save combined database
    save_xas_database(combined_spectra, database_filepath, metadata)
    
    return combined_spectra