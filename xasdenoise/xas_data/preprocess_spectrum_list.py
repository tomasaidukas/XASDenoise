"""
Functions used to process a list of Spectrum objects.
"""

import numpy as np
from scipy.interpolate import interp1d
from . import preprocess_spectrum as preproc
from xasdenoise.utils import artefacts
from multiprocessing import Pool


def time_average(spectrum_list):
    """
    Compute and return the time-averaged spectrum for each spectrum in the list.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    for data in spectrum_list:
        preproc.time_average(data)

def remove_bad_time_instances(spectrum_list):
    """
    Remove bad time instances (e.g., where the sample was not present) for each spectrum in the list.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    for data in spectrum_list:
        preproc.remove_bad_time_instances(data)

def remove_nans(spectrum_list):
    """
    Interpolate and remove NaN values for each spectrum in the list.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    for data in spectrum_list:
        preproc.remove_nans(data)

def remove_compound(spectrum_list, compound):
    """
    Remove spectra associated with a specific compound from the list.

    Args:
        spectrum_list (list): List of Spectrum objects.
        compound (str): Compound name to remove.
    """
    spectrum_list[:] = [s for s in spectrum_list if s.metadata['compound'] != compound]

def remove_element(spectrum_list, element):
    """
    Remove spectra associated with a specific element from the list.

    Args:
        spectrum_list (list): List of Spectrum objects.
        element (str): Element name to remove.
    """
    spectrum_list[:] = [s for s in spectrum_list if s.metadata['element'] != element]

def show_nans(spectrum_list):
    """
    Display the spectra in the list that contain NaN values.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    corrupted_indices = [i for i, s in enumerate(spectrum_list) if preproc.show_nans(s)]

    if corrupted_indices:
        path_key = [key for key in spectrum_list[0].metadata.keys() if 'path' in key][0]
        print("The following spectra contain NaNs:")
        for i in corrupted_indices:
            print(spectrum_list[i].metadata[path_key])
    else:
        print("No NaNs were found.")

def crop_other_edges(spectrum_list):
    """
    Crop the spectra to retain only the main edge, removing additional L-edges.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    for data in spectrum_list:
        preproc.crop_other_edges(data)

def crop_spectra(spectrum_list, pre_edge=50, post_edge=50):
    """
    Crop the spectra to a specified pre-edge and post-edge region.

    Args:
        spectrum_list (list): List of Spectrum objects.
        pre_edge (int, optional): Energy range before the edge. Defaults to 50.
        post_edge (int, optional): Energy range after the edge. Defaults to 50.
    """
    for data in spectrum_list:
        preproc.crop_spectrum(data, pre_edge, post_edge)

def crop_spectra_energy(spectrum_list, energy_range):
    """
    Crop the spectra to a specified energy range.

    Args:
        spectrum_list (list): List of Spectrum objects.
        energy_range (np.ndarray): The energy range to crop the spectrum to.
    """
    for data in spectrum_list:
        preproc.crop_spectrum_energy(data, energy_range)

def compute_mu(spectrum_list):
    """
    Compute the absorption spectrum for each spectrum in the list.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    for data in spectrum_list:
        data.compute_mu()

def normalize_single_spectrum(args):
    """
    Normalize a single Spectrum object.

    Args:
        args (tuple): A tuple containing:
            - Spectrum object to process.
            - fitting_funcs (list): Fitting functions for pre/post-edge fitting.
            - fit_individual (bool): Whether to fit each time instance individually.
            - downsample (int): Downsample factor for normalization.

    Returns:
        Spectrum: The normalized Spectrum object.
    """
    data, fitting_funcs, fit_individual, downsample = args
    print(f"Normalizing {data.metadata['compound']}...")
    preproc.normalize_spectrum(
        data,
        fitting_funcs=fitting_funcs,
        fit_individual=fit_individual,
        downsample=downsample,
    )
    return data

def normalize_spectrum(spectrum_list, fitting_funcs=['1','V'], fit_individual=True, downsample=1):
    """
    Normalize the spectra in the list using multiprocessing.

    Args:
        spectrum_list (list): List of Spectrum objects.
        fitting_funcs (list, optional): Functions for pre/post-edge fitting. Defaults to linear fit for pre-edge and victoreen fit for post-edge.
        fit_individual (bool, optional): Whether to fit each time instance individually. Defaults to True.
        downsample (int, optional): Downsample factor for normalization. Defaults to 1.

    Returns:
        list: List of normalized Spectrum objects.
    """
    # Prepare arguments for multiprocessing
    args_list = [
        (data, fitting_funcs, fit_individual, downsample)
        for data in spectrum_list
    ]

    # Parallelize using Pool
    with Pool() as pool:
        updated_spectra = pool.map(normalize_single_spectrum, args_list)
        
    # Overwrite the original list with the normalized objects (in-place)
    for i, spectrum in enumerate(updated_spectra):
        spectrum_list[i] = spectrum

def estimate_single_background(spectrum):
    """
    Estimate the background for a single Spectrum object.

    Args:
        spectrum (Spectrum): A Spectrum object to process.

    Returns:
        Spectrum: The Spectrum object with updated background.
    """
    print(f"Estimating background for {spectrum.metadata['compound']}...")
    background = preproc.estimate_background(spectrum)
    spectrum.background = background
    return spectrum

def estimate_background(spectrum_list):
    """
    Estimate the background of an XAS spectrum using multiprocessing.

    Args:
        spectrum_list (list): List of Spectrum objects.
        

    Returns:
        list: List of Spectrum objects with updated background.
    """
    # Prepare arguments for multiprocessing
    args_list = [data for data in spectrum_list]

    # Parallelize using Pool
    with Pool() as pool:
        updated_spectra = pool.map(estimate_single_background, args_list)

    # Overwrite the original list with the normalized objects
    for i, spectrum in enumerate(updated_spectra):
        spectrum_list[i] = spectrum
        
def downsample_spectra(spectrum_list, size=None, factor=None):
    """
    Downsample the spectra by a given size or factor.

    Args:
        spectrum_list (list): List of Spectrum objects.
        size (int, optional): Desired size of the downsampled spectrum. Defaults to None.
        factor (int, optional): Downsampling factor. Defaults to None.
    """
    for data in spectrum_list:
        preproc.downsample_spectrum(data, size, factor)

def copy_spectra(spectrum_list):
    """
    Create a copy of the spectrum list.

    Args:
        spectrum_list (list): List of Spectrum objects.

    Returns:
        list: Copied list of Spectrum objects.
    """
    return [s.copy() for s in spectrum_list]

def get_spectra(spectrum_list, key=None, value=None, copy=True):
    """
    Retrieve a subset of spectra based on metadata key and value.

    Args:
        spectrum_list (list): List of Spectrum objects.
        key (str, optional): Metadata key for filtering (e.g., 'element'). Defaults to None.
        value (str, optional): Metadata value for filtering. Defaults to None.
        copy (bool, optional): Whether to return a copied subset. Defaults to True.

    Returns:
        list: Subset of Spectrum objects.
    """
    metadata_keys = spectrum_list[0].metadata.keys()
    if key not in metadata_keys:
        raise ValueError("Please specify a valid key: 'element' or 'name'")

    if key is None:
        if copy:
            return [s.copy() for s in spectrum_list]
        else:
            return [s for s in spectrum_list]
    else:
        if copy:
            filtered_spectra = [s.copy() for s in spectrum_list if s.metadata[key] == value]
        else:
            filtered_spectra = [s for s in spectrum_list if s.metadata[key] == value]
        return filtered_spectra
    
def filter_spectra_by_element(spectrum_list, element):
    """
    Filter spectra by a specified element.

    Args:
        spectrum_list (list): List of Spectrum objects.
        element (str): Element to filter by.

    Returns:
        list: Filtered list of Spectrum objects containing the specified element.
    """
    return [s for s in spectrum_list if s.metadata['element'] == element]

def get_loaded_elements(spectrum_list):
    """
    Get a list of unique elements present in the spectrum list.

    Args:
        spectrum_list (list): List of Spectrum objects.

    Returns:
        np.ndarray: Array of unique elements in the spectra.
    """
    return np.unique([s.metadata['element'] for s in spectrum_list])

def print_loaded_elements(spectrum_list):
    """
    Print the unique elements present in the spectrum list.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    elements = ', '.join(get_loaded_elements(spectrum_list))
    print(f'The following elements were loaded: {elements}')

def get_loaded_compounds(spectrum_list):
    """
    Get a list of unique compounds present in the spectrum list.

    Args:
        spectrum_list (list): List of Spectrum objects.

    Returns:
        np.ndarray: Array of unique compounds in the spectra.
    """
    return np.unique([s.metadata['compound'] for s in spectrum_list])

def print_loaded_compounds(spectrum_list):
    """
    Print the unique compounds present in the spectrum list.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    compounds = ', '.join(get_loaded_compounds(spectrum_list))
    print(f'The following compounds were loaded: {compounds}')

def compute_all_descriptors(spectrum_list):
    """
    Compute all descriptors for each spectrum in the list.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    for spectrum in spectrum_list:
        spectrum.compute_all_descriptors()

def create_glitch_mask_from_regions(spectrum_list, glitch_regions):
    """
    Create glitch masks for spectra based on glitch region indices.

    Args:
        spectrum_list (list): List of Spectrum objects.
        glitch_regions (dict): Dictionary of glitch regions with start and end indices for each spectrum.
    """
    for spectrum in spectrum_list:
        spectrum.glitch_mask = None
        for key, value in glitch_regions.items():
            if spectrum.metadata['compound'] == key:
                glitch_mask = artefacts.construct_glitch_mask_from_regions(spectrum.energy, value)
                if glitch_mask is not None and np.sum(glitch_mask) > 0:
                    spectrum.glitch_mask = glitch_mask

def find_common_glitches(spectrum_list, element, threshold=None):
    """
    Identify common glitches for a given element using averaged spectra.

    Args:
        spectrum_list (list): List of Spectrum objects.
        element (str): Element for which to find glitches.
        threshold (float, optional): Threshold value for glitch identification. Defaults to None.
    """
    spectrum_element = get_spectra(spectrum_list, key='element', value=element, copy=True)
    interpolate_to_max_range(spectrum_element)
    x_new = spectrum_element[0].energy

    dummy_spectrum = spectrum_element[0].copy()
    dummy_spectrum.spectrum = np.mean([s.time_averaged_spectrum for s in spectrum_element], axis=0)[:, None]
    dummy_spectrum.I0 = np.mean([s.time_averaged_I0 for s in spectrum_element], axis=0)[:, None]
    dummy_spectrum.I1 = np.mean([s.time_averaged_I1 for s in spectrum_element], axis=0)[:, None]

    glitch_mask = preproc.find_glitches(dummy_spectrum, glitch_refinement_fit=True, threshold=threshold, plot=True)
    f = interp1d(x_new, glitch_mask, kind='nearest', bounds_error=False, axis=0, fill_value=(0, 0))

    spectrum_element = get_spectra(spectrum_list, key='element', value=element, copy=False)
    for spectrum in spectrum_element:
        spectrum.glitch_mask = (f(spectrum.energy) >= 0.5).astype(bool)

def compute_common_glitch_mask(spectrum_list, num_pts=None):
    """
    Compute a common glitch mask across all spectra by averaging individual masks.

    Args:
        spectrum_list (list): List of Spectrum objects.
        num_pts (int, optional): Number of points for the common glitch mask. Defaults to None.

    Returns:
        np.ndarray: Common glitch mask for all spectra.
    """
    min_e = np.median([np.min(s.energy) for s in spectrum_list])
    max_e = np.median([np.max(s.energy) for s in spectrum_list])
    if num_pts is None:
        num_pts = np.max([len(s.energy) for s in spectrum_list])
    x_new = np.linspace(min_e, max_e, num_pts)

    glitch_masks_new = [
        interp1d(s.energy, s.glitch_mask, kind='nearest', bounds_error=False, axis=0, fill_value=(0, 0))(x_new) >= 0.5
        for s in spectrum_list if s.glitch_mask is not None
    ]

    common_glitch_mask = np.mean(glitch_masks_new, axis=0)
    f = interp1d(x_new, common_glitch_mask, kind='nearest', bounds_error=False, axis=0, fill_value=(0, 0))

    for s in spectrum_list:
        if s.glitch_mask is not None:
            s.glitch_mask = (f(s.energy) >= 0.5).astype(bool)

    return common_glitch_mask

def pad_to_max_length(spectrum_list):
    """
    Pad spectra such that all of them are of the same length.
    The spectrum will be padded to the left and right of the absorption edge.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    max_len = np.max([len(s.energy) for s in spectrum_list])
    
    for s in spectrum_list:
        pre_edge_len = np.sum(s.energy <= s.edge)
        post_edge_len = np.sum(s.energy > s.edge)
        total_len = pre_edge_len + post_edge_len
        pad_len = max_len - total_len
        
        if pad_len > 0:
            # use padding as a fraction of the pre-edge and post-edge lengths
            pre_edge_pad = int(pad_len / total_len * pre_edge_len)
            post_edge_pad = int(pad_len / total_len * post_edge_len)
            # Ensure total padding is correct
            post_edge_pad += pad_len - pre_edge_pad - post_edge_pad 
            
            # Pad the spectrum
            s.pad_data(pre_edge_pad, post_edge_pad)          
            
            # Store the padding information into the data_mask such that we can exclude later if needed
            # this mask has zeros for values which must be excluded and ones for valid values
            s.data_mask[:pre_edge_pad] = False
            s.data_mask[-post_edge_pad:] = False
        
def interpolate_to_same_range(spectrum_list, min_e=None, max_e=None, num_pts=None, method='linear'):
    """
    Interpolate spectra onto a common energy range across the spectrum list.

    Args:
        spectrum_list (list): List of Spectrum objects.
        min_e (float, optional): Minimum energy for the common range. Defaults to the median of minimum energies.
        max_e (float, optional): Maximum energy for the common range. Defaults to the median of maximum energies.
        num_pts (int, optional): Number of points for the interpolated range. Defaults to the max points in the spectra.
        method (str, optional): Interpolation method. Defaults to 'linear'.
    """
    if min_e is None:
        min_e = np.median([np.min(s.energy) for s in spectrum_list])
    if max_e is None:
        max_e = np.median([np.max(s.energy) for s in spectrum_list])
    if num_pts is None:
        num_pts = np.max([len(s.energy) for s in spectrum_list])

    new_energy = np.linspace(min_e, max_e, num_pts)
    for spectrum in spectrum_list:
        spectrum.interpolate_spectrum(new_energy, method)

def interpolate_to_max_range(spectrum_list, num_pts=None, method='linear'):
    """
    Interpolate spectra onto the largest energy range across all spectra.

    Args:
        spectrum_list (list): List of Spectrum objects.
        num_pts (int, optional): Number of points for the interpolated range. Defaults to the max points in the spectra.
        method (str, optional): Interpolation method. Defaults to 'linear'.
    """
    min_e = np.min([np.min(s.energy) for s in spectrum_list])
    max_e = np.max([np.max(s.energy) for s in spectrum_list])
    interpolate_to_same_range(spectrum_list, min_e=min_e, max_e=max_e, num_pts=num_pts, method=method)

def interpolate_to_min_range(spectrum_list, num_pts=None, method='linear'):
    """
    Interpolate spectra onto the smallest overlapping energy range across all spectra.

    Args:
        spectrum_list (list): List of Spectrum objects.
        num_pts (int, optional): Number of points for the interpolated range. Defaults to the max points in the spectra.
        method (str, optional): Interpolation method. Defaults to 'linear'.
    """
    min_e = np.max([np.min(s.energy) for s in spectrum_list])
    max_e = np.min([np.max(s.energy) for s in spectrum_list])
    interpolate_to_same_range(spectrum_list, min_e=min_e, max_e=max_e, num_pts=num_pts, method=method)

def find_duplicate_compounds(spectrum_list, delete=False):
    """
    Identify and optionally remove duplicate compounds from the spectrum list.

    Args:
        spectrum_list (list): List of Spectrum objects.
        delete (bool, optional): Whether to remove duplicate spectra. Defaults to False.
    """
    elements = get_loaded_elements(spectrum_list)
    for element in elements:
        spectrum_element = get_spectra(spectrum_list, key='element', value=element, copy=False)
        compounds = get_loaded_compounds(spectrum_element)

        for compound in compounds:
            spectrum_compound = get_spectra(spectrum_element, key='compound', value=compound, copy=False)
            N = len(spectrum_compound)
            if N > 1:
                for _ in range(N - 1):  # Remove duplicates until only one remains
                    noise = [np.mean(np.std(s.spectrum, axis=1)) for s in spectrum_compound]
                    idx = np.argmax(noise)
                    noisiest_spectrum = spectrum_compound[idx]
                    if delete:
                        spectrum_list.remove(noisiest_spectrum)
                    spectrum_compound.remove(noisiest_spectrum)
                    print(f"Removed: {noisiest_spectrum.metadata['compound']}, {noisiest_spectrum.metadata.get('path_spectrum', 'Unknown path')}")

def get_spectrum_noise(spectrum_list):
    """
    Compute noise for each spectrum in the list.

    Args:
        spectrum_list (list): List of Spectrum objects.

    Returns:
        list: Noise values for each spectrum.
    """
    return [s.noise for s in spectrum_list]

def sort_by_edge_location(spectrum_list):
    """
    Sort the spectra by their edge energy.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    spectrum_list.sort(key=lambda s: s.max_derivative)

def sort_by_whiteline_location(spectrum_list):
    """
    Sort the spectra by their whiteline energy.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    spectrum_list.sort(key=lambda s: s.first_maxima_after_edge)

def sort_by_noise(spectrum_list):
    """
    Sort the spectra by their noise level.

    Args:
        spectrum_list (list): List of Spectrum objects.
    """
    spectrum_list.sort(key=lambda s: s.noise, reverse=True)
