"""
Functions used to load XAS data from various file formats and load them into a Spectrum object.
"""

from pathos.multiprocessing import ProcessingPool as Pool
from xasdenoise.xas_data.spectrum import Spectrum
from xasdenoise.utils import artefacts
import pandas as pd
# from multiprocessing import Pool
import os
import h5py
import numpy as np
import csv
import pickle


def _load_single_spectrum(path, metadata, path_I0=None, path_I1=None):
    """
    Load a single spectrum from a file.

    Args:
        path (str): Path to the spectrum file.
        metadata (pd.Series): Metadata associated with the spectrum.
        path_I0 (str, optional): Path to the I0 file. Defaults to None.
        path_I1 (str, optional): Path to the I1 file. Defaults to None.

    Returns:
        Spectrum: The loaded Spectrum object, or None if loading failed.
    """
    
    try:
        energy, spectrum = _load_from_file(path)
        I0 = None
        I1 = None
        
        if path_I0 is not None:
            _, I0 = _load_from_file(path_I0)
        
        if path_I1 is not None:
            _, I1 = _load_from_file(path_I1)
                        
        try:
            metadata = metadata.to_dict()
        except:
            pass
                
        return Spectrum(energy, spectrum, metadata=metadata, I0=I0, I1=I1)
    except Exception as e:
        print(f"Loading failed for {path}: {e}")
        return None

def _load_from_file(path):
    """
    Load a single spectrum from a file.

    Args:
        path (str): Path to the spectrum file.

    Returns:
        tuple: Tuple containing the energy and spectrum arrays.
    """
    

    if path.endswith(('.csv', '.dat', '.txt', '.nor')):
        return _load_from_csv(path)
    # elif path.endswith('.nor'):
    #     return _load_from_nor(path)
    # elif path.endswith('.nxs'):
    #     return _load_from_nxs(path)
    else:
        raise ValueError(f"Unsupported file format for {path}")
    
def _save_to_file(path, energy, spectrum):
    """
    Save a single spectrum to a file.

    Args:
        path (str): Path to the spectrum file.
        energy (np.ndarray): Energy values of the spectrum.
        spectrum (np.ndarray): Spectrum values.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    
    if path.endswith(('.csv', '.dat', '.txt', '.nor')):
        return _save_to_csv(path, energy, spectrum)
    # elif path.endswith('.nor'):
    #     return _save_to_nor(path, energy, spectrum)
    # elif path.endswith('.nxs'):
    #     return _save_to_nxs(path, energy, spectrum)
    else:
        raise ValueError(f"Unsupported file format for {path}")
    
def _load_from_csv(path):
    """
    Load a single spectrum from a CSV file.

    Args:
        path (str): Path to the spectrum file.

    Returns:
        tuple: Tuple containing the energy and spectrum arrays.
    """
    
    try:
        df = pd.read_csv(path,  sep=r'\s+',  # Use whitespace as the delimiter
                                comment='#',  # Skip comment lines starting with '#'
                                # delimiter='\t',
                                )
        energy = df.values[:, 0]
        spectrum = df.values[:, 1:]
        
        return energy, spectrum
    
    except Exception as e:
        print(f"Loading failed for {path}: {e}")
        return None

def _save_to_csv(path, energy, spectrum):
    """
    Save a single spectrum to a CSV file.

    Args:
        path (str): Path to the spectrum file.
        energy (np.ndarray): Energy values of the spectrum.
        spectrum (np.ndarray): Spectrum values.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    
    try:
        df = pd.DataFrame(data=np.column_stack((energy, spectrum)))
        df.to_csv(path, index=False, header=False)
    except Exception as e:
        print(f"Saving failed for {path}: {e}")

    
    
def load_spectra(paths, metadata, paths_I0=None, paths_I1=None):
    """
    Load multiple spectra from files using multiprocessing.

    Args:
        paths (list of str): List of paths to spectrum files.
        metadata (pd.DataFrame): DataFrame containing metadata for each spectrum.
        paths_I0 (list of str, optional): List of paths to I0 files. Defaults to None.
        paths_I1 (list of str, optional): List of paths to I1 files. Defaults to None.

    Returns:
        list of Spectrum: List of loaded Spectrum objects.
    """
    
    print('Loading the following files:')
    for path in paths:
        print(path)

    if paths_I0 is None: paths_I0 = [None] * len(paths)
    if paths_I1 is None: paths_I1 = [None] * len(paths)

    try:
        metadata_list = [row for _, row in metadata.iterrows()]
    except:
        metadata_list = [metadata] * len(paths)
    
    # sequential loading
    # path_metadata_pairs = list(zip(paths, metadata_list))
    # spectra = []
    # for path, metadata in path_metadata_pairs:
    #     spectra.append(_load_single_spectrum(path, metadata))
    
    # parallelized loading
    with Pool(os.cpu_count()) as pool:
        spectra = pool.map(_load_single_spectrum, paths, metadata_list, paths_I0, paths_I1)

    return spectra

def load_spectrum(path, metadata, path_I0=None, path_I1=None):
    """
    Load a single spectrum from a file.

    Args:
        path (str): Path to the spectrum file.
        metadata (pd.Series): Metadata associated with the spectrum.
        path_I0 (str, optional): Path to the I0 file. Defaults to None.
        path_I1 (str, optional): Path to the I1 file. Defaults to None.

    Returns:
        Spectrum: The loaded Spectrum object.
    """
    
    print('Loading the following file:')
    print(path)
    return _load_single_spectrum(path, metadata, path_I0, path_I1)
    
def save_spectra_to_h5(spectra, filename, element=None):
    """
    Save a list of spectra to an HDF5 file. Optionally update only the spectra for a specified element.

    Args:
        spectra (list of Spectrum): List of Spectrum objects to save.
        filename (str): Name of the HDF5 file for storage.
        element (str, optional): Element to filter and save. Defaults to None.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # check if spectra is a list
    if not isinstance(spectra, list):
        spectra = [spectra]
        
    with h5py.File(filename, 'w') as h5file:
        for i, spectrum in enumerate(spectra):
            spectrum_element = spectrum.metadata.get('element', None)
            if element is not None and spectrum_element != element:
                continue  # Skip if element does not match

            spectrum_compound = spectrum.metadata.get('compound', None)
            if spectrum_compound is not None:
                group_name = f'spectrum_{spectrum_compound}'
            else:
                group_name = f'spectrum_{i}'
                
            if group_name in h5file:
                del h5file[group_name]  # Remove existing group for clean saving
            grp = h5file.create_group(group_name)
            
            # Save energy
            grp.create_dataset('energy', data=np.array(spectrum.energy))
            
            # Save all predefined arrays in y_arrays
            for attr in spectrum.y_arrays:
                array = getattr(spectrum, attr)
                if array is not None:
                    grp.create_dataset(attr, data=np.array(array))

            # Save dynamically stored arrays in `new_arrays`
            new_arrays_grp = grp.create_group('new_arrays')
            for key, array in spectrum.new_arrays.items():
                try:
                    new_arrays_grp.create_dataset(key, data=np.array(array))
                except Exception as e:
                    print(f"Failed to save array '{key}' for spectrum {spectrum_compound}: {e}")
                    # print(f"Array shape: {array.shape}")
                    # print(f"Array dtype: {array.dtype}")
                    print(f"Array: {array}")
                    # print(f"Array type: {type(array)}")

            # Save metadata
            metadata_grp = grp.create_group('metadata')
            for key, value in spectrum.metadata.items():
                if value is not None:
                    if isinstance(value, (list, np.ndarray)):
                        metadata_grp.create_dataset(key, data=np.array(value))
                    elif isinstance(value, dict):
                        subgrp = metadata_grp.create_group(key)
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (list, np.ndarray)):
                                subgrp.create_dataset(subkey, data=np.array(subvalue))
                            else:
                                subgrp.attrs[subkey] = subvalue
                    else:
                        metadata_grp.attrs[key] = value
            print(f"Saved spectrum {spectrum.metadata.get('compound', 'Unnamed')}.")
    print(f"Saved spectra to {filename}. Updated only element '{element}'" if element else "Saved all spectra.")

def load_spectra_from_h5(filename, element=None, compound=None):
    """
    Load spectra from an HDF5 file with optional filtering by element and compound.

    Args:
        filename (str): Name of the HDF5 file.
        element (str or list of str, optional): Element(s) to filter spectra. Defaults to None.
        compound (str, optional): Compound to filter spectra. Defaults to None.

    Returns:
        list of Spectrum: List of Spectrum objects matching the filter criteria.
    """
    spectra = []
    with h5py.File(filename, 'r') as h5file:
        for spectrum_name in h5file:
            grp = h5file[spectrum_name]

            # Load metadata
            metadata_grp = grp['metadata']
            metadata = {}
            for key in metadata_grp:
                if isinstance(metadata_grp[key], h5py.Group):
                    subgrp = metadata_grp[key]
                    subdict = {}
                    for subkey in subgrp:
                        subdict[subkey] = np.array(subgrp[subkey])
                    for subkey, subvalue in subgrp.attrs.items():
                        subdict[subkey] = subvalue
                    metadata[key] = subdict
                else:
                    metadata[key] = np.array(metadata_grp[key])
            for key, value in metadata_grp.attrs.items():
                metadata[key] = value

            # Filter spectra by element and compound
            spectrum_element = str(metadata.get('element'))
            spectrum_compound = str(metadata.get('compound'))

            if element is not None:
                if not isinstance(element, list):
                    element = [element]
                element = [str(e).strip().lower() for e in element]
                if spectrum_element.strip().lower() not in element:
                    continue

            if compound is not None:
                if not isinstance(compound, list):
                    compound = [compound]
                compound = [str(c).strip().lower() for c in compound]
                if spectrum_compound.strip().lower() not in compound:
                    continue

            # Load energy and predefined y_arrays
            energy = np.array(grp['energy'])
            arrays = {attr: np.array(grp[attr]) if attr in grp else None for attr in ["spectrum", "I0", "I1", "background", "glitch_mask", "data_mask"]}

            # Load `new_arrays`
            new_arrays = {}
            if 'new_arrays' in grp:
                new_arrays_grp = grp['new_arrays']
                for key in new_arrays_grp:
                    new_arrays[key] = np.array(new_arrays_grp[key])

            # Create Spectrum object and assign new_arrays
            spectrum = Spectrum(energy, arrays['spectrum'], metadata=metadata,
                                I0=arrays['I0'], I1=arrays['I1'], 
                                background=arrays['background'], 
                                glitch_mask=arrays['glitch_mask'],
                                data_mask=arrays['data_mask'])

            # Store dynamically loaded arrays
            spectrum.new_arrays = new_arrays            
            spectra.append(spectrum)
            print(f'Loaded {spectrum_compound} spectrum.')

    print(f'Loaded {len(spectra)} spectra from {filename}')
    
    if len(spectra) == 1:
        return spectra[0]
    else:
        return spectra


def save_glitch_regions(energy, glitch_mask, filename):
    """
    Save glitch regions to a CSV file.

    Args:
        energy (np.ndarray): Energy values of the spectrum.
        glitch_mask (np.ndarray): Boolean mask indicating glitches.
        filename (str): Path to save the glitch regions CSV file.
    """
    
    # Extract the glitch regions
    glitch_regions = artefacts.extract_glitch_region_energies(energy, glitch_mask)
    
    # Save the glitch regions to a CSV file
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Start Energy (eV)', 'End Energy (eV)'])
        for region in glitch_regions:
            writer.writerow(region)

def save_metadata_dict(spectrum_list, filename):
    """
    Save metadata for a list of spectra to a CSV file and a dictionary.

    Args:
        spectrum_list (list of Spectrum): List of Spectrum objects.
        filename (str): Path to save the metadata CSV file.
    """

    # Extract the metadata from all the spectra
    metadata = []
    for spectrum in spectrum_list:
        metadata.append(spectrum.metadata)
    
    # Collect all unique fieldnames from all metadata entries
    all_keys = set()
    for entry in metadata:
        all_keys.update(entry.keys())  # Collect all keys across all dictionaries

    # Convert to a sorted list for consistency
    fieldnames = sorted(all_keys)

    # Save the metadata to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata:
            # Ensure all rows have the same keys by filling missing ones with empty values
            writer.writerow({key: row.get(key, '') for key in fieldnames})
            
    # Save the metadata to a dictionary using pickle
    metadata_dict = {spectrum.metadata['compound']: spectrum.metadata for spectrum in spectrum_list}
    with open(filename.replace('.csv', '.pkl'), 'wb') as file:
        pickle.dump(metadata_dict, file)
    
    
    
def load_glitch_regions(filename):
    """
    Load glitch regions from a CSV file.

    Args:
        filename (str): Path to the glitch regions CSV file.

    Returns:
        list of tuple: List of tuples with start and end energies for each glitch region.
    """
    
    # Load the glitch regions from a CSV file
    glitch_regions = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            start, end = row
            glitch_regions.append((float(start), float(end)))
    
    return glitch_regions

def save_glitch_regions_from_all_spectra(spectrum_list, filename):
    """
    Save glitch regions for all spectra to a CSV file.

    Args:
        spectrum_list (list of Spectrum): List of Spectrum objects.
        filename (str): Path to save the glitch regions CSV file.
    """
    if not isinstance(spectrum_list, list):
        spectrum_list = [spectrum_list]
        
    # Extract the glitch regions from all the spectra
    glitch_regions = {}
    for spectrum in spectrum_list:
        compound = spectrum.metadata['compound']
        glitch_mask = spectrum.glitch_mask
        energy = spectrum.energy
        spectrum_glitch_regions = artefacts.extract_glitch_region_energies(energy, glitch_mask)
        if compound not in glitch_regions:
            glitch_regions[compound] = []
        glitch_regions[compound].extend(spectrum_glitch_regions)
    
    # Save the glitch regions to a CSV file
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Compound', 'Start Energy (eV)', 'End Energy (eV)'])
        for compound, regions in glitch_regions.items():
            for region in regions:
                writer.writerow([compound, *region])
            
def load_glitch_regions_from_all_spectra(filename):
    """
    Load glitch regions for all spectra from a CSV file.

    Args:
        filename (str): Path to the glitch regions CSV file.

    Returns:
        dict: Dictionary with compounds as keys and their glitch regions as values.
    """
    
    # Load the glitch regions from a CSV file
    glitch_regions = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            compound, start, end = row
            if compound not in glitch_regions:
                glitch_regions[compound] = []
            if float(end) - float(start) != 0:
                glitch_regions[compound].append((float(start), float(end)))
    
    return glitch_regions