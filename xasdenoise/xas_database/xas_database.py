"""
XAS Database Management
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import h5py
import pickle
import pandas as pd
import os
from dataclasses import fields as dataclass_fields

from xasdenoise.xas_data.spectrum import Spectrum
from xasdenoise.xas_database.xas_metadata import XASMetadata
from xasdenoise.xas_data import preprocess_spectrum
from xasdenoise.xas_data import data_io
from xasdenoise.xas_database import visualization_utils
from xasdenoise.xas_database import resampling_utils
from xasdenoise.xas_database import processing_utils


class XASDatabase:
    """
    Simple database for managing collections of XAS spectra.
    
    Attributes:
        spectra (List[Spectrum]): List of XAS spectrum objects
        processing_history (List[Dict]): History of processing operations applied
    """
    
    def __init__(self):
        """Initialize the XAS database."""
        self.spectra: List[Spectrum] = []
        self.database_metadata: List[XASMetadata] = []
        self.processing_history: List[Dict[str, Any]] = []

    def __repr__(self) -> str:
        """String representation of the database."""
        return f"XASDatabase(spectra={len(self.spectra)})"
    
    def __len__(self) -> int:
        """Return number of spectra in database."""
        return len(self.spectra)

    def __copy__(self) -> "XASDatabase":
        """Create a shallow copy of the database."""
        import copy
        new_db = XASDatabase()
        new_db.spectra = copy.deepcopy(self.spectra)
        new_db.database_metadata = copy.deepcopy(self.database_metadata)
        new_db.processing_history = copy.deepcopy(self.processing_history)
        return new_db

    def copy(self) -> "XASDatabase":
        return self.__copy__()
    
    # ========================================================================
    # BASIC OPERATIONS
    # ========================================================================
    
    def add_spectrum(self, spectrum: Spectrum, create_metadata: bool = True) -> None:
        """Add a single spectrum to the database."""
        self.spectra.append(spectrum.copy())
        if create_metadata:
            self.append_metadata_from_spectrum(spectrum)

    def add_spectra(self, spectra: List[Spectrum], create_metadata: bool = True) -> None:
        """Add multiple spectra to the database."""
        for s in spectra:
            self.add_spectrum(s, create_metadata)

    def get_spectrum(self, key: str, value: str) -> List[Spectrum]:
        """Get spectra by metadata key-value pair."""
        return [s for i, s in enumerate(self.spectra) if self.database_metadata[i].get(key) == value]

    def delete_all_spectra(self) -> None:
        """Clear all spectra from the database."""
        self.spectra.clear()
        self.processing_history.clear()
        self.database_metadata.clear()

    def delete_spectrum(self, key: str, value: str) -> None:
        """Delete spectra by metadata key-value pair."""
        self.database_metadata = [m for m in self.database_metadata if not m.get(key) == value]
        self.spectra = [s for i, s in enumerate(self.spectra) if not self.database_metadata[i].get(key) == value]

    def delete_spectrum_attributes(self, attribute: str) -> None:
        for s in self.spectra:
            if hasattr(s, attribute):
                print(f"Deleting attribute {attribute} from spectrum {s.compound}")
                setattr(s, attribute, None)

    def process_new_spectra(self):
        """Apply processing steps to newly added spectra"""
        if not self.spectra:
            return
        
        # Loop over processing steps
        for processing_step in self.processing_history:
            
            # Processing params
            params = processing_step['parameters']
            func = processing_step['operation']

            # Loop over spectra
            for i, (spectrum, metadata) in enumerate(zip(self.spectra, self.database_metadata)):

                if func == 'process_xas_spectra' and not metadata.get('processed'):
                    processed_spectra = [spectrum]
                    self._process_xas_spectra(processed_spectra, **params)
                    self.spectra[i] = processed_spectra[0]  # Update the spectrum object in self.spectra
                    metadata.processed = True
                    
                if func == 'resample_xas_spectra' and not metadata.get('resampled'):
                    resampled_spectra = [spectrum]
                    self._resample_xas_spectra(resampled_spectra, **params)
                    self.spectra[i] = resampled_spectra[0]  # Update the spectrum object in self.spectra
                    metadata.resampled = True
            
    def _log_processing_step(self, operation: str, parameters: Dict[str, Any]) -> None:
        """Log a processing step to the history."""
        step = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'parameters': parameters.copy()
        }
        self.processing_history.append(step)

    def get_subset(self, indices: List[int]) -> "XASDatabase":
        """Get a subset of the database by indices."""
        subset_db = XASDatabase()
        subset_db.spectra = [self.spectra[i].copy() for i in indices]
        subset_db.database_metadata = [self.database_metadata[i] for i in indices]
        subset_db.processing_history = self.processing_history.copy()
        return subset_db
    
    # ========================================================================
    # METADATA OPERATIONS
    # ========================================================================
    def create_metadata_from_spectra(self) -> None:
        """Create metadata for all spectra in the database."""
        self.clear_metadata()
        for spectrum in self.spectra:
            self.append_metadata_from_spectrum(spectrum)

    def append_metadata_from_spectrum(self, spectrum: Spectrum) -> None:
        """Append metadata from a single spectrum."""
        metadata = XASMetadata()
        metadata.from_dict(spectrum.metadata)
        self.database_metadata.append(metadata)

    def update_spectrum_metadata(self) -> None:
        for spectrum, metadata in zip(self.spectra, self.database_metadata):
            spectrum.metadata = metadata.to_dict()
            
    def clear_metadata(self):
        """Clear all metadata entries."""
        self.database_metadata.clear()

    def show_metadata(self, attributes: List[str] = None) -> None:
        """Display metadata for all spectra."""
        if not isinstance(attributes, list):
            attributes = [attributes]
            
        for attribute in attributes:
            temp = []
            for metadata in self.database_metadata:
                if attribute is not None:
                    temp.append(metadata.get(attribute))
                else:
                    temp.append(metadata)
            print(f'{attribute}: {temp}')

    def update_metadata(self, attribute: str, value: Any) -> None:
        """Update metadata for all spectra."""
        for metadata in self.database_metadata:
            metadata.update(**{attribute: value})
    
    def get_metadata_field_names(self, include_dynamic_fields: bool = False):
        all_field_names = [f.name for f in dataclass_fields(XASMetadata)]
        
        if include_dynamic_fields:
            all_field_names = set(all_field_names)
            for metadata in self.database_metadata:
                metadata_dict = metadata.to_dict()
                all_field_names.update(metadata_dict.keys())

        self.metadata_field_names = all_field_names
        return all_field_names
 
    def show_metadata_field_names(self):
        print("XASMetadata dataclass fields:")
        predefined_fields = self.get_metadata_field_names()
        for i, field_name in enumerate(predefined_fields, 1):
            print(f"{i:2d}. {field_name}")
        print(f"\nTotal predefined fields: {len(predefined_fields)}")
        
    def get_metadata_as_dataframe(self, metadata_field_name: list = None) -> pd.DataFrame:
        if metadata_field_name is None:
            metadata_field_names = self.get_metadata_field_names(include_dynamic_fields=True)
        
        if self.database_metadata == []:
            raise ValueError("No metadata available to create DataFrame.")

        metadata_rows = []

        for i, metadata in enumerate(self.database_metadata):
            row = {'spectrum_index': i}
            metadata_dict = metadata.to_dict()
            
            # Fill in all possible fields
            for field_name in metadata_field_names:
                row[field_name] = metadata_dict.get(field_name, None)
            
            metadata_rows.append(row)
            
        metadata_df = pd.DataFrame(metadata_rows)

        # Reorder columns for better readability
        preferred_order = ['spectrum_index', 'element', 'compound', 'name', 'edge', 'edge_type', 
                        'oxidation_state', 'coordination_number', 'chemical_name', 'beamline',
                        'measurement_type', 'processed', 'resampled']

        # Get columns in preferred order that actually exist
        ordered_columns = []
        for col in preferred_order:
            if col in metadata_df.columns:
                ordered_columns.append(col)

        # Add remaining columns
        for col in sorted(metadata_df.columns):
            if col not in ordered_columns:
                ordered_columns.append(col)

        metadata_df = metadata_df[ordered_columns]

        # Create DataFrame
        self.metadata_df = metadata_df
        return metadata_df
    
    # ========================================================================
    # XAS RESAMPLING UTILS
    # ========================================================================
    
    def resample_xas_spectra(self, energy_grid: Optional[np.ndarray] = None,
                                    resampling_method: Optional[str] = 'linear_in_energy',
                                    **kwargs) -> None:

        """Wrapper function for spectrum resampling"""
        
        # Log the resampling operation
        parameters = {
            'energy_grid': energy_grid,
            'resampling_method': resampling_method,
            **kwargs
        }
        
        self._resample_xas_spectra(self.spectra, **parameters)
        self._log_processing_step('resample_xas_spectra', parameters)
        self.update_metadata('resampled', True)
        
    def _resample_xas_spectra(self, spectra, 
                              energy_grid: Optional[np.ndarray] = None,
                              resampling_method: Optional[str] = 'linear_in_energy',
                              **kwargs) -> None:
        """
        Resample all spectra onto a new energy grid.
        
        Args:
            spectra: List of spectra to resample.
            energy_grid (np.ndarray, optional): New energy grid (relative to edge).
                                               If None, will be created from kwargs.
            resampling_method (str): Data resampling method ('interpolate', 'downsample', 'bin')
            **kwargs: Parameters for grid creation or resampling:
                - grid_type (str): 'linear_in_energy' or 'linear_in_wavenumber'
                - E_pre_edge, E_post_edge, N_pts: Grid parameters
                - E_post_edge, E_post_edge, N_pts: Grid parameters
                - bin_size, bin_factor: Binning parameters
                - downsample_size, downsample_factor: Downsampling parameters
                - interpolation_method (str): Interpolation method to use (if applicable)
                - resolution (float): Energy resolution for binning/downsampling

        """
        # Create energy grid if not provided
        if energy_grid is None and 'E_pre_edge' in kwargs and 'resolution' not in kwargs:
            grid_type = kwargs.get('grid_type', 'linear_in_energy')
            E_pre_edge = kwargs['E_pre_edge']
            E_post_edge = kwargs['E_post_edge'] 
            N_pts = kwargs['N_pts']
            energy_grid = resampling_utils.create_energy_grid(grid_type, E_pre_edge, E_post_edge, N_pts, **kwargs)
            
        # Process each spectrum
        for spectrum in spectra:
            if resampling_method == 'interpolate':
                resampling_utils._apply_interpolation(spectrum, energy_grid, **kwargs)

            # Apply binning if requested
            if resampling_method == 'bin':
                resampling_utils._apply_binning(spectrum, energy_grid, **kwargs)
            
            # Apply downsampling if requested
            if resampling_method == 'downsample':
                resampling_utils._apply_downsampling(spectrum, energy_grid, **kwargs)
        

        return spectra
    
    # ========================================================================
    # XAS PROCESSING FUNCTIONS
    # ========================================================================
    def process_xas_spectra(self, 
                            remove_nans=False, remove_other_absorption_edges=False,
                            remove_bad_time_instances=False, remove_duplicate_compounds=False,
                            time_pts_keep=None, time_bin_size=None, time_average=False,
                            crop_energy=None, normalize_spectrum=False, center_edges=False,
                            estimate_background=False, remove_glitches=False) -> None:
        """Wrapper function around process_xas_spectra"""
        
        # Log the processing operation
        parameters = {
            'remove_nans' : remove_nans,
            'remove_other_absorption_edges': remove_other_absorption_edges,
            'remove_bad_time_instances': remove_bad_time_instances,
            'remove_duplicate_compounds': remove_duplicate_compounds,
            'time_pts_keep': time_pts_keep,
            'time_bin_size': time_bin_size,
            'time_average': time_average,
            'crop_energy': crop_energy,
            'normalize_spectrum': normalize_spectrum,
            'center_edges': center_edges,
            'estimate_background': estimate_background,
            'remove_glitches': remove_glitches
        }

        self._process_xas_spectra(self.spectra, **parameters)
        self._log_processing_step('process_xas_spectra', parameters)
        self.update_metadata('processed', True)
        
        
    def _process_xas_spectra(self, 
                            spectra,
                            remove_nans=False, remove_other_absorption_edges=False,
                            remove_bad_time_instances=False, remove_duplicate_compounds=False,
                            time_pts_keep=None, time_bin_size=None, time_average=False,
                            crop_energy=None, normalize_spectrum=False, center_edges=False,
                            estimate_background=False, remove_glitches=False) -> None:
        """
        Preprocess a list of XAS spectra with cropping, normalization, and edge centering.
        
        This function performs all the common preprocessing steps for XAS spectra:
        - Time manipulation (binning, cropping, averaging of time instances)
        - Energy cropping (crop spectrum energy to the desired pre/post edge values)
        - Normalization
        - Edge centering (align absorption edges to the middle of the rising absorption edge)
        - Background estimation
        
        Args:
            spectra (list): List of Spectrum objects to process
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
        # Initialize masks if not present
        for spectrum in spectra:        
            if spectrum.glitch_mask is None:
                spectrum.glitch_mask = np.zeros_like(spectrum.energy, dtype=bool)
            if spectrum.data_mask is None:
                spectrum.data_mask = np.ones_like(spectrum.energy, dtype=bool)

        # Remove nans
        if remove_nans:
            processing_utils.remove_nans(spectra)

        # Crop other energy edges
        if remove_other_absorption_edges:
            processing_utils.crop_other_edges(spectra)

        # Remove bad time instances
        if remove_bad_time_instances:
            processing_utils.remove_bad_time_instances(spectra)

        # Find duplicate compounds
        if remove_duplicate_compounds:
            processing_utils.find_duplicate_compounds(spectra, delete=False)

        # Crop time instances
        if time_pts_keep is not None:
            for spectrum in spectra:
                n_times = spectrum.spectrum.shape[1]
                start_idx = n_times // 2 - time_pts_keep // 2
                end_idx = start_idx + time_pts_keep
                spectrum.spectrum = spectrum.spectrum[:, start_idx:end_idx]

        # Bin time instances
        if time_bin_size is not None:
            processing_utils.time_average_spectra(spectra, size=time_bin_size)

        # Average time instances
        if time_average:
            processing_utils.time_average_spectra(spectra)

        # Center edges if requested
        if center_edges:
            processing_utils.center_edges(spectra)

        # Crop energy range
        if crop_energy is not None:
            processing_utils.crop_spectra(spectra, pre_edge=crop_energy[0], post_edge=crop_energy[1])

        # Normalize spectra
        if normalize_spectrum:
            processing_utils.normalize_spectrum(spectra, downsample=10, fit_individual=True)

        # Estimate background
        if estimate_background:
            for spectrum in spectra:
                preprocess_spectrum.estimate_background(spectrum)
        
        # Remove glitches
        if remove_glitches:
            for spectrum in spectra:
                glitch_mask = spectrum.glitch_mask
                if (glitch_mask is not None) and (np.sum(glitch_mask) > 0):
                    print(f'Processing spectrum {spectrum.compound}')
                    preprocess_spectrum.remove_glitches(spectrum, glitch_mask, glitch_fill='interp_avg', crop_edges=True)
                    # preprocess_spectrum.remove_glitches(spectrum, glitch_mask, glitch_fill='delete', crop_edges=True)

        return spectra
    
    # ========================================================================
    # SPLIT XAS SPECTRA INTO ARRAYS/TENSORS
    # ========================================================================
    
    # def create_training_tensors(self, target_db: 'XASDatabase') -> Tuple[np.ndarray, ...]:
    #     """
    #     Create training tensors from this database (noisy) and target database (clean).
        
    #     Args:
    #         target_db: XASDatabase containing clean/target spectra
            
    #     Returns:
    #         Tuple of (energy, noisy_spectra, clean_spectra, metadata)
    #     """
    #     if len(self.spectra) != len(target_db.spectra):
    #         raise ValueError("Training and target databases must have same number of spectra")
        
    #     # Extract data from spectra
    #     energies = []
    #     noisy_data = []
    #     clean_data = []
    #     metadata = []
        
    #     for train_spec, target_spec in zip(self.spectra, target_db.spectra):
    #         energies.append(train_spec.energy)
    #         noisy_data.append(train_spec.spectrum)
    #         clean_data.append(target_spec.spectrum)
    #         metadata.append(train_spec.database_metadata)
        
    #     # Convert to numpy arrays
    #     energies = np.array(energies)
    #     noisy_data = np.array(noisy_data)
    #     clean_data = np.array(clean_data)
        
    #     return energies, noisy_data, clean_data, metadata

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    def plot_spectra(self, key=None, value=None, center_on_edge=False, displace_vertically=True, vertical_displacement_offset=0.1):
        
        if key is not None and value is not None:
            spectra_to_plot = self.get_spectrum(key, value)
        else: 
            spectra_to_plot = self.spectra

        visualization_utils.plot_spectrum(spectra_to_plot, center_on_edge=center_on_edge, displace_vertically=displace_vertically, vertical_displacement_offset=vertical_displacement_offset)

    # ========================================================================
    # XAS DATABASE LOADING/SAVING FUNCTIONS
    # ========================================================================

    def load_from_h5(self, filepath: str, element_to_load=None, compound_to_load=None) -> None:
        """
        Load XAS database from HDF5 file.
        
        This method will load spectra, processing history, and database metadata
        from a previously saved HDF5 file.
        
        Args:
            filepath: Path to the HDF5 file
        """
        # Clear current data
        self.delete_all_spectra()
        
        # Load processing history and database metadata
        try:
            with h5py.File(filepath, 'r') as h5file:
                if 'database_metadata' in h5file:
                    db_metadata_grp = h5file['database_metadata']

                    # Load processing history using pickle
                    if 'processing_history' in db_metadata_grp:
                        pickled_data = db_metadata_grp['processing_history'][()]
                        self.processing_history = pickle.loads(pickled_data)

                    if 'database_metadata' in db_metadata_grp:
                        pickled_metadata = db_metadata_grp['database_metadata'][()]
                        self.database_metadata = pickle.loads(pickled_metadata)
                        
        except Exception as e:
            print(f"Warning: Could not load processing history from {filepath}: {e}")
            print("Continuing with loaded spectra only.")

        
        # Load spectra using existing data_io function
        loaded_spectra = data_io.load_spectra_from_h5(filepath, element=element_to_load, compound=compound_to_load)
        if not isinstance(loaded_spectra, list):
            loaded_spectra = [loaded_spectra]
        
        # Take only a fraction of the databases if only a fraction of elements/compounds was requested
        if element_to_load is not None or compound_to_load is not None:
            print(f"Loaded {len(loaded_spectra)} spectra matching element={element_to_load}, compound={compound_to_load}")
            self.database_metadata = [m for m in self.database_metadata if (element_to_load is None or m.element == element_to_load) and (compound_to_load is None or m.compound == compound_to_load)]
        
        # Loading is asynchronous so the order between database and loaded spectra is not the same
        # Sort accordingly if metadata was loaded
        if self.database_metadata and len(self.database_metadata) == len(loaded_spectra):
            print("Sorting loaded spectra to match database metadata...")
            sorted_metadata = []
            for spectrum, metadata in zip(self.spectra, self.database_metadata):
                if spectrum.compound == metadata.compound:
                    sorted_metadata.append(metadata)
                else:
                    print(f"Warning: No matching spectrum found for metadata {metadata.compound}, {metadata.edge}")
            self.database_metadata = sorted_metadata
        
        # Add spectra to database
        self.add_spectra(loaded_spectra) #, create_metadata=True)
        
            
    def save_to_h5(self, filepath: str, include_processing_history: bool = True) -> None:
        """
        Save XAS database to HDF5 file.
        
        This method saves all spectra, processing history, and database metadata
        to allow full reconstruction of the database state.
        
        Args:
            filepath: Output HDF5 file path
            include_processing_history: Whether to save processing history
        """
        # Save spectra using existing data_io function
        # self.update_spectrum_metadata()
        data_io.save_spectra_to_h5(self.spectra, filepath)
        
        # Add database metadata (processing history, etc.)
        with h5py.File(filepath, 'a') as h5file:  # 'a' for append mode
            # Create database metadata group
            if 'database_metadata' in h5file:
                del h5file['database_metadata']
            db_metadata_grp = h5file.create_group('database_metadata')
            
            # Save database metadata
            pickled_metadata = pickle.dumps(self.database_metadata)
            db_metadata_grp.create_dataset('database_metadata', data=np.void(pickled_metadata))
            
            if include_processing_history and self.processing_history:
                # Save processing history using pickle
                pickled_history = pickle.dumps(self.processing_history)
                db_metadata_grp.create_dataset('processing_history', data=np.void(pickled_history))
            
        print(f"Successfully saved XAS database to {filepath}")
    
    def append_from_h5(self, filepath: str, process: bool = True) -> None:
        """
        Append spectra from another HDF5 database file.
        
        Loads spectra from another database file and optionally applies the same
        processing steps as the current database to maintain consistency.
        
        Args:
            filepath: Path to HDF5 file containing spectra to append
            process: Whether to apply current processing steps to appended spectra
        """
        # Load spectra from file
        loaded_spectra = data_io.load_spectra_from_h5(filepath)
        if not isinstance(loaded_spectra, list):
            loaded_spectra = [loaded_spectra]
        
        # Add to database
        self.add_spectra(loaded_spectra, create_metadata=True)
        
        # Apply processing steps if requested
        if process and self.processing_history:
            print(f"Applying {len(self.processing_history)} processing steps to new spectra...")
            self.process_new_spectra()
        
        print(f"Successfully appended {len(loaded_spectra)} spectra from {filepath}")

    def load_metadata_from_csv(self, filepath: str) -> None:
        """
        Load metadata from CSV file into the database.
        
        This method loads only metadata from a CSV file and creates dummy spectrum objects.
        All entries will have empty spectrum arrays but full metadata.
        
        Args:
            filepath: Path to the CSV metadata file
                         
        Raises:
            FileNotFoundError: If the specified CSV file doesn't exist
        """
        import ast

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Clear current data
        self.delete_all_spectra()
        
        # Load CSV data
        df = pd.read_csv(filepath)
        
        if df.empty:
            print("CSV file is empty. No data loaded.")
            return
        
        print(f"Loading metadata for {len(df)} spectra from {filepath}")
        
        # Process each row to create metadata objects
        for idx, row in df.iterrows():
            try:
                # Create XASMetadata object from row data
                metadata_dict = {}
                
                # Convert row to dict and filter out NaN values
                for key, value in row.items():
                    if pd.notna(value) and value != '':
                        # Handle special cases for certain fields
                        if key in ['edge', 'oxidation_state', 'coordination_number', 
                                  'pre_edge_min_E', 'pre_edge_max_E', 'post_edge_min_E', 'post_edge_max_E',
                                  'edge_theoretical']:
                            try:
                                metadata_dict[key] = float(value)
                            except (ValueError, TypeError):
                                metadata_dict[key] = value
                        elif key in ['processed', 'resampled']:
                            metadata_dict[key] = bool(value) if value in [True, False, 'True', 'False'] else value
                        elif key in ['glitches']:
                            lst = ast.literal_eval(value) if isinstance(value, str) else value
                            metadata_dict[key] = lst
                        else:
                            metadata_dict[key] = value
                
                # Create metadata object
                metadata = XASMetadata(**{k: v for k, v in metadata_dict.items() 
                                        if k in [f.name for f in dataclass_fields(XASMetadata)]})
                
                # Add any dynamic fields not in the dataclass
                predefined_fields = {f.name for f in dataclass_fields(XASMetadata)}
                for key, value in metadata_dict.items():
                    if key not in predefined_fields:
                        setattr(metadata, key, value)
                
                # Add metadata and dummy spectrum
                self.database_metadata.append(metadata)
                dummy_spectrum = Spectrum(np.array([]), np.array([]), metadata=metadata.to_dict())
                self.spectra.append(dummy_spectrum)
                    
            except Exception as e:
                print(f"Warning: Failed to process row {idx}: {e}")
                continue
        
        print(f"Successfully loaded metadata for {len(self.database_metadata)} spectra from {filepath}")

    def load_spectra_into_database(self, filter_key: str = None, filter_value: str = None) -> None:
        """
        Load spectrum data from file paths stored in metadata.
        
        This method loads actual spectrum data using file paths stored in the metadata.
        It supports filtering to load only spectra that match specific criteria.
        
        Args:
            filter_key: Metadata key to filter by (e.g., 'element', 'compound', 'edge_type')
            filter_value: Value to filter by (e.g., 'Fe', 'Fe2O3', 'K')
                         
        Example:
            >>> # Load all spectra
            >>> db.load_spectra_into_database()
            >>> 
            >>> # Load only Fe spectra
            >>> db.load_spectra_into_database(filter_key='element', filter_value='Fe')
            >>> 
            >>> # Load only K-edge spectra
            >>> db.load_spectra_into_database(filter_key='edge_type', filter_value='K')
        """
        if not self.database_metadata:
            print("No metadata available. Load metadata first using load_metadata_from_csv().")
            return
        
        # Collect spectrum paths and metadata for loading
        spectrum_paths = []
        I0_paths = []
        I1_paths = []
        selected_metadata = []
        selected_indices = []
        
        print("Collecting spectrum paths and applying filters...")
        
        for idx, metadata in enumerate(self.database_metadata):
            # Apply filter if specified
            if filter_key is not None and filter_value is not None:
                metadata_value = getattr(metadata, filter_key, None)
                if metadata_value != filter_value:
                    continue
            
            # Check if spectrum file paths are available
            spectrum_path = getattr(metadata, 'path_spectrum', None)
            if spectrum_path:
                spectrum_paths.append(spectrum_path)
                I0_paths.append(getattr(metadata, 'path_I0', None))
                I1_paths.append(getattr(metadata, 'path_I1', None))
                selected_metadata.append(metadata.to_dict())
                selected_indices.append(idx)
        
        if not spectrum_paths:
            if filter_key and filter_value:
                print(f"No spectra found matching {filter_key}='{filter_value}' with valid file paths.")
            else:
                print("No spectra found with valid file paths.")
            return
        
        if filter_key and filter_value:
            print(f"Loading {len(spectrum_paths)} spectra matching {filter_key}='{filter_value}'")
        else:
            print(f"Loading {len(spectrum_paths)} spectra")
        
        # Use parallelized loading from data_io
        self.spectra = data_io.load_spectra(spectrum_paths, selected_metadata, I0_paths, I1_paths)
        # Construct the database metadata for the selected spectra
        self.database_metadata = [XASMetadata().from_dict(s.metadata) for s in self.spectra]

        if filter_key and filter_value:
            print(f"Successfully loaded {len(self.spectra)} spectra matching {filter_key}='{filter_value}'")
        else:
            print(f"Successfully loaded {len(self.spectra)} spectra")

        print(f"Database now contains {len([s for s in self.spectra if len(s.energy) > 0])} spectra with data")
        
    def export_all_metadata_fields_to_csv(self, filepath: str, export_all_fields: bool = False) -> None:
        """
        Export all metadata fields (as defined in XASMetadata dataclass) for every spectrum to a CSV file.
        
        This method creates a comprehensive CSV with one row per spectrum and columns for all
        possible metadata fields defined in the XASMetadata dataclass, including both predefined
        and dynamically added fields.
        
        Args:
            filepath: Path for the output CSV file
        """
        if not self.database_metadata:
            print("No metadata available to export. Database metadata is empty.")
            return
        
        # Get all possible metadata field names from the dataclass and dynamic fields
        from dataclasses import fields as dataclass_fields
        
        # Start with predefined fields from XASMetadata dataclass
        predefined_fields = {f.name for f in dataclass_fields(XASMetadata)}
        
        # Collect all possible field names (predefined + dynamic)
        all_field_names = set(predefined_fields)
        if export_all_fields:
            for metadata in self.database_metadata:
                metadata_dict = metadata.to_dict()
                all_field_names.update(metadata_dict.keys())
        
        # Sort field names for consistent column ordering
        sorted_field_names = sorted(all_field_names)
        
        # Create rows for CSV
        csv_data = []
        for i, metadata in enumerate(self.database_metadata):
            row = {'spectrum_index': i}
            metadata_dict = metadata.to_dict()
            
            # Fill in all possible fields
            for field_name in sorted_field_names:
                row[field_name] = metadata_dict.get(field_name, None)
            
            csv_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(csv_data)
        
        # Reorder columns to put spectrum_index first, then element, compound, etc.
        preferred_order = ['spectrum_index', 'element', 'compound', 'name', 'edge', 'edge_type', 
                          'oxidation_state', 'coordination_number', 'chemical_name']
        
        # Get columns in preferred order that actually exist, then add remaining columns
        columns = []
        for col in preferred_order:
            if col in df.columns:
                columns.append(col)
        
        # Add remaining columns
        for col in sorted(df.columns):
            if col not in columns:
                columns.append(col)
        
        df = df[columns]
        df.to_csv(filepath, index=False)
        
        print(f"Exported metadata for {len(csv_data)} spectra to {filepath}")
        print(f"Exported {len(sorted_field_names)} metadata fields: {', '.join(sorted_field_names[:10])}{'...' if len(sorted_field_names) > 10 else ''}")
    
    # ========================================================================
    # GLITCH MASK OPERATIONS
    # ========================================================================
    def create_glitch_mask_from_metadata(self) -> None:
        """ 
        XASMetadata.glitches contains a list of (start_E, end_E) tuples for glitches.
        This function creates boolean glitch masks for all spectra based on this metadata.
        """
        if self.spectra is None:
            print("No spectra in database. Load spectra first.")
            return
        
        for spectrum, metadata in zip(self.spectra, self.database_metadata):
            if metadata.glitches and spectrum.energy is not None and len(spectrum.energy) > 0:
                glitch_mask = np.zeros_like(spectrum.energy, dtype=bool)

                for glitch_energy in metadata.glitches:
                    start_E, end_E = glitch_energy
                    glitch_mask |= (spectrum.energy >= start_E) & (spectrum.energy <= end_E)
                spectrum.glitch_mask = glitch_mask
            else:
                spectrum.glitch_mask = np.zeros_like(spectrum.energy, dtype=bool)
    
    def create_glitch_metadata_from_masks(self) -> None:
        """ 
        Create XASMetadata.glitches list of (start_E, end_E) tuples from boolean glitch masks.
        """
        from scipy.ndimage import label

        if self.spectra is None:
            print("No spectra in database. Load spectra first.")
            return
        
        for spectrum, metadata in zip(self.spectra, self.database_metadata):
            glitch_mask = spectrum.glitch_mask
            if glitch_mask is not None and len(glitch_mask) == len(spectrum.energy):
                # Label the continuous regions
                if np.sum(glitch_mask) == 0:
                    return []
                else:
                    labeled_array, num_features = label(glitch_mask)
                    # Extract start and end indices for each labeled region
                    regions = []
                    for label_id in range(1, num_features + 1):  # Labels start from 1
                        indices = np.where(labeled_array == label_id)[0]
                        start, end = indices[0], indices[-1]
                        
                        regions.append((spectrum.energy[start], spectrum.energy[end]))
            
            # update glitch metadata            
            metadata.glitches = regions

                             
    # ========================================================================
    # INTERACTIVE GUI
    # ========================================================================
    
    def launch_gui(self) -> None:
        """
        Launch interactive GUI for spectrum processing and glitch mask creation.
        
        This method creates an interactive interface that allows you to:
        - Browse and visualize spectra in the database
        - Create and edit glitch masks interactively
        - Export glitch masks for individual spectra or the entire database
        - Perform normalization updates on selected spectra
        - Filter spectra by element, compound, or other metadata
        
        The GUI operates directly on self.spectra and updates the database in-place.
        
        Example:
            >>> db = XASDatabase()
            >>> db.load_from_csv("metadata.csv", load_spectra=True)
            >>> db.launch_gui()  # Opens interactive interface
            
        Features:
            - Interactive plotting with click-to-add glitch regions
            - Automatic glitch detection with adjustable thresholds
            - Apply glitch masks to individual spectra or all spectra
            - Export glitch masks to various formats
            - Real-time normalization parameter adjustment
            - Filter and browse by metadata fields
        """
        if not self.spectra:
            print("No spectra loaded in database. Load spectra first using load_from_csv() or add_spectrum().")
            return
        
        # Import GUI components
        from xasdenoise.xas_database.xas_database_gui import InteractiveDataProcessing
        
        # Launch the GUI with the current database
        self.gui = InteractiveDataProcessing(self.spectra)
        return self.gui