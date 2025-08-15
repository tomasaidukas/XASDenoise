"""
Class to create a Spectrum object which is used for X-ray absorption spectroscopy data analysis.
"""

import numpy as np
from typing import Optional, Dict, Union
from scipy.ndimage import convolve1d, uniform_filter1d
from scipy.spatial import KDTree
import warnings


class Spectrum:
    def __init__(
        self,
        energy: np.ndarray,
        spectrum: np.ndarray,
        metadata: Optional[Dict] = {},
        I0: Optional[np.ndarray] = None,
        I1: Optional[np.ndarray] = None,
        background: Optional[np.ndarray] = None,
        glitch_mask: Optional[np.ndarray] = None,
        data_mask: Optional[np.ndarray] = None,
    ):
        """
        Initialize a Spectrum object.

        Args:
            energy (np.ndarray): 1D array of energy values in eV.
            spectrum (np.ndarray): 2D array of spectrum values vs. time.
            metadata (dict, optional): Dictionary of spectrum metadata.
            I0 (np.ndarray, optional): 1D or 2D array of I0 values.
            I1 (np.ndarray, optional): 1D or 2D array of I1 values.
            background (np.ndarray, optional): 1D or 2D array of background values.
            glitch_mask (np.ndarray, optional): Boolean mask for glitches.
            data_mask (np.ndarray, optional): Boolean mask for data.
        """
        # Initialize new_arrays which will store additional attributes allocated dynamically when needed
        object.__setattr__(self, "new_arrays", {})
        object.__setattr__(self, "y_arrays", ["spectrum", "I0", "I1", "background", "glitch_mask", "data_mask"])
        
        # x-grid arrays
        self.energy = self._ensure1D(energy)
        
        # y-grid arrays
        self.spectrum = self._ensure2D(spectrum)
        self.I0 = self._ensure2D(I0)
        self.I1 = self._ensure2D(I1)
        self.background = self._ensure2D(background)

        if glitch_mask is None:
            self.glitch_mask = np.zeros_like(self.energy, dtype=bool)
        else:
            self.glitch_mask = glitch_mask

        if data_mask is None:
            self.data_mask = np.ones_like(self.energy, dtype=bool)
        else:
            self.data_mask = data_mask
        
        # Metadata dictionary
        self.metadata = metadata
        
        # Populate metadata using xraydb if available
        self.populate_metadata()
        
        # # Compute background if needed
        # if self.background is None:
        #     self.compute_background()

    def __setattr__(self, name, value):
        """
        Custom attribute setting:
        - If name is in predefined attributes, store normally.
        - Otherwise, store in `new_arrays`.
        """
        predefined_attrs = ["energy", "spectrum", "I0", "I1", "background", "glitch_mask", "data_mask", "metadata", "new_arrays", "y_arrays"]
        if name in predefined_attrs:
            super().__setattr__(name, value)
        else:
            self.new_arrays[name] = value  # Store extra arrays dynamically

    def __getattr__(self, name):
        """
        If an attribute isn't found in the object, check `new_arrays`.
        """
        if "new_arrays" in self.__dict__ and name in self.new_arrays:
            return self.new_arrays[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self) -> str:
        """
        Return a string representation of the Spectrum object.

        Returns:
            str: A string showing the compound name, if available.
        """
        return f"Spectrum({self.compound or 'Unnamed'})"

    def copy(self) -> "Spectrum":
        """
        Create a deep copy of the Spectrum object.

        Returns:
            Spectrum: A new instance with copied attributes.
        """
        return Spectrum(
            self.energy.copy(),
            self.spectrum.copy(),
            self.metadata.copy(),
            I0=self.I0.copy() if self.I0 is not None else None,
            I1=self.I1.copy() if self.I1 is not None else None,
            background=self.background.copy() if self.background is not None else None,
            glitch_mask=self.glitch_mask.copy() if self.glitch_mask is not None else None,
            data_mask=self.data_mask.copy() if self.data_mask is not None else None,
        )

    def _ensure2D(self, array: np.ndarray) -> np.ndarray:
        """
        Ensure the input array is 2D.

        Args:
            array (np.ndarray): Input array.

        Returns:
            np.ndarray: 2D array.
        """
        if array is None:
            return None
        
        if array.ndim == 1:
            array = array[:, np.newaxis]
        return array

    def _ensure1D(self, array: np.ndarray) -> np.ndarray:
        """
        Ensure the input array is 1D.

        Args:
            array (np.ndarray): Input array.

        Returns:
            np.ndarray: 1D array.
        """
        if array is None:
            return None
        
        if array.ndim > 1:
            array = array[:, 0]
            warnings.warn("Flattened array to 1D.")
        
        return array
    def populate_metadata(self) -> None:
        """
        Populate metadata using the xraydb library, if installed.
        """
        try:
            with warnings.catch_warnings():
                import xraydb
                element, edge_type = xraydb.guess_edge(self.edge, edges=["K", "L3", "L2", "L1"])
                if self.metadata.get("element", None) is None:
                    self.metadata["element"] = element
                if self.metadata.get("edge_type", None) is None:
                    self.metadata["edge_type"] = edge_type
                if self.metadata.get("edge", None) is None:
                    self.metadata["edge"] = xraydb.xray_edge(element, edge_type).energy
                self.metadata["edge_theoretical"] = xraydb.xray_edge(element, edge_type).energy
                self.metadata["all_edges"] =  [v.energy for v in xraydb.xray_edges(element).values()]
        except:
            print("Cannot populate metadata from xraydb")

    def compute_mu(self) -> None:
        """
        Calculates the absorption coefficient (mu) using I0 and I1.

        Returns:
            np.ndarray: 1D array of the absorption coefficient.
        """
        if self.I0 is None or self.I1 is None:
            raise ValueError("Cannot compute mu: I0 or I1 is missing.")
        self.spectrum = np.log(self.I0 / (self.I1 + 1e-9))
        return self.spectrum

    def compute_background(self) -> None:
        """
        Estimates the background of the spectrum.
        
        Returns:
            np.ndarray: 1D array of the estimated background.
        """
        from xasdenoise.xas_data.preprocess_spectrum import estimate_background
        self.background = estimate_background(self)
        return self.background
    
    def delete_energy_indices(self, indices: np.ndarray):
        """
        Delete specified energy indices.

        Args:
            indices (np.ndarray): Indices to delete along the energy axis.
        """             
        self.energy = np.delete(self.energy, indices)
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None:
                setattr(self, attr, np.delete(array, indices, axis=0))        
        
        # also apply to "new_arrays" if they exist
        for attr in self.new_arrays:
            array = getattr(self, attr)
            if array is not None:
                setattr(self, attr, np.delete(array, indices, axis=0))
                
        # Adjust pre-edge and post-edge regions if defined
        if hasattr(self.metadata, 'pre_edge_min_E') and (self.metadata["pre_edge_min_E"] < self.energy[0]):
            self.metadata["pre_edge_min_E"] = self.energy[0]
        if hasattr(self.metadata, 'pre_edge_max_E') and (self.metadata["pre_edge_max_E"] > self.energy[-1]):
            self.metadata["pre_edge_max_E"] = self.edge - 30 if (self.edge-30) > self.energy[0] else self.energy[10]
        if hasattr(self.metadata, 'post_edge_min_E') and (self.metadata["post_edge_min_E"] < self.energy[0]):
            self.metadata["post_edge_min_E"] = self.edge + 30 if (self.edge+30) < self.energy[-1] else self.energy[-10]
        if hasattr(self.metadata, 'post_edge_max_E') and (self.metadata["post_edge_max_E"] > self.energy[-1]):
            self.metadata["post_edge_max_E"] = self.energy[-1]
            
    def delete_time_indices(self, indices: np.ndarray):
        """
        Delete specified time indices.

        Args:
            indices (np.ndarray): Indices to delete along the time axis.
        """
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None and array.ndim > 1:
                setattr(self, attr, np.delete(array, indices, axis=1))

        # also apply to "new_arrays" if they exist
        for attr in self.new_arrays:
            array = getattr(self, attr)
            if array is not None and array.ndim > 1:
                setattr(self, attr, np.delete(array, indices, axis=1))
                
    def crop_energy_indices(self, indices: np.ndarray):
        """
        Crop the spectrum to a specific range of energy indices.

        Args:
            indices (np.ndarray): Indices to keep along the energy axis.
        """
        self.energy = self.energy[indices]
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None:
                if array.ndim == 1:  # 1D array
                    setattr(self, attr, array[indices])
                elif array.ndim == 2:  # 2D array
                    setattr(self, attr, array[indices, :])
        
        # also crop the "new_arrays" if they exist
        for attr in self.new_arrays:
            array = getattr(self, attr)
            if array is not None:
                if array.ndim == 1:
                    setattr(self, attr, array[indices])
                elif array.ndim == 2:
                    setattr(self, attr, array[indices, :])
                    
        # Adjust pre-edge and post-edge regions if defined
        if hasattr(self.metadata, 'pre_edge_min_E') and (self.metadata["pre_edge_min_E"] < self.energy[0]):
            self.metadata["pre_edge_min_E"] = self.energy[0]
        if hasattr(self.metadata, 'pre_edge_max_E') and (self.metadata["pre_edge_max_E"] > self.energy[-1]):
            self.metadata["pre_edge_max_E"] = self.edge - 30 if (self.edge-30) > self.energy[0] else self.energy[10]
        if hasattr(self.metadata, 'post_edge_min_E') and (self.metadata["post_edge_min_E"] < self.energy[0]):
            self.metadata["post_edge_min_E"] = self.edge + 30 if (self.edge+30) < self.energy[-1] else self.energy[-10]
        if hasattr(self.metadata, 'post_edge_max_E') and (self.metadata["post_edge_max_E"] > self.energy[-1]):
            self.metadata["post_edge_max_E"] = self.energy[-1]
                
    def crop_time_indices(self, indices: np.ndarray):
        """
        Crop the spectrum to a specific range of time indices.

        Args:
            indices (np.ndarray): Indices to keep along the time axis.
        """
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None and array.ndim > 1 and array.shape[1] > 1:
                setattr(self, attr, array[:, indices])
        
        # also crop the "new_arrays" if they exist
        for attr in self.new_arrays:
            array = getattr(self, attr)
            if array is not None and array.ndim > 1 and array.shape[1] > 1:
                setattr(self, attr, array[:, indices])
                
    def pad_data(self, pad_width_start: int = 0, pad_width_end: int = 0):
        """
        Pad the spectrum data with zeros.

        Args:
            pad_width (int): Number of points to pad on each side.
        """
        self.energy = np.pad(self.energy, (pad_width_start, pad_width_end), mode='edge')
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None:
                if array.ndim == 1:
                    setattr(self, attr, np.pad(array, (pad_width_start, pad_width_end), mode='edge'))
                elif array.ndim == 2:
                    setattr(self, attr, np.pad(array, ((pad_width_start, pad_width_end), (0, 0)), mode='edge'))

        # also apply to "new_arrays" if they exist
        for attr in self.new_arrays:
            array = getattr(self, attr)
            if array is not None:
                if array.ndim == 1:
                    setattr(self, attr, np.pad(array, (pad_width_start, pad_width_end), mode='edge'))
                elif array.ndim == 2:
                    setattr(self, attr, np.pad(array, ((pad_width_start, pad_width_end), (0, 0)), mode='edge'))
    
    def bin_time_instances(self, size: int):
        """
        Bin the spectrum data by averaging over time instances.
        
        Args:
            size (int): Number of time instances to average over.            
        """                   
        def _bin_time(data, w):
            win = np.ones(w) / w
            return convolve1d(data, win, mode='reflect', axis=1)        
        
        # Bin all y-grid array time instances
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None:
                if array.ndim == 2 and array.shape[1] > 1:
                    max_size = (array.shape[1] // size) * size
                    binned = _bin_time(array, size)[:, 0:max_size:size]
                    setattr(self, attr, binned)
        
    def _interpolate(self, x: np.ndarray, y: np.ndarray, x_new: np.ndarray, method: str = "pchip") -> np.ndarray:
        """
        Interpolate y-values to a new x-grid.
        
        Args:
            x (np.ndarray): Original x-values.
            y (np.ndarray): Original y-values.
            x_new (np.ndarray): New x-values.
            method (str): Interpolation method. Defaults to 'linear'.

        Returns:
            np.ndarray: Interpolated spectrum data.
        """
        from scipy.interpolate import interp1d, PchipInterpolator, UnivariateSpline

        # Handle boolean arrays (e.g., masks)
        if y.dtype == bool:
            method = 'nearest'
            bounds = (y[0], y[-1]) if len(y.shape) == 1 else (y[0, :], y[-1, :])
            f = interp1d(x, y, kind=method, bounds_error=False, axis=0, fill_value=bounds)
            return (f(x_new) >= 0.5).astype(bool)
        
        # Handle standard interpolation methods
        elif method in ['nearest', 'zero', 'linear', 'slinear', 'quadratic', 'cubic']:
            bounds = (y[0], y[-1]) if len(y.shape) == 1 else (y[0, :], y[-1, :])
            f = interp1d(x, y, kind=method, bounds_error=False, axis=0, fill_value=bounds)
            return f(x_new)
        
        # PCHIP interpolation
        elif method == "pchip":
            f = PchipInterpolator(x, y, axis=0)
            return f(x_new)

        # Spline interpolation
        elif method == "spline":
            y_interp = np.zeros((len(x_new), y.shape[1])) if y.ndim > 1 else np.zeros(len(x_new))
            for t in range(y.shape[1] if y.ndim > 1 else 1):
                if y.ndim > 1:
                    spline = UnivariateSpline(x, y[:, t], s=0, k=3)
                    y_interp[:, t] = spline(x_new)
                else:
                    spline = UnivariateSpline(x, y, s=0, k=3)
                    y_interp = spline(x_new)
            return y_interp

        # Unsupported method
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
    def interpolate_spectrum(self, new_energy: np.ndarray, method: str = "pchip", mask: Optional[np.ndarray] = None):
        """
        Interpolate all y_arrays to a new energy grid.

        Args:
            new_energy (np.ndarray): The new energy grid to interpolate to.
            method (str, optional): The interpolation method. Defaults to 'cubic'.
            mask (np.ndarray, optional): Boolean mask defining what data is used for inteprolation. Defaults to None.
        """        
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None:
                # Extract the arrays used for interpolation and extrapolation
                x = self.energy if mask is None else self.energy[mask]
                if array.ndim > 1:
                    y = array if mask is None else array[mask, :]
                else:
                    y = array if mask is None else array[mask]

                x_new = new_energy
                
                # Interpolate array and store it in "self"
                y_new = self._interpolate(x, y, x_new, method)                            
                setattr(self, attr, y_new)

        # Update the energy grid
        self.energy = new_energy
                
        # Update metadata arrays relevant to data normalization and background estimation
        try:
            if self.metadata["pre_edge_min_E"] < self.energy[0]:
                self.metadata["pre_edge_min_E"] = self.energy[0]
            if self.metadata["post_edge_max_E"] > self.energy[-1]:
                self.metadata["post_edge_max_E"] = self.energy[-1]
            if self.metadata["pre_edge_max_E"] - self.metadata["pre_edge_min_E"] < 10:
                self.metadata["pre_edge_max_E"] = self.edge - 30
            if self.metadata["post_edge_max_E"] - self.metadata["post_edge_min_E"] < 10:
                self.metadata["post_edge_min_E"] = self.edge + 30
        except:
            pass       
        
    
    def bin_spectrum(self, 
                     size: Optional[int] = None, 
                     factor: Optional[int] = None, 
                     resolution: Optional[int] = None,
                     energy_grid_type: str = 'linear_in_energy'):
        """
        Bin the spectrum by averaging neighboring points.

        Args:
            size (int, optional): The desired size of the binned spectrum.
            factor (int, optional): The binning factor for the spectrum.
            resolution (int, optional): The desired resolution of the binned spectrum.
            energy_grid_type (str, optional): The type of energy grid to use at the pre-edge 'linear_in_energy' or 'linear_in_wavenumber'. Defaults to 'linear_in_energy'.
            
        Raises:
            ValueError: If neither size nor factor is specified.
        """        
        # Binning method which preserves correlations between neighboring points
        # by applying a moving average filter and then binning the result
        def _bin_spectrum(data, w):
            win = np.ones(w) / w
            return convolve1d(data, win, mode='reflect', axis=0)
        # Binning method which averages neighboring points without overlap
        # def _bin_spectrum(data, w):
        #     return np.mean(array.reshape(new_size, factor), axis=1)
        
        if factor is None and size is None:
            raise ValueError("Either factor or size must be specified.")
                
        if size is None and factor is not None:
            size = len(self.energy) // factor
        
        if size is None and resolution is not None:
            energy_range = self.energy[-1] - self.energy[0]
            size = int(energy_range / resolution) + 1
            
        if size >= len(self.energy):
            print("Data is smaller than the desired downsampling size, returning the original data.")
            return 
        
        # Determine the binning factor
        if factor is None:
            factor = len(self.energy) // size   

        # once we have the number of points we can generate the sampling indices which will downsample the
        # spectrum onto the desired energy grid
        if energy_grid_type == 'linear_in_wavenumber':
            new_energy_grid = np.linspace(self.energy[0], self.energy[-1], size)
            new_energy_grid = self.get_energy_linear_in_wavenumber(new_energy_grid, pre_edge='linear_wavenumber')
            indices = self.find_matching_indices_between_two_energy_grids(self.energy, new_energy_grid)
            
        elif energy_grid_type == 'linear_in_energy':
            new_energy_grid = np.linspace(self.energy[0], self.energy[-1], size)
            indices = self.find_matching_indices_between_two_energy_grids(self.energy, new_energy_grid)
            
        # Bin energy
        self.energy = _bin_spectrum(self.energy, factor)[indices]

        # Bin all y-grid arrays
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None:
                if array.ndim == 1:
                    setattr(self, attr, _bin_spectrum(array, factor)[indices])
                elif array.ndim == 2:
                    setattr(self, attr, _bin_spectrum(array, factor)[indices, :])
    
    
    def downsample_spectrum(self, 
                            size: Optional[int] = None, 
                            factor: Optional[int] = None, 
                            resolution: Optional[int] = None,
                            energy_grid_type: Optional[str] = 'linear_in_energy',
                            new_energy_grid: Optional[np.ndarray] = None):
        """
        Downsample the spectrum by selecting every nth point.
        
        Args:
            size (int, optional): The desired size of the downsampled spectrum.
            factor (int, optional): The downsampling factor for the spectrum.
            resolution (int, optional): The desired resolution of the downsampled spectrum.
            energy_grid_type (str, optional): The type of energy grid to use at the pre-edge 'linear_in_energy' or 'linear_in_wavenumber'. Defaults to 'linear_in_energy'.
            new_energy_grid (np.ndarray, optional): The new energy grid to downsample to. If None, will be generated based on size, factor, or resolution.
            
        Raises:
            ValueError: If neither size nor factor is specified.
        """
        
        # Create a new energy grid based on the specified parameters
        if new_energy_grid is None:
            if factor is None and size is None and resolution is None:
                raise ValueError("Either factor or size or resolution must be specified.")
        
            if size is None and factor is not None:
                size = len(self.energy) // factor
                print(f"Data will be downsampled to {size} points.")
                
            if size is None and resolution is not None:
                energy_range = self.energy[-1] - self.energy[0]
                size = int(energy_range / resolution) + 1
                print(f"Data will be downsampled to {size} points. Target energy resolution is {resolution} eV.")
                
            if size >= len(self.energy):
                print("Data is smaller than the desired downsampling size, returning the original data.")
                return 
            
            if energy_grid_type == 'linear_in_wavenumber':
                new_energy_grid = np.linspace(self.energy[0], self.energy[-1], size)
                new_energy_grid = self.get_energy_linear_in_wavenumber(new_energy_grid, pre_edge='linear_wavenumber')
                
            elif energy_grid_type == 'linear_in_energy':
                new_energy_grid = np.linspace(self.energy[0], self.energy[-1], size)
        
        # Find the downsampling indices that would map the original energy grid to the new energy grid
        if new_energy_grid is not None:
            indices = self.find_matching_indices_between_two_energy_grids(self.energy, new_energy_grid)
        else:
            raise ValueError("Either new_energy_grid or energy_grid_type must be specified.")

        # Downsample the energy grid
        self.energy = self.energy[indices]
        
        # Downsample all y-grid arrays
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None:
                if array.ndim == 1:
                    setattr(self, attr, array[indices])
                elif array.ndim == 2:
                    setattr(self, attr, array[indices, :])
    
    def bin_spectrum_onto_grid(self, new_energy_grid: np.ndarray):
        def _bin_func(x, y, x_new):
            """
            Interpolate data onto a new grid using binning.
            
            Args:
                x (array): Original grid.
                y (array): Original data.
                x_new (array): New grid.
                
            Returns:
                array: Interpolated data.
            """
            dtype = y.dtype
        
            # Vectorize the binning operation
            df = np.diff(x_new).max()
            x_new_bins = np.concatenate([[x_new[0] - df], (x_new[1:] + x_new[:-1]) / 2, [x_new[-1] + df]])
            digitized = np.digitize(x, x_new_bins)
            
            # Preallocate arrays
            x_avg = np.zeros(len(x_new)-1)
            if y.ndim > 1:
                y_avg = np.zeros((len(x_new)-1, y.shape[1]), dtype=dtype)
            else:
                y_avg = np.zeros(len(x_new)-1, dtype=dtype)
            
            # Average data within downsampling bins
            valid_bins = 0
            for i in range(1, len(x_new)):
                indices = np.where(digitized == i)[0]
                if len(indices) > 0:
                    x_avg[valid_bins] = np.mean(x[indices])
                    if y.ndim > 1:
                        y_avg[valid_bins, :] = np.mean(y[indices, :], axis=0)
                    else:
                        y_avg[valid_bins] = np.mean(y[indices])
                    valid_bins += 1
                    
            # Trim arrays to only include valid bins
            x_avg = x_avg[:valid_bins]
            if y.ndim > 1:
                y_avg = y_avg[:valid_bins, :]
            else:
                y_avg = y_avg[:valid_bins]         
            return x_avg, y_avg
        
        # Bin all y-grid arrays        
        for attr in self.y_arrays:
            array = getattr(self, attr)
            if array is not None:
                new_energy, y = _bin_func(self.energy, array, new_energy_grid)
                setattr(self, attr, y)
        self.energy = new_energy
        
    def find_matching_indices_between_two_energy_grids(self, energy, new_energy) -> np.ndarray:
        """
        Find indices in the original energy grid that closely match the new energy grid.
        
        Args:
            indices (np.ndarray): Energy indices to be converted.
            
        Returns:
            np.ndarray: Indices for the energy grid linear in k-space.
        """
        num_pts = len(new_energy)
        
        # find the most closely matching energy values of "new_energy" with respect to initial "energy" grid
        tree = KDTree(energy.reshape(-1, 1))
        _, indices = tree.query(new_energy.reshape(-1, 1))  # Find closest original indices
        
        # ensure unique to avoid duplicates
        unique_indices = np.unique(indices)
        
        # if len(unique_indices) < num_pts:
        #     missing_count = num_pts - len(unique_indices)
        #     available_indices = np.setdiff1d(np.arange(num_pts), unique_indices)  # Find unused indices
        #     rng = np.random.default_rng(seed=42)
        #     additional_indices = rng.choice(available_indices, size=missing_count, replace=False)
        #     unique_indices = np.concatenate([unique_indices, additional_indices])
        unique_indices = np.sort(unique_indices)
        return unique_indices
                 
    def get_energy_linear_in_wavenumber(self, energy=None, pre_edge='linear_energy'):
        """
        Generates an energy grid which is linear in k-space (and non linear in energy) 
        at the post-edge region and linear in energy at the pre-edge.
        
        Args:
            energy (np.ndarray, optional): The energy grid to use. Defaults to None.
            pre_edge (str, optional): The type of energy grid to use at the pre-edge 'linear_energy' or 'linear_wavenumber'. Defaults to 'linear_energy'.
        Returns:
            np.ndarray: The energy grid linear in k-space.
        """
        if energy is None:
            energy = self.energy
        
        if pre_edge == 'linear_energy':
            idx = energy > self.edge
            dE = energy[idx] - self.edge
            
            # get wavenumber
            k = np.sqrt(0.2625 * dE)
            # make k linear
            new_k = np.linspace(k[0], k[-1], len(k))
            # obtain non-linear energy  grid
            new_energy = energy.copy()
            new_energy[idx] = new_k**2 / 0.2625 + self.edge
            
        elif pre_edge == 'linear_wavenumber':
            dE = energy - self.edge
            # get wavenumber
            k = np.sign(dE) * np.sqrt(0.2625 * abs(dE))
            # make k linear
            new_k = np.linspace(k[0], k[-1], len(k))
            # obtain non-linear energy  grid
            new_energy =  np.sign(new_k) * new_k**2 / 0.2625 + self.edge
        return new_energy
    
    def ksquared_weights(self, pre_edge=None) -> np.ndarray:
        """
        Returns the exafs weighting function using k-space squared scaling.

        Args:
            pre_edge (str, optional): The type of pre-edge scaling to use.
            
        Returns:
            np.ndarray: The energy grid in k-space.
        """
        if pre_edge == 'ones':
            # keep pre-edge weights at 1
            weights = np.ones_like(self.energy)
            weights[self.wavenumber_idx] = 1 + abs(self.wavenumber)**2
        else:
            dE = self.energy - self.edge
            weights = np.sign(dE) * np.sqrt(0.2625 * abs(dE)) 
            weights = 1+ abs(weights)**2
        return weights
    
    @property
    def wavenumber_idx(self) -> np.ndarray:
        """
        Wavenumber is only valid for energy values above the edge.
        This function returns the indices of the energy values above the edge.
        
        Returns:
            np.ndarray: Indices of the energy values above the edge.
        """
        return np.where(self.energy > self.edge)[0]
    
    @property
    def wavenumber(self) -> np.ndarray:
        """
        Calculates the wavenumber from the energy.

        Returns:
            np.ndarray: 1D array of the wavenumber.
        """
        # Calculate energy difference relative to the edge
        idx = self.energy > self.edge
        dE = self.energy[idx] - self.edge
        return np.sqrt(0.2625 * dE) 
    
    @property
    def time_averaged_spectrum(self) -> np.ndarray:
        """
        Calculates the time-averaged spectrum.

        Returns:
            np.ndarray: 1D array of the time-averaged spectrum.
        """
        return np.mean(self.spectrum, axis=1) if self.spectrum.ndim > 1 else self.spectrum

    @property
    def time_averaged_I0(self) -> Optional[np.ndarray]:
        """
        Calculates the time-averaged I0 signal.

        Returns:
            Optional[np.ndarray]: 1D array of the time-averaged I0 signal, or None if I0 is missing.
        """
        return np.mean(self.I0, axis=1) if self.I0 is not None else None

    @property
    def time_averaged_I1(self) -> Optional[np.ndarray]:
        """
        Calculates the time-averaged I1 signal.

        Returns:
            Optional[np.ndarray]: 1D array of the time-averaged I1 signal, or None if I1 is missing.
        """
        return np.mean(self.I1, axis=1) if self.I1 is not None else None

    @property
    def compound(self) -> Optional[str]:
        """
        Returns the compound name from metadata.

        Returns:
            Optional[str]: Compound name, or None if unavailable.
        """
        return self.metadata.get("compound", None)
    
    @property
    def edge(self) -> float:
        """
        Returns the absorption edge energy from metadata or computes it as the position of the maximum derivative.

        Returns:
            float: Absorption edge energy in eV.
        """
        edge = self.metadata.get("edge", None)
        if edge is None:
            print("Calculating the edge position using max derivative...")
            edge = self.max_derivative
            self.metadata["edge"] = edge
        return edge

    @property
    def edge_type(self) -> Optional[str]:
        """
        Returns the absorption edge type (e.g., 'K' or 'L') from metadata.

        Returns:
            Optional[str]: Absorption edge type, or None if unavailable.
        """
        return self.metadata.get("edge_type", None)

    @property
    def noise(self) -> float:
        """
        Estimates the standard deviation of noise in the pre-edge region.

        Returns:
            float: Estimated noise level.
        """
        crop = self.pre_edge_region_indices
        if crop is None or self.spectrum is None:
            return 0.0
        y = self.spectrum[crop, :]
        return np.median(np.std(y, axis=0))

    @property
    def compound(self) -> Optional[str]:
        """
        Returns the compound name from metadata.

        Returns:
            Optional[str]: Compound name, or None if unavailable.
        """
        return self.metadata.get("compound", None)
    
    def _find_extremum(self, kind="maxima", edge=None, after_edge=True):
        """
        Finds the first extrema (maxima or minima) relative to the edge.

        Args:
            kind (str): Type of extremum to find ("maxima" or "minima").
            edge (float): Edge energy. Defaults to None.
            after_edge (bool): Whether to look after the edge. Defaults to True.

        Returns:
            float: Energy of the first extremum, or the edge if not found.
        """
        spectrum = self.time_averaged_spectrum
        energy = self.energy
        if edge is None:
            edge = self.edge

        # filter data with a window size of around 10 eV
        # win_len_eV = 10
        # win = np.argmin(np.cumsum(abs(np.diff(energy))) <= win_len_eV)
        # win = win // 2 * 2 + 1
        # spectrum = uniform_filter1d(spectrum, win, axis=0)

        crop = np.where(energy > edge)[0] if after_edge else np.where(energy < edge)[0]
        if crop.size == 0:
            return edge

        if kind == "maxima":
            extrema_indices = np.where(np.gradient(np.sign(np.gradient(spectrum[crop]))) == 1)[0][0] - 2 
        else:
            extrema_indices = np.where(np.gradient(np.sign(np.gradient(spectrum[crop]))) == -1)[0][-1] - 2 

        if extrema_indices.size > 0:
            idx = extrema_indices + (crop[0] if after_edge else 0)
            return round(energy[idx], 1)
        return edge    

    @property
    def first_maxima_after_edge(self):
        return self._find_extremum(kind="maxima", after_edge=True)

    @property
    def first_minima_after_edge(self):
        return self._find_extremum(kind="minima", after_edge=True)

    @property
    def first_minima_before_edge(self):
        return self._find_extremum(kind="minima", after_edge=False)

    @property
    def max_derivative(self) -> float:
        """
        Finds the energy where the derivative of the spectrum is maximal.

        Returns:
            float: Energy of the maximum derivative.
        """
        spectrum = self.time_averaged_spectrum
        energy = self.energy
        
        # Find unique energy values
        unique_indices = np.unique(energy, return_index=True)[1]
        if len(unique_indices) < len(energy):
            # Use only unique energy points
            energy = energy[np.sort(unique_indices)]
            spectrum = spectrum[np.sort(unique_indices)]
        
        # filter data with a window size of around 10 eV
        win_len_eV = 10
        win = np.argmin(np.cumsum(abs(np.diff(energy))) <= win_len_eV)
        win = win // 2 * 2 + 1
        spectrum = uniform_filter1d(spectrum, win, axis=0)
        
        derivative = np.gradient(spectrum, energy)
        max_deriv_pos = np.argmax(derivative) - 1
        return round(energy[max_deriv_pos], 1)
    
    def _get_region_indices(
        self, min_e: Optional[float] = None, max_e: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Gets the indices for a specified energy range.

        Args:
            min_e (float, optional): Minimum energy. Defaults to the first energy value.
            max_e (float, optional): Maximum energy. Defaults to the last energy value.

        Returns:
            Optional[np.ndarray]: Indices for the specified range, or None if no indices are found.
        """
        min_e = min_e if min_e is not None else self.energy[0]
        max_e = max_e if max_e is not None else self.energy[-1]
        indices = np.where((self.energy > min_e) & (self.energy < max_e))[0]
        return indices if indices.size > 0 else None

    def get_region_indices(self, region: str) -> Optional[np.ndarray]:
        """
        Gets the indices for a specific spectral region (e.g., 'xanes', 'exafs', 'pre_edge', 'post_edge').
        If region indices are not defined in metadata, they are automatically computed.

        Args:
            region (str): The region name ('xanes', 'exafs', 'pre_edge', 'post_edge').

        Returns:
            Optional[np.ndarray]: Indices for the specified region, or None if invalid region or no indices.
        """
        # First, try to get the region boundaries from metadata or compute them if needed
        if region == 'pre_edge':
            pre_edge_min = self.metadata.get("pre_edge_min_E", None)
            pre_edge_max = self.metadata.get("pre_edge_max_E", None)

            # Auto-compute if not available
            if pre_edge_min is None or pre_edge_max is None or pre_edge_min < self.energy[0] or pre_edge_max < self.energy[0]:
                pre_edge_min = self.energy[0]
                pre_edge_max = self.edge - 30 if (self.edge-30) > self.energy[0] else self.energy[10]
                # Store in metadata for future use
                if hasattr(self, 'metadata') and self.metadata is not None:
                    self.metadata["pre_edge_min_E"] = pre_edge_min
                    self.metadata["pre_edge_max_E"] = pre_edge_max
                
            region_range = (pre_edge_min, pre_edge_max)
                
        elif region == 'post_edge':
            post_edge_min = self.metadata.get("post_edge_min_E", None)
            post_edge_max = self.metadata.get("post_edge_max_E", None)
            
            # Auto-compute if not available
            if post_edge_min is None or post_edge_max is None:
                post_edge_min = self.edge + 30 if (self.edge+30) < self.energy[-1] else self.energy[-10]                
                post_edge_max = self.energy[-1]
                
                # Store in metadata for future use
                if hasattr(self, 'metadata') and self.metadata is not None:
                    self.metadata["post_edge_min_E"] = post_edge_min
                    self.metadata["post_edge_max_E"] = post_edge_max
            
            region_range = (post_edge_min, post_edge_max)
            
        elif region == 'xanes':
            # XANES region is typically defined relative to the edge
            if not hasattr(self, 'edge') or self.edge is None:
                raise ValueError("Edge energy must be defined to determine XANES region")
            region_range = (self.edge - 20, self.edge + 30)
            
        elif region == 'exafs':
            # EXAFS region is typically defined relative to the edge
            if not hasattr(self, 'edge') or self.edge is None:
                raise ValueError("Edge energy must be defined to determine EXAFS region")
            region_range = (self.edge + 30, self.energy[-1])
            
        else:
            raise ValueError(f"Invalid region specified: {region}")

        # Call the helper method to get indices based on the energy range
        min_e, max_e = region_range
        return self._get_region_indices(min_e, max_e)
    
    @property
    def xanes_region_indices(self) -> Optional[np.ndarray]:
        """
        Gets the indices corresponding to the XANES region of the spectrum.

        Returns:
            Optional[np.ndarray]: Indices for the XANES region.
        """
        return self.get_region_indices("xanes")

    @property
    def exafs_region_indices(self) -> Optional[np.ndarray]:
        """
        Gets the indices corresponding to the EXAFS region of the spectrum.

        Returns:
            Optional[np.ndarray]: Indices for the EXAFS region.
        """
        return self.get_region_indices("exafs")

    @property
    def pre_edge_region_indices(self) -> Optional[np.ndarray]:
        """
        Gets the indices corresponding to the pre-edge region of the spectrum.

        Returns:
            Optional[np.ndarray]: Indices for the pre-edge region.
        """
        return self.get_region_indices("pre_edge")

    @property
    def post_edge_region_indices(self) -> Optional[np.ndarray]:
        """
        Gets the indices corresponding to the post-edge region of the spectrum.

        Returns:
            Optional[np.ndarray]: Indices for the post-edge region.
        """
        return self.get_region_indices("post_edge")