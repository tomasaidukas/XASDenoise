"""
Regular denoiser class.
"""

import numpy as np
import warnings
import statsmodels.api as sm
import scipy.signal as signal
from scipy import interpolate, ndimage
from scipy.linalg import svd
from skimage.restoration import denoise_tv_chambolle
from scipy.signal import savgol_filter
import pywt
from xasdenoise.denoising_methods.denoising_utils import downsample_data
from sklearn.decomposition import DictionaryLearning
import time
        
try:
    warnings.filterwarnings('ignore', category=UserWarning, module='pywt')
except:
    pass

class RegularDenoiser:
    """
    A class to perform signal denoising using various built-in denoising methods and optimize their parameters.
    
    params_bounds define the range over which the parameters can be optimized
    """
    
    # Define default parameters and their bounds for each method
    DEFAULT_PARAMS = {
        "lowess": {
                            "params": {"frac": 0.05, "it": 10},
                            "params_bounds": {"frac": [0.01, 0.5]}
                   },
        "butterworth": {
                            "params": {"order": 3, "cutoff": 0.025},
                            "params_bounds": {"cutoff": [0.01, 0.5]}
                        },
        "moving_average": {
                            "params": {"window_size": 9},
                            "params_bounds": {"window_size": [3, 101]}
                           },
        "median": {
                            "params": {"window_size": 9},
                            "params_bounds": {"window_size": [3, 101]}
                   },
        "gaussian": {
                            "params": {"sigma": 10},
                            "params_bounds": {"sigma": [1, 30]}
                     },        
        "total_variation": {
                            "params": {"weight": 0.1},
                            "params_bounds": {"weight": [0.001, 1]}
                            },
        "savitzky_golay": {
                            "params": {"window_size": 11, "order": 2},
                            "params_bounds": {"window_size": [5, 101]}
                            },
        "wavelet": {
                            "params": {"modes": 10, "modes_to_keep": 1},
                            "params_bounds": {"modes": [1, 20], "modes_to_keep": [1, 20]}
                    },
        "dictionary_learning": {
                            "params": {"n_components": 50, "alpha": 1, "max_iter": 100},
                            "params_bounds": {"n_components": [1, 100], "alpha": [0, 10], "max_iter": [1, 100]}
                                },
    }
    
    # New denoising methods added to the DEFAULT_PARAMS
    DEFAULT_PARAMS.update({
        "non_local_means_2d": {
                            "params": {"patch_size": 11, "search_size": 11, "h": 0.1},                               
                            "params_bounds": {"h": [0.1, 1.5]}
                               },
        "total_variation_2d": {
                            "params": {"weight": 1, "max_iter": 200, "eps": 2.0e-6},
                            "params_bounds": {"weight": [0.001, 10]}
                               },
        "pca": {
                            "params": {"modes": 1},
                            "params_bounds": {"modes": [1, 10]}
                },
        "gaussian_filter_2d": {
                            "params": {"sigma_r": 1, "sigma_c": 1},
                            "params_bounds": {"sigma_r": [0.1, 30], "sigma_c": [0.1, 30]}
                            },
        "uniform_filter_2d": {
                            "params": {"size_r": 3, "size_c": 3},
                            "params_bounds": {"size_r": [3, 101], "size_c": [3, 101]}
                            },
        
    })
    
    # define_methods which take the whole 2D data into account when denoising instead of looping over individual spectra
    METHODS_2D = ['non_local_means_2d', 'total_variation_2d', 'pca', 'gaussian_filter_2d', 'uniform_filter_2d']
    
    def __init__(self, method_name="butterworth", **params):
        """
        Initialize the RegularDenoiser with the specified denoising method.

        Args:
            method_name (str): Name of the denoising method to use.
            **params: Additional parameters for the denoising method.
        """
        
        self.method_name = method_name
        # Load default params
        self.params = self.DEFAULT_PARAMS[method_name]["params"].copy()
        self.params_bounds = self.DEFAULT_PARAMS[method_name]["params_bounds"].copy()
        self.params.update(params)
        self.norm_params = {}
        self.verbose = 0

        # optimization params
        self.optimize_params_flag = False
        self.optimize_params_grid_num = 50
        self.optimize_params_tv_lambda = 1e-4
        self.optimize_params_time_idx = 0 # time index to be used for optimization instead of all
        self.optimize_params_method = 'grid_search' # 'grid_search' or 'gradient_descent'
        self.optimize_params_gd_max_iter = 100
        self.optimize_params_gd_lr = 0.01
        self.optimize_params_scale_tv_lambda_by_noise = False
        self.optimize_params_find_tv_lambda = False
        
        
        print(f"Denoising using method {self.method_name}, with the following parameters")
        for param, value in self.params.items():
            print(f"  {param}: {value}")
    
    def _clean_model(self):
        """
        Clean the model by setting training and prediction attributes to None.
        """
        for attr in ["x_train", "y_train", "noise_train", "x_predict", "noise_redict"]:
            if hasattr(self, attr):
                setattr(self, attr, None)
                
    def _get_current_state(self):
        """
        Get the current state of all parameters defined in the object.

        Returns:
            dict: A dictionary of the current state of the object.
        """

        return {key: value for key, value in self.__dict__.items()}

    def save_state(self):
        """
        Save the current state of the denoiser, which includes all
        arrays and parameters used for training and prediction.

        Returns:
            dict: A dictionary of the current state of the object.
        """

        return self._get_current_state()
    
    def restore_state(self, state):
        """
        Restore the state of the denoiser from the given dictionary.
        
        Args:
            state (dict): A dictionary containing the state to restore.
        """
        
        for key, value in state.items():
            setattr(self, key, value)
                  
    def initialize_denoiser(self, **kwargs):
        """
        Initialize data which will be used for subsequent denoiser methods.

        Args:
            **kwargs: Dictionary containing keys 'x' (np.ndarray), 'y' (np.ndarray), 
                     'x_predict' (np.ndarray, optional), 'y_reference' (np.ndarray, optional),
                     and 'noise' (np.ndarray, optional).
            
        Raises:
            ValueError: If required keys 'x' or 'y' are not provided.
        """
        
        self.x_train0 = kwargs.get("x", None)
        self.y_train0 = kwargs.get("y", None)
        self.x_predict0 = kwargs.get("x_predict", self.x_train0)
        self.y_reference0 = kwargs.get("y_reference", None)
        self.noise = kwargs.get("noise", None)
        
        if self.x_train0 is None or self.y_train0 is None:
            raise ValueError("Missing required arguments: x or y.")
        
    def adjust_params(self, denoising_method):
        """
        Adjust the parameters of the denoiser based on data characteristics.

        Args:
            denoising_method (str): Name of the denoising method.
        """
        # check if the default denoising parameters will work for a given denoiser. If not, adjust
        if denoising_method.lower() in ['savitzky_golay', 'moving_average', 'median']:
            if self.params['window_size'] > len(self.x_train0):
                self.params['window_size'] = int(len(self.x_train0))
                print(f"Window size adjusted to {self.params['window_size']} for {denoising_method} denoiser.")
                
            bounds = self.params_bounds['window_size']
            if bounds[-1] > len(self.x_train0):
                self.params_bounds['window_size'] = (bounds[0], int(len(self.x_train0)))
                print(f"Window size bounds adjusted to {self.params_bounds['window_size']} for {denoising_method} denoiser.")
                
    # --------------------------- Normalization Methods ---------------------------
    def normalize_data(self, x, y, x_predict, y_reference=None):
        """
        Normalize x to the range [-1, 1], and normalize y by subtracting the mean and dividing by the range.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            x_predict (np.ndarray): Independent variable values for prediction.
            y_reference (np.ndarray, optional): Reference dependent variable values. Defaults to None.

        Returns:
            tuple: Normalized x, y, x_predict, and y_reference values.
        """

        # Normalize x to [-1, 1]
        x_min, x_max = np.min(x), np.max(x)
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        self.norm_params['x'] = (x_min, x_max)

        if x_predict is not None:
            # x_min, x_max = np.min(x_predict), np.max(x_predict)
            x_predict_norm = 2 * (x_predict - x_min) / (x_max - x_min) - 1
            # self.norm_params['x_predict'] = (x_min, x_max)
        else:
            x_predict_norm = None

        # Normalize y: Subtract mean and scale by range
        y_mean, y_range = np.mean(y, axis=0), np.max(y, axis=0) - np.min(y, axis=0)
        # y_mean, y_range = np.mean(y), np.max(y) - np.min(y)
        y_norm = (y - y_mean) / y_range
        self.norm_params['y'] = (y_mean, y_range)


        if y_reference is not None:
            y_mean, y_range = np.mean(y_reference, axis=0), np.max(y_reference, axis=0) - np.min(y_reference, axis=0)
            y_reference_norm = (y_reference - y_mean) / y_range
        else:
            y_reference_norm = None
            
        return x_norm, y_norm, x_predict_norm, y_reference_norm

    def denormalize_data(self, y_norm):
        """
        Denormalize y using stored normalization parameters.

        Args:
            y_norm (np.ndarray): Normalized dependent variable values.

        Returns:
            np.ndarray: Denormalized y values.
        """

        y_mean, y_range = self.norm_params['y']

        # Denormalize y
        y = y_norm * y_range + y_mean

        return y  
        
    def denoise_with_downsampling(self, downsampling_pts=None, downsampling_method=None, smoothness=None):
        """
        Downsample the data and then denoise using the selected method.

        Args:
            downsampling_pts (int, optional): Number of points to downsample to. Defaults to None.
            downsampling_method (str, optional): Method to use for downsampling. Defaults to None.
            smoothness (np.ndarray, optional): Smoothness values for downsampling. Defaults to None.

        Returns:
            tuple: Denoised signal, error estimates, and noise estimates.
        """

        if downsampling_pts is None or downsampling_method is None:
            y_denoised, y_error, y_noise = self.denoise()
            return y_denoised, y_error, y_noise
        
        num_samples = np.min([downsampling_pts, len(self.x_train0)]).astype(int)             
        if self.verbose: print(f'Denoising with downsampling. Using {num_samples} data points out of {len(self.x_train0)}')   
        
        if num_samples == len(self.x_train0):
            y_denoised, y_error, y_noise = self.denoise()
            
        else:
            # downsample the training data
            num_samples = np.min([downsampling_pts, len(self.x_train0)]).astype(int)  
            
            self.x_train0, self.y_train0 = downsample_data(self.x_train0, self.y_train0,
                                                            method=downsampling_method, 
                                                            num_samples=num_samples, 
                                                            smoothness=smoothness)
            
            # denoise the downsampled data
            y_denoised, y_error, y_noise = self.denoise()
            
        return y_denoised, y_error, y_noise            
            
    def denoise(self):
        """
        Denoise the data using the selected denoising method.

        Returns:
            tuple: Denoised signal, error estimates, and noise estimates.

        Raises:
            ValueError: If 'x' or 'y' data are not initialized.
            ValueError: If the specified denoising method is unavailable.
        """

        # Extract required arguments
        x = self.x_train0
        y = self.y_train0
        x_predict = self.x_predict0
        y_reference = self.y_reference0
        noise = self.noise
        
        if x is None or y is None:
            raise ValueError("Missing required arguments: x or y.")
        
        if self.verbose: print(f'Denoising using method: {self.method_name}.')
        
        # Adjust denoising parameters if needed
        self.adjust_params(self.method_name)
        
        # Normalize the data
        x, y, x_predict, y_reference = self.normalize_data(x, y, x_predict, y_reference)

        # The denoisers can operate on 1D or 2D arrays
        dim = y.ndim
        if dim == 1: y = y[:, np.newaxis]
            
        # Use the specified denoising method
        method = getattr(self, self.method_name.lower(), None)
        if method is None:
            raise ValueError(f"Denoising method '{self.method_name}' is not available.")
        
        # Optimize denoising parameters if needed
        if self.optimize_params_flag:
            self.params = self.optimize_denoiser_params(x, y, y_reference=y_reference, noise=noise, denoising_method=method)
        
        # Call the denoising method with the provided parameters
        y_denoised = method(x, y, **self.params)
        
        # If x_predict is provided and it is different than x it means that 
        # a downsampled dataset was denoised and it needs to be interpolated 
        # onto x_predict grid        
        if x_predict is not None and not np.array_equal(x, x_predict):
            f = interpolate.interp1d(x, y_denoised, axis=0, fill_value=(y_denoised[0,:], y_denoised[-1,:]), kind='linear', bounds_error=False)
            y_denoised = f(x_predict)
            # for t in range(y_denoised.shape[1]):
            #     tmp = np.zeros((len(x_predict), y_denoised.shape[1]))
            #     tmp[:,t] = np.interp(x_predict, x, y_denoised[:, t])
            # y_denoised = tmp
            
        if dim == 1: y_denoised = y_denoised[:, 0]
        
        y_denoised = self.denormalize_data(y_denoised)

        y_error = np.zeros_like(y_denoised)
        y_noise = np.zeros_like(y_denoised)

        return y_denoised, y_error, y_noise
    
    def gaussian_filter_2d(self, x, y, sigma_r=1, sigma_c=1):
        """
        Apply 2D Gaussian filter with different sigmas for energy and time dimensions.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            sigma_r (float): Standard deviation for rows (energy dimension). Defaults to 1.
            sigma_c (float): Standard deviation for columns (time dimension). Defaults to 1.

        Returns:
            np.ndarray: Filtered signal.
        """
        from scipy import ndimage
        # Use larger sigma for energy (vertical) dimension, smaller for time
        return ndimage.gaussian_filter(y, sigma=(sigma_r, sigma_c))

    def uniform_filter_2d(self, x, y, size_r=3, size_c=3):
        """
        Apply 2D box filter with different sizes for energy and time dimensions.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            size_r (int): Filter size for rows (energy dimension). Defaults to 3.
            size_c (int): Filter size for columns (time dimension). Defaults to 3.

        Returns:
            np.ndarray: Filtered signal.
        """
        from scipy import ndimage
        return ndimage.uniform_filter(y, size=(size_r, size_c))
    
    def total_variation_2d(self, x, y, weight=1, max_iter=200, eps=2.0e-6):
        """
        Apply 2D Total Variation denoising.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            weight (float): Weight of the total variation regularization. Defaults to 1.
            max_iter (int): Maximum number of iterations. Defaults to 200.
            eps (float): Convergence tolerance. Defaults to 2.0e-6.

        Returns:
            np.ndarray: Denoised signal.
        """        
        # Apply TV denoising considering both spectral and temporal gradients
        return denoise_tv_chambolle(y, weight=weight, max_num_iter=max_iter, eps=eps)

    def non_local_means_2d(self, x, y, patch_size=5, search_size=11, h=0.1):
        """
        Apply Non-Local Means denoising to the 2D spectra+time matrix.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values (2D spectra matrix).
            patch_size (int): Size of comparison patches (spectral segments). Defaults to 5.
            search_size (int): Search neighborhood size (temporal window). Defaults to 11.
            h (float): Filtering parameter controlling decay. Defaults to 0.1.

        Returns:
            np.ndarray: Denoised signal.
        """
        from skimage.restoration import denoise_nl_means
        
        # Normalize the input for better processing
        y_min, y_max = np.min(y), np.max(y)
        y_norm = (y - y_min) / (y_max - y_min)
        
        # Apply NL-means 
        # Parameters tuned for spectroscopy:
        # - patch_size: size of comparison patches (spectral segments)
        # - patch_distance: search neighborhood (temporal window)
        y_denoised = denoise_nl_means(y_norm, 
                                    patch_size=patch_size,
                                    patch_distance=search_size,
                                    h=h,
                                    fast_mode=True)
        
        # Restore original scale
        return y_denoised * (y_max - y_min) + y_min

    def lowess(self, x, y, frac=0.1, it=1):
        """
        Perform lowess denoising on the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            frac (float): Fraction of data points to use for smoothing. Defaults to 0.1.
            it (int): Number of iterations for the lowess algorithm. Defaults to 1.

        Returns:
            np.ndarray: Denoised signal.
        """

        for t in range(y.shape[1]):
            y[:, t] = sm.nonparametric.lowess(exog=x, endog=y[:, t], frac=frac, it=it)[:, 1]
        return y

    def butterworth(self, x, y, order=3, cutoff=0.025):
        """
        Apply a Butterworth filter to the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            order (int): Order of the filter. Defaults to 3.
            cutoff (float): Cutoff frequency as a fraction of the Nyquist frequency. Defaults to 0.025.

        Returns:
            np.ndarray: Filtered signal.
        """

        b, a = signal.butter(order, cutoff)
        return signal.filtfilt(b, a, y, axis=0)

    def moving_average(self, x, y, window_size=9):
        """
        Perform moving average smoothing on the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            window_size (int): Size of the moving average window. Defaults to 9.

        Returns:
            np.ndarray: Smoothed signal.
        """
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(y, size=window_size, axis=0)

    def median(self, x, y, window_size=9):
        """
        Apply a median filter to the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            window_size (int): Size of the median filter window. Defaults to 9.

        Returns:
            np.ndarray: Filtered signal.
        """

        return ndimage.median_filter(y, (window_size, 1))

    def gaussian(self, x, y, sigma=10):
        """
        Apply Gaussian smoothing to the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            sigma (float): Standard deviation for Gaussian kernel. Defaults to 10.

        Returns:
            np.ndarray: Smoothed signal.
        """
        if sigma == 0:
            warnings.warn("Gaussian Sigma is set to 0, returning original signal without smoothing.")
            return y
        return ndimage.gaussian_filter1d(y, sigma, axis=0)

    def total_variation(self, x, y, weight=0.1):
        """
        Apply total variation denoising to the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            weight (float): Weight of the total variation regularization. Defaults to 0.1.

        Returns:
            np.ndarray: Denoised signal.
        """
        return denoise_tv_chambolle(y, weight=weight, channel_axis=1)

    def savitzky_golay(self, x, y, window_size=11, order=2):
        """
        Apply Savitzky-Golay filtering to the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            window_size (int): Size of the smoothing window. Defaults to 11.
            order (int): Order of the polynomial to fit. Defaults to 2.

        Returns:
            np.ndarray: Smoothed signal.
        """
        if window_size <= order:
            print(f"Window size {window_size} is less than or equal to order {order}.")            
            window_size = max(window_size, order+1)  # Ensure window size does not exceed data length
            print(f"Adjusting window size to {window_size} for Savitzky-Golay filter.")
        return savgol_filter(y, window_length=window_size, polyorder=order, axis=0)
    
    
    # ------------------- Basis decomposition methods -------------------
    def pca(self, x, y, modes=1):
        """
        Perform Principal Component Analysis (PCA) denoising on the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            modes (int): Number of principal components to retain. Defaults to 1.

        Returns:
            np.ndarray: Denoised signal.

        Raises:
            RuntimeError: If PCA is attempted on single-instance data.
        """
        assert y.shape[1] > modes, "Number of modes/components to be removed should be less than the number of time instances"

        if y.shape[1] > 1:
            # Center the data
            y_mean = np.mean(y, axis=1, keepdims=True)
            y_centered = y - y_mean
            
            # Direct SVD decomposition
            U, S, Vh = svd(y_centered, full_matrices=False)
            
            # Low-rank reconstruction using first 'modes' components
            y_lowrank = np.dot(U[:, :modes] * S[:modes], Vh[:modes, :])
            
            # Add back the mean
            y_denoised = y_lowrank + y_mean
            
            return y_denoised
        else:
            raise('pca denoising requires multiple time instances to work.')

    def wavelet(self, x, y, modes=10, modes_to_keep=5, wavelet='db8', determine_cutoff=False):
        """
        Perform wavelet denoising on the input signal.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            modes (int): Number of decomposition levels. Defaults to 10.
            modes_to_keep (int): Number of coefficients to retain. Defaults to 5.
            wavelet (str): Name of the wavelet to use. Defaults to 'db8'.
            determine_cutoff (bool): Whether to determine the optimal cutoff mode automatically. Defaults to False.

        Returns:
            np.ndarray: Denoised signal.
        """        
        # Get the number of time instances
        n_times = y.shape[1]
        
        # Prepare output array with same shape as input
        y_filtered = np.zeros_like(y)
        
        # Process each time instance separately
        for t in range(n_times):
            # Extract the time series for this instance
            y_t = y[:, t]
            
            # Ensure modes is valid for this signal length
            max_modes = pywt.dwt_max_level(len(y_t), pywt.Wavelet(wavelet).dec_len)
            current_modes = min(modes, max_modes)
            
            # Perform 1D wavelet decomposition for this time instance
            coeffs = pywt.wavedec(y_t, wavelet, level=current_modes)
            
            if determine_cutoff:
                threshold = 0.90
                
                # Calculate the energy of each coefficient
                coeff_energy = [np.sum(c**2) for c in coeffs]
                
                # Determine how many modes to keep based on cumulative energy
                total_energy = np.sum(coeff_energy)
                cumulative_energy = np.cumsum(coeff_energy) / total_energy
                cutoff = np.argmax(cumulative_energy >= threshold) + 1
            else:
                cutoff = min(modes_to_keep, len(coeffs))
                
            # Zero-out coefficients beyond the cutoff
            sparse_coeffs = [
                coeffs[i] if i < cutoff else np.zeros_like(coeffs[i]) for i in range(len(coeffs))
            ]
            
            # Reconstruct the denoised signal for this time instance
            y_reconstructed = pywt.waverec(sparse_coeffs, wavelet)
            # Match original length (wavelet reconstruction can sometimes return slightly different length)
            y_filtered[:, t] = y_reconstructed[:len(y_t)]
            
        return y_filtered
    
    def dictionary_learning(self, x, y, n_components=50, alpha=1.0, max_iter=1000):
        """
        Perform dictionary learning-based denoising on the input signal.

        Args:
            x (np.ndarray): Independent variable values (not used directly but included for compatibility).
            y (np.ndarray): Dependent variable values (signal to be denoised).
            n_components (int): Number of dictionary components to learn. Defaults to 50.
            alpha (float): Sparsity controlling parameter (higher values lead to sparser representations). Defaults to 1.0.
            max_iter (int): Maximum number of iterations for dictionary learning. Defaults to 1000.

        Returns:
            np.ndarray: Denoised signal.
        """
        # Apply dictionary learning
        dict_learner = DictionaryLearning(
            n_components=n_components, alpha=alpha, max_iter=max_iter, transform_algorithm='lasso_lars'
        )
        y_denoised = dict_learner.fit_transform(y)
        y_reconstructed = np.dot(y_denoised, dict_learner.components_)

        return y_reconstructed



    # --------------------------- Optimization Methods ---------------------------
    def optimize_denoiser_params(self, x, y, y_reference=None, noise=None, denoising_method=None):
        """
        Optimize denoising parameters using a self-supervised approach with optional TV lambda adjustment.
        
        TV lambda controls the roughness of the signal and allows automated estimation of denoising 
        parameters for a given spectrum. It can be fixed to a specific value or tuned automatically.
        
        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            y_reference (np.ndarray, optional): Reference signal to compare against. Defaults to None.
            noise (np.ndarray, optional): Noise estimates for scaling TV lambda. Defaults to None.
            denoising_method (callable, optional): Denoising method to be optimized. Defaults to None.
            
        Returns:
            dict: Optimized parameters.
        """
        if self.optimize_params_scale_tv_lambda_by_noise and noise is not None:
            self.optimize_params_tv_lambda_scaling = np.mean(noise[:])
        else:
            self.optimize_params_tv_lambda_scaling = 1.0

        # The denoised signal will be compared against the reference signal
        # This can be either the same noisy signal, it can be provided manually
        # or for time-resolved data it can be the time-averaged signal.
        if y_reference is not None:
            reference = y_reference
            print(f"Using provided reference signal for parameter optimization.")                
        elif self.method_name.lower() in self.METHODS_2D:
            # reference = np.mean(y, axis=1) # use time averaged signal as the reference
            reference = y.copy() # compare all time instances during optimization
        else:
            reference = y[:, self.optimize_params_time_idx]

        
        if reference.ndim == 1:
            reference = reference[:, np.newaxis]                                    
                
        if self.method_name.lower() in self.METHODS_2D:
            if self.optimize_params_method == 'grid_search':
                self.params = self.optimize_denoiser_params_grid_search(x, y, reference=reference, denoising_method=denoising_method, verbose=self.verbose) 
            elif self.optimize_params_method == 'gradient_descent':
                self.params = self.optimize_denoiser_params_gd(x, y, reference=reference, denoising_method=denoising_method, verbose=self.verbose)  

        else:
            y_noisy = y[:, self.optimize_params_time_idx]
            if y_noisy.ndim == 1:
                y_noisy = y_noisy[:, np.newaxis]
                
            if self.optimize_params_method == 'grid_search':
                self.params = self.optimize_denoiser_params_grid_search(x, y_noisy, reference=reference, denoising_method=denoising_method, verbose=self.verbose)  
            elif self.optimize_params_method == 'gradient_descent':
                self.params = self.optimize_denoiser_params_gd(x, y_noisy, reference=reference, denoising_method=denoising_method, verbose=self.verbose) 

        # get the best params for the final denoising            
        return self.params
    
    
    def optimize_denoiser_params_grid_search(self, x, y, reference, denoising_method, verbose=0):
        """
        Optimize denoising parameters using a simple grid search.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            reference (np.ndarray): Reference signal to compare against.
            denoising_method (callable): Denoising method to be optimized.
            verbose (int): Verbosity level. Defaults to 0.
            
        Returns:
            dict: Optimized parameters.
        """
        
        def _TV(signal):
            """Compute the total variation of a signal."""
            return np.sum(np.abs(np.diff(signal, axis=0)), axis=0)
        
        def fix_param_type(param_value, original_param_value):
            """Ensure the parameter is converted back to its original type."""
            if isinstance(param_value, (list, tuple, np.ndarray)):
                param_value = param_value[0] if len(param_value) > 0 else param_value

            if isinstance(original_param_value, int):
                return int(param_value)
            elif isinstance(original_param_value, float):
                return float(param_value)
            
            
        if y.ndim == 1:
            y = y[:, np.newaxis]
            
        if reference.ndim == 1:
            reference = reference[:, np.newaxis]

        for param, param_bounds in self.params_bounds.items():
            if verbose:
                print(f"Optimizing parameter: {param} = {self.params[param]}, with bounds {param_bounds}")
            time_start = time.time()
            param_value = self.params[param]
            loss = []
            # candidates = [param_value, param_bounds[0], param_bounds[1], param_bounds[0] + (param_bounds[1] - param_bounds[0]) / 2]
            
            if denoising_method.__name__.lower() in ['total_variation', 'total_variation_2d']:
                # For TV denoising, we use a fixed set of candidates
                candidates = np.logspace(np.log(param_bounds[0]), np.log(param_bounds[1]), self.optimize_params_grid_num, dtype=type(param_value))
            else:
                candidates = np.linspace(param_bounds[0], param_bounds[1], self.optimize_params_grid_num, dtype=type(param_value))
                
            if isinstance(param_value, int):
                candidates = [int(c) for c in candidates]
                candidates = np.unique(candidates)
 
            # Track best parameter and its loss
            best_loss = float('inf')
            best_param = None
            
            for p in candidates:
                self.params[param] = p
                
                y_denoised = denoising_method(x, y, **self.params)
                
                # Compute loss as weighted sum of data fidelity and TV regularization
                # Data fidelity - how well denoised signal matches noisy data
                loss = np.mean((reference - y_denoised) ** 2, axis=0)

                if self.optimize_params_tv_lambda > 0:
                    # Total variation - encourages smoothness
                    tv_term = _TV(y_denoised)
                    
                    # Combined loss with TV prior                   
                    loss += self.optimize_params_tv_lambda * tv_term * self.optimize_params_tv_lambda_scaling
                
                loss = np.mean(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_param = p
            
            best_param = fix_param_type(best_param, param_value)

            self.params[param] = best_param
            
            time_end = time.time()
            if verbose:
                # print(f"Grid search completed in {(time_end - time_start)*1e3:.2f} miliseconds.")
                print(f"Optimized parameter value: {param} = {self.params[param]}")
        return self.params
    
    def optimize_denoiser_params_gd(self, x, y, reference, denoising_method, verbose=0):
        """
        Optimize denoising parameters using gradient descent.

        Args:
            x (np.ndarray): Independent variable values.
            y (np.ndarray): Dependent variable values.
            reference (np.ndarray): Reference signal to compare against.
            denoising_method (callable): Denoising method to be optimized.
            verbose (int): Verbosity level. Defaults to 0.

        Returns:
            dict: Optimized parameters.
        """
        
        def _TV(signal):
            """Compute the total variation of a signal."""
            return np.sum(np.abs(np.diff(signal, axis=0)), axis=0)
        
        def loss_function(param, param_value):
            # Evaluate the denoising performance using the new parameter value
            self.params[param] = param_value
            y_denoised = denoising_method(x, y, **self.params)

            # Data fidelity term
            data_loss = np.mean((reference - y_denoised) ** 2, axis=0)

            # TV regularization term
            if self.optimize_params_tv_lambda > 0:
                tv_loss = self.optimize_params_tv_lambda * _TV(y_denoised) * self.optimize_params_tv_lambda_scaling
                return np.mean(data_loss + tv_loss)
            else:
                return np.mean(data_loss)

        def enforce_bounds(value, bounds):
            lower_bound, upper_bound = bounds
            if lower_bound is not None:
                value = max(value, lower_bound)
            if upper_bound is not None:
                value = min(value, upper_bound)
            return value

        def gradient_update(param, param_value, bounds, learning_rate, epsilon=1e-3):
            # Evaluate loss at two different value directions
            # This gives us f(x+delta) and f(x-delta)
            param_value_plus = enforce_bounds(param_value + epsilon, bounds)
            param_value_minus = enforce_bounds(param_value - epsilon, bounds)   
            
            loss_plus = loss_function(param, param_value_plus)
            loss_minus = loss_function(param, param_value_minus)
            
            # Approximate the gradient using the formula
            # grad = (f(x+delta) - f(x-delta)) / 2*delta
            grad = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Perform gradient step update
            new_value = param_value - learning_rate * grad
            new_value = enforce_bounds(new_value, bounds)   

            return new_value

        def gradient_update_int(param, param_value, bounds):
            # Evaluate loss at two different value directions
            # This gives us f(x+delta) and f(x-delta)
            param_value_plus = int(enforce_bounds(param_value + 1, bounds))
            param_value_minus = int(enforce_bounds(param_value - 1, bounds))
            
            loss_plus = loss_function(param, param_value_plus)
            loss_minus = loss_function(param, param_value_minus)
            
            if loss_plus < loss_minus:
                new_value = param_value_plus
            else:
                new_value = param_value_minus
            return new_value
        
        def fix_param_type(param_value, original_param_value):
            """Ensure the parameter is converted back to its original type."""
            if isinstance(param_value, (list, tuple, np.ndarray)):
                param_value = param_value[0] if len(param_value) > 0 else param_value

            if isinstance(original_param_value, int):
                return int(param_value)
            elif isinstance(original_param_value, float):
                return float(param_value)
            
            
        if y.ndim == 1:
            y = y[:, np.newaxis]
            
        if reference.ndim == 1:
            reference = reference[:, np.newaxis]
            
        for param, param_bounds in self.params_bounds.items():
            if verbose:
                print(f"Optimizing parameter: {param} = {self.params[param]} with bounds {param_bounds}")
            time_start = time.time()
            
            # Store the original parameter type for later use
            original_param_value = self.params[param]
            original_param_type = type(original_param_value)
                        
            # Step 1: Grid search for initial estimate
            if verbose:
                print(f"Initializing with grid search...")
            self.params = self.optimize_denoiser_params_grid_search(x, y, reference, denoising_method)
            
            # # Set parameter to best found value from grid search
            # if verbose:
            #     print(f"Initial best parameter from grid search: {param} = {self.params[param]}")
            
            # Step 2: Fine-tune with gradient descent
            if verbose:
                print("Fine-tuning with gradient descent...")
            time_start = time.time()
            memory = np.ones(10)
            
            for itr in range(self.optimize_params_gd_max_iter):
                param_value = self.params[param]
                
                # Use the original type, not param0
                if original_param_type == float:
                    new_value = gradient_update(param, param_value, param_bounds, self.optimize_params_gd_lr, epsilon=1e-3)
                elif original_param_type == int:
                    new_value = gradient_update_int(param, param_value, param_bounds)                
                else:
                    print(f"self.params = {self.params}")
                    print(f"param = {param}")
                    print(f"param_value = {param_value}")
                    print(f"original_param_type = {original_param_type}")
                    raise ValueError(f"Parameter type not supported: {original_param_type}")
                
                new_value = fix_param_type(new_value, original_param_value)
                
                # Store the new parameter value
                self.params[param] = new_value
                changes = np.abs(param_value - new_value)
                
                # Add the error to the memory and shift the array by one
                memory[0] = changes
                memory = np.roll(memory, 1)
                
                # Check convergence
                if np.mean(memory) < 1e-6:
                    break
                
            time_end = time.time()
            if verbose:
                # print(f"Gradient descent optimization completed in {(time_end - time_start)*1e3:.2f} miliseconds.")
                print(f"Optimized parameter: {param} = {self.params[param]}. Took {itr+1} iterations out of {self.optimize_params_gd_max_iter}.")
        return self.params
                
    def list_methods(self):
        """
        List all available denoising methods in the RegularDenoiser class.

        Returns:
            list: Names of available denoising methods.
        """

        return [method for method in dir(self) if callable(getattr(self, method)) and not method.startswith("__")]

