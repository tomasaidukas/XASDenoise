"""
Gaussian Process denoiser class.
"""

import numpy as np
import gc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from xasdenoise.denoising_methods.denoising_utils import downsample_data, blend_overlapping_windows
import copy
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.spatial.distance import pdist
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
import joblib

# Conditional imports for torch and gpytorch
TORCH_AVAILABLE = False
GPYTORCH_AVAILABLE = False
MPS_AVAILABLE = False
MPS_BUILT = False
try:
    import torch
    TORCH_AVAILABLE = True
    # Probe Metal (Apple GPU) backend if torch is present
    try:
        MPS_AVAILABLE = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        MPS_BUILT = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_built()
    except Exception:
        MPS_AVAILABLE = False
        MPS_BUILT = False
except ImportError:
    torch = None
    # warnings.warn("PyTorch not available - GPDenoiser will not be functional")

# Only import GPyTorch if PyTorch is available
if TORCH_AVAILABLE:
    try:
        import gpytorch
        from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, SpectralMixtureKernel, PeriodicKernel
        from gpytorch.models.exact_gp import GPInputWarning
        GPYTORCH_AVAILABLE = True
    except ImportError:
        gpytorch = None
        # warnings.warn("GPyTorch not available - GPDenoiser will not be functional")


class GPDenoiser:
    """
    A class to perform signal denoising using Gaussian Processes (GPs).
    
    Note: Requires PyTorch and GPyTorch. If not available, raises ImportError on initialization.
    """
    def __init__(self, lr=.1, training_iter=100, lengthscale=.1, verbose=0, gpu_index=0):
        """
        Initialize the GPDenoiser with the specified parameters.
        
        Args:
            lr (float): Learning rate for the optimizer. Defaults to 0.1.
            training_iter (int): Number of training iterations. Defaults to 100.
            lengthscale (float or list): Initial lengthscale for the GP kernel. Defaults to 0.1.
            verbose (int): Verbosity level (0 for silent, 1 for verbose). Defaults to 0.
            gpu_index (int): Index of the GPU to use. Defaults to 0.
            
        Raises:
            ImportError: If PyTorch or GPyTorch is not available.
        """
        # Check dependencies before initialization
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GPDenoiser but not available. "
                "Install with: pip install torch"
            )
        
        if not GPYTORCH_AVAILABLE:
            raise ImportError(
                "GPyTorch is required for GPDenoiser but not available. "
                "Install with: pip install gpytorch"
            )
            
        # Training parameters
        self.lr = lr
        self.training_iter = training_iter
        self.lengthscale = lengthscale

        # Internal parameters
        self.verbose = verbose
        # Default dtype; will be adjusted to float32 when using Apple MPS, which doesn't support float64 well
        self.dtype = torch.float64

        # Device and GPU settings
        self.gpu_index = gpu_index
        self.use_gpu = False
        self.gpu_device = self._initialize_gpu_device()

        # noise estimation params
        self.noise_scale = 1  # multiplicative scaling parameter to enhance or dampen the noise.

        # internal state
        self.x_train = self.y_train = self.noise_train = None
        self.x_predict = self.noise_predict = None
        self.model = self.likelihood = None
        self.training_metrics = None
        self.weights = None

        # data normalization
        self.norm_params = {}
        self.normalize_training_data = True

        # noise estimation
        self.refine_noise_estimate = False
        self.noise_window_gp = 1000
        # for noise-free signals it is best to use a small nu value, otherwise signal oscillations will be picked up as noise
        self.noise_est_mattern_nu = 2.5
        self.learn_additional_noise = True
        self.filter_refined_noise_estimate = False
        # 'smooth' - smooth noise estimation; 'sharp' - aggressive noise estimation for every signal value
        self.refine_noise_model = 'sharp'
        # whether to estimate local noise from residuals of the GP model
        self.estimate_local_noise_from_residuals = True
        # Method for local noise estimation: 'rms', 'mad', 'percentile_68', 'abs_mean'
        self.estimate_local_noise_from_residuals_method = 'rms'

        # Windowed GP denoising flags
        self.denoise_with_windows_flag = False
        self.window_size = 1000
        self.overlap_factor = 1
        self.gp_refinement_epochs = 1  # number of epochs to run the GP refinement

        # lengthscale parameters
        self.estimate_lengthscale = False
        self.lengthscale_quantile_low = 0.05
        self.lengthscale_quantile_high = 0.5
        self.lengthscale_prior = False
        self.dont_update_lengthscale = False
        self.dont_update_lengthscale_iters = np.inf

        # Flags and auxiliary parameters
        # 2 will include the noise in the prediction, 1 will not, 3 will not use the likelihood at all
        self.prediction_mode = 1
        # 'FixedNoiseGaussianLikelihood', 'GaussianLikelihood'
        self.likelihood_model = 'FixedNoiseGaussianLikelihood'
        # Matern kernel parameter. 2.5 is for very smooth signals, 1.5 is for rough signals, 0.5 is for very rough signals.
        self.mattern_nu = 2.5
        self.auto_stop_training = True
        # We typically downsample data since it is too expensive to denoise the entire dataset.
        self.refine_downsampled_estimate = False
        # There is also an option to refine the downsampled estimate by running a few iterations on the full dataset on CPU if GPU memory is insufficient.
        self.refine_downsampled_estimate_iters = 5

        # marginal log likelihood (MLL) of the GP model.
        # Use 'ExactMarginalLogLikelihood' for standard GP,
        # Use 'LeaveOneOutPseudoLikelihood' for sparse signals where sensitivity to sharp peaks is needed
        self.mll = 'ExactMarginalLogLikelihood'
        self.num_kernels = 1  # Number of kernels to use in the GP model
        self.use_DKL = False  # Use Deep Kernel Learning

        # Save initial state
        self._initial_state = self._get_current_state()

        # Temporal data denoising
        self.joint_temporal_optimization = False

        warnings.filterwarnings("ignore", category=GPInputWarning)
        # warnings.filterwarnings("ignore", category=NumericalWarning,
        #                       message="Very small noise values detected")

        # Add parallel processing parameters
        self.parallel_windows = False  # Enable parallel window processing
        self.max_workers = mp.cpu_count() // 2  # Number of parallel workers
        # Track available accelerators; on macOS MPS is treated as a GPU-like accelerator
        try:
            cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            cuda_count = 0
        self.available_gpus = list(range(cuda_count)) if cuda_count > 0 else (["mps"] if MPS_AVAILABLE else [])
        self.parallel_backend = 'joblib'  # 'joblib', 'multiprocessing', 'threading'

        
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
        
        Parameters:
            state (dict): A dictionary containing the state to restore.
        """
        
        for key, value in state.items():
            setattr(self, key, value)

    def reset_to_initial_state(self):
        """
        Reset the denoiser to its initial state at initialization.
        """
        
        self.restore_state(self._initial_state)
        
    def _select_device(self):
        """
        Select the appropriate computation device (CPU or GPU).
        """
        
        if self.use_gpu:
            # Prefer CUDA, else Apple MPS
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = self.gpu_device if isinstance(self.gpu_device, torch.device) else torch.device(f"cuda:{self.gpu_index}")
            elif TORCH_AVAILABLE and MPS_AVAILABLE:
                self.device = torch.device("mps")
                # Force float32 on MPS for compatibility/perf
                self.dtype = torch.float32
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
    def _initialize_gpu_device(self):
        """
        Initialize and return the computation device (GPU or CPU).

        Returns:
            torch.device: The computation device to use.
        """
        
        # Prefer CUDA if available, else Apple MPS, else CPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.gpu_index}")
            try:
                torch.cuda.set_device(device)  # Specify your GPU device index
                torch.empty(1, device=device)  # Initialize CUDA context on the target GPU
                self._clean_gpu_memory()
            except Exception:
                pass

            if self.verbose:
                try:
                    print(f"Using GPU index: {self.gpu_index}, {torch.cuda.get_device_name(device)}")
                except Exception:
                    print(f"Using CUDA device: {device}")
        elif TORCH_AVAILABLE and MPS_AVAILABLE:
            device = torch.device("mps")
            # MPS prefers float32; adjust dtype for compatibility
            self.dtype = torch.float32
            if self.verbose:
                backend_info = "(built)" if MPS_BUILT else "(runtime)"
                print(f"Using Apple GPU via MPS {backend_info} backend")
        else:
            device = torch.device("cpu")
            if self.verbose:
                print("Using CPU.")
        
        self.device = device
        return device

    def _clean_gpu_memory(self):
        """
        Forcefully clear all allocated memory on the GPU.
        """
        
        if TORCH_AVAILABLE and self.use_gpu:
            if self.device.type == 'cuda' and torch.cuda.is_available():
                try:
                    # Delete class attributes that might hold GPU memory
                    for attr in ["x_train", "y_train", "noise_train", "x_predict", "noise_redict", 
                                 "model", "models", "likelihood", "likelihoods"]:
                        if hasattr(self, attr):
                            setattr(self, attr, None)

                    # Call garbage collector and clear GPU cache
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Optionally synchronize and reset memory stats
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                except Exception as e:
                    if self.verbose:
                        print(f"Error occurred while cleaning CUDA memory: {e}")
            elif self.device.type == 'mps' and MPS_AVAILABLE:
                # No explicit cache clear API for MPS; drop references and GC
                try:
                    for attr in ["x_train", "y_train", "noise_train", "x_predict", "noise_redict", 
                                 "model", "models", "likelihood", "likelihoods"]:
                        if hasattr(self, attr):
                            setattr(self, attr, None)
                    gc.collect()
                except Exception as e:
                    if self.verbose:
                        print(f"Error occurred while cleaning MPS memory: {e}")
    
    def _clean_model(self):
        """
        Clean the model and likelihood attributes to free up GPU memory.
        """
        for attr in ["x_train", "y_train", "noise_train", "x_predict", "noise_redict", 
                     "model", "models", "likelihood", "likelihoods", "mll"]:
            if hasattr(self, attr):
                setattr(self, attr, None)
            
        # Now systematically search for and clean any tensor attributes
        for attr_name in list(vars(self).keys()):
            attr_val = getattr(self, attr_name)
            
            # Clean PyTorch tensors
            if isinstance(attr_val, torch.Tensor) and (attr_val.device.type == 'cuda' or attr_val.device.type == 'mps'):
                # Either set to None or move to CPU depending on size
                if attr_val.numel() > 10000:  # Large tensors - just delete
                    setattr(self, attr_name, None)
                else:  # Small tensors - move to CPU
                    try:
                        setattr(self, attr_name, attr_val.cpu().clone().detach())
                    except:
                        setattr(self, attr_name, None)
            
            # Clean PyTorch modules
            elif isinstance(attr_val, torch.nn.Module):
                setattr(self, attr_name, None)
            
            # Clean lists or tuples that might contain tensors
            elif isinstance(attr_val, (list, tuple)) and len(attr_val) > 0:
                # Check if it contains tensors and clean if needed
                if any(isinstance(item, torch.Tensor) for item in attr_val):
                    setattr(self, attr_name, None)
        
        # Force garbage collection
        gc.collect()
        
        # Clean GPU cache thoroughly
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                except Exception as e:
                    if self.verbose:
                        print(f"Error clearing CUDA memory: {e}")
            # For MPS there is no explicit empty_cache, so nothing to do beyond GC
        
        
    def _to_tensor(self, data):
        """
        Convert data to a tensor and move it to the appropriate device.

        Args:
            data (np.ndarray): The data to convert.

        Returns:
            torch.Tensor or None: The converted data as a tensor, or None if input is None.
        """
        
        if data is None:
            return None
        return torch.tensor(data, dtype=self.dtype, device=self.device)
    
    def _estimate_lengthscale(self):
        """
        Estimate a reasonable initial lengthscale for Gaussian Process training.
        """
        
        x = self.x_train
        x = torch.sort(x).values  # Ensure x is sorted
        x_range = x.max() - x.min()
        x_spacing = torch.mean(torch.diff(x))
        # Larger lengthscale for smooth functions
        lengthscale_guess = max(x_range * 0.1, x_spacing * 5)
        
        # Compute the median pairwise Euclidean distance for lengthscale estimation.
        try: x = x.numpy()
        except: x = x.cpu().numpy()
        x = np.array(x).reshape(-1, 1)  # Ensure it's 2D
        pairwise_dists = pdist(x, metric="euclidean")  # Compute pairwise distances
        
        # self.lengthscale =  list(np.ones(self.num_kernels) * np.median(pairwise_dists)) # Return the median distance
        # self.lengthscale =  np.median(pairwise_dists) # Return the median distance
        
        quantiles = np.linspace(self.lengthscale_quantile_low, self.lengthscale_quantile_high, self.num_kernels)
        self.lengthscale = list(np.quantile(pairwise_dists, quantiles)) 
        
    def _set_lengthscale(self, lengthscale):
        """
        Set the lengthscale of the GP model.

        Args:
            lengthscale (float or list of floats): The lengthscale(s) to set.
        """
        if self.num_kernels == 1:  # Single kernel case
            try: self.model.covar_module.initialize(lengthscale=lengthscale)
            except: pass

            try: self.model.covar_module.base_kernel.initialize(lengthscale=lengthscale)
            except: pass

            # If there are multiple kernels, set the same lengthscale for all
            if hasattr(self.model.covar_module, "kernels"):
                for kernel in self.model.covar_module.kernels:
                    try: kernel.base_kernel.lengthscale = lengthscale
                    except: pass
                    
                    try: kernel.lengthscale = lengthscale
                    except: pass
                    
        elif self.num_kernels > 1:
            if type(lengthscale) is not list or self.num_kernels != len(lengthscale):
                Warning("Number of lengthscale values must match the number of kernels. Setting lengthscale as a list with value number equal to num_kernel")
                lengthscale = [lengthscale] * self.num_kernels
            
            if hasattr(self.model.covar_module, "kernels"):
                for i, kernel in enumerate(self.model.covar_module.kernels):
                    try:kernel.base_kernel.lengthscale = lengthscale[i]
                    except:pass
                    
                    try:kernel.lengthscale = lengthscale[i]
                    except:pass

        # # Print updated lengthscales
        # if hasattr(self.model.covar_module, "kernels"):
        #     for i, kernel in enumerate(self.model.covar_module.kernels):
        #         print(f"Kernel {i} lengthscale: {kernel.base_kernel.lengthscale.item()}")
        # else:
        #     print(f"Single Kernel lengthscale: {self.model.covar_module.base_kernel.lengthscale.item()}")

            
    def _get_lengthscale(self):
        """
        Get the lengthscale(s) of the GP model.

        Returns:
            float or list of floats: The current lengthscale(s) of the GP model.
        """
        if hasattr(self.model.covar_module, "kernels"):  # Multiple kernel case
            lengthscales = []
            for kernel in self.model.covar_module.kernels:
                try:    lengthscales.append(kernel.base_kernel.lengthscale.item())
                except: pass
                
                try:    lengthscales.append(kernel.lengthscale.item())
                except: pass
            return lengthscales

        try: return self.model.covar_module.lengthscale.item()  # Single kernel case
        except: pass

        try: return self.model.covar_module.base_kernel.lengthscale.item()  # ScaleKernel case
        except: pass
        
        return None  # Return None if lengthscale retrieval fails

    def initialize_denoiser(self, **kwargs):
        """
        Initialize the denoiser with provided training and prediction data.

        Args:
            **kwargs: Keyword arguments containing x, y, noise, x_predict, noise_predict, and weights.
        """
        
        self.x_train0 = kwargs.get("x", None)
        self.y_train0 = kwargs.get("y", None)
        self.noise_train0 = kwargs.get("noise", None)
        self.x_predict0 = kwargs.get("x_predict", self.x_train0)
        self.noise_predict0 = kwargs.get("noise_predict", self.noise_train0)
        self.weights0 = kwargs.get("weights", None)
        self.y_reference0 = kwargs.get("y_reference", None)        
        
        if self.x_train0 is None or self.y_train0 is None or self.noise_train0 is None:
            raise ValueError("Missing required arguments: x, y, or noise.")
                         
    def initialize_training_data(self, x_train, y_train, noise_train, x_predict=None, noise_predict=None, weights=None):
        """
        Initialize training and test data for the denoiser.

        Args:
            x_train (torch.Tensor): Input training data (1D tensor).
            y_train (torch.Tensor): Output training data (1D tensor).
            noise_train (torch.Tensor): Noise associated with the training data.
            x_predict (torch.Tensor, optional): Test input data. Defaults to None.
            noise_predict (torch.Tensor, optional): Noise associated with test data. Defaults to None.
            weights (torch.Tensor, optional): Weights for the training data. Defaults to None.
        """
        
        # self._initialize_device()
        self._select_device()
        self._clean_gpu_memory()
        
        self.x_train = self._to_tensor(x_train)
        self.y_train = self._to_tensor(y_train)
        self.noise_train = torch.clamp(self._to_tensor(noise_train*self.noise_scale)**2, min=1e-6)
        if x_predict is not None:
            self.x_predict = self._to_tensor(x_predict)
        if noise_predict is not None:
            self.noise_predict = torch.clamp(self._to_tensor(noise_predict*self.noise_scale)**2, min=1e-6)
        if weights is not None:
            self.weights = self._to_tensor(weights)
    
        # Ensure 2D format for time-series data
        if self.y_train.dim() == 1:
            self.y_train = self.y_train[:, None]
        if self.noise_train.dim() == 1:
            self.noise_train = self.noise_train[:, None]
        if self.noise_predict is not None and self.noise_predict.dim() == 1:
            self.noise_predict = self.noise_predict[:, None]
        if self.weights is not None and self.weights.dim() == 1:
            self.weights = self.weights[:, None]
                
    def initialize_model(self, time=0):
        """
        Initialize the Gaussian Process model and likelihood.

        Raises:
            ValueError: If training data is not properly initialized.
        """
        
        if self.likelihood_model == 'FixedNoiseGaussianLikelihood':
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=self.noise_train[:, time],
                learn_additional_noise=self.learn_additional_noise
            ).to(self.device, dtype=self.dtype)
        elif self.likelihood_model == 'GaussianLikelihood':
            # self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.Interval(1e-6,1)).to(self.device)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device, dtype=self.dtype)
        
        if self.use_DKL:
            feature_extractor = LargeFeatureExtractor(2)
            feature_extractor = feature_extractor.to(self.device, dtype=self.dtype)
            self.model = GPModel(self.x_train, self.y_train[:, time], self.likelihood, self.lengthscale_prior, self.mattern_nu, self.num_kernels, feature_extractor).to(self.device, dtype=self.dtype)
        else:
            self.model = GPModel(self.x_train, self.y_train[:, time], self.likelihood, self.lengthscale_prior, self.mattern_nu, self.num_kernels).to(self.device, dtype=self.dtype)
        self._set_lengthscale(self.lengthscale)
        
    def initialize_training_metrics(self):
        """
        Initialize training metrics container.

        Returns:
            TrainingMetrics: An object to store training metrics like loss, lengthscale, and noise.
        """

        metrics = TrainingMetrics()
        metrics.lengthscale = torch.zeros(self.training_iter, device=self.device)
        metrics.noise = torch.zeros(self.training_iter, device=self.device)
        metrics.loss = torch.zeros(self.training_iter, device=self.device)
        return metrics
    
    # --------------------------- data normalization ---------------------------
    def normalize_data(self):
        """
        Normalize input data for training and prediction.

        This includes normalizing x to [-1, 1] and scaling y and noise with respect to the range of y.

        Raises:
            ValueError: If x_train or y_train is not initialized.
        """
        
        # Normalize x_train to [-1, 1]
        x_min, x_max = self.x_train.min(), self.x_train.max()
        self.x_train = 2 * (self.x_train - x_min) / (x_max - x_min) - 1
        self.norm_params['x_train'] = (x_min, x_max)

        # Normalize x_predict to [-1, 1]
        if self.x_predict is not None:
            # The normalization should be the same as for x_train
            # x_min, x_max = self.x_predict.min(), self.x_predict.max()
            self.x_predict = 2 * (self.x_predict - x_min) / (x_max - x_min) - 1
            self.norm_params['x_predict'] = (x_min, x_max)

        # Normalize y_train: Subtract mean and scale by range
        y_mean = self.y_train.mean()
        y_range = self.y_train.max() - self.y_train.min()
        self.y_train = (self.y_train - y_mean) / y_range
        self.norm_params['y_train'] = (y_mean, y_range)

        # Scale noise_train by the same factor (range of y_train squared for variance scaling)
        self.noise_train /= y_range**2
        self.norm_params['noise_train'] = y_range
        
        if self.noise_predict is not None:
            self.noise_predict /= y_range**2
            self.norm_params['noise_predict'] = y_range

    def denormalize_data(self):
        """
        Denormalize training and prediction data using stored normalization parameters.
        """
        # Denormalize x_train
        if 'x_train' in self.norm_params:
            x_min, x_max = self.norm_params['x_train']
            self.x_train = 0.5 * (self.x_train + 1) * (x_max - x_min) + x_min

        # Denormalize x_predict
        if 'x_predict' in self.norm_params:
            x_min, x_max = self.norm_params['x_predict']
            self.x_predict = 0.5 * (self.x_predict + 1) * (x_max - x_min) + x_min

        # Denormalize y_train
        if 'y_train' in self.norm_params:
            y_mean, y_range = self.norm_params['y_train']
            self.y_train = self.y_train * y_range + y_mean

        # Denormalize noise_train
        if 'noise_train' in self.norm_params:
            y_range = self.norm_params['noise_train']
            self.noise_train *= y_range**2
    
        # Denormalize noise_predict
        if 'noise_predict' in self.norm_params:
            y_range = self.norm_params['noise_predict']
            self.noise_predict *= y_range**2
            
    def denormalize(self, y_denoised, y_err, y_noise):
        """
        Denormalize the predicted outputs.

        Args:
            y_denoised (np.ndarray): The denoised signal.
            y_err (np.ndarray): Error estimates.
            y_noise (np.ndarray): Noise estimates.

        Returns:
            tuple: Denormalized versions of y_denoised, y_err, and y_noise.
        """
        
        # Retrieve y normalization parameters
        y_mean, y_range = self.norm_params['y_train']
        y_mean, y_range = y_mean.cpu().numpy(), y_range.cpu().numpy()
        y_denoised = y_denoised * y_range + y_mean
        y_noise = y_noise * y_range
        y_err = y_err * y_range
        return y_denoised, y_err, y_noise
    
    def train_joint_temporal(self):
        """Advanced joint temporal training with better hyperparameter sharing."""
        if self.y_train.dim() == 1:
            self.y_train = self.y_train[:, None]
        
        n_time = self.y_train.shape[1]
        
        # Initialize models for all time instances
        models = []
        likelihoods = []
        optimizers = []
        
        for t in range(n_time):
            self.initialize_model(time=t)
            models.append(copy.deepcopy(self.model))
            likelihoods.append(copy.deepcopy(self.likelihood))
            optimizers.append(torch.optim.Adam(models[t].parameters(), lr=self.lr))
        
        self.training_metrics = self.initialize_training_metrics()

        # Training loop with shared hyperparameter updates
        for i in range(self.training_iter):
            total_loss = 0
            lengthscales = []
            
            # Train each model individually
            for t in range(n_time):
                optimizers[t].zero_grad()
                
                if self.mll == 'ExactMarginalLogLikelihood':
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihoods[t], models[t])
                elif self.mll == 'LeaveOneOutPseudoLikelihood':
                    mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihoods[t], models[t]).to(device=self.device, dtype=self.dtype)    
                else:
                    print(f'Unknown MLL {self.mll}, defaulting to ExactMarginalLogLikelihood')
                    
                output = models[t](self.x_train)
                loss_t = -mll(output, self.y_train[:, t])
                
                # Add temporal smoothness penalty
                if t > 0:
                    pred_t = output.mean
                    pred_prev = models[t-1](self.x_train).mean
                    temporal_penalty = 0.01 * torch.mean((pred_t - pred_prev)**2)
                    loss_t += temporal_penalty
                
                loss_t.backward()
                optimizers[t].step()
                
                if self.dont_update_lengthscale and i < self.dont_update_lengthscale_iters:
                    self._set_lengthscale_for_model(models[t], self.lengthscale)


                total_loss += loss_t.item()
                lengthscales.append(self._get_lengthscale_from_model(models[t]))
            
            # Share lengthscale information across models
            if i % 10 == 0 and i > 0:  # Every 10 iterations
                avg_lengthscale = np.mean(lengthscales)
                for t in range(n_time):
                    self._set_lengthscale_for_model(models[t], avg_lengthscale)
            
            if self.verbose >= 2:
                avg_ls = np.mean(lengthscales)
                print(f'Joint iter {i+1}/{self.training_iter} - '
                    f'avg_lengthscale: {avg_ls:.6f} - '
                    f'Total Loss: {total_loss:.6f}')

            # Store metrics
            self.training_metrics.lengthscale[i] = np.mean(lengthscales) if self.num_kernels == 1 else self._get_lengthscale()[0]
            self.training_metrics.loss[i] = total_loss
            if self.likelihood_model == 'FixedNoiseGaussianLikelihood':
                self.training_metrics.noise[i] = self.likelihood.noise.mean()

        self.models = models
        self.likelihoods = likelihoods

    def _get_lengthscale_from_model(self, model):
        """Get lengthscale from a specific model."""
        if hasattr(model.covar_module, 'base_kernel'):
            return model.covar_module.base_kernel.lengthscale.item()
        else:
            return model.covar_module.lengthscale.item()

    def _set_lengthscale_for_model(self, model, lengthscale):
        """Set lengthscale for a specific model."""
        if hasattr(model.covar_module, 'base_kernel'):
            model.covar_module.base_kernel.lengthscale = lengthscale
        else:
            model.covar_module.lengthscale = lengthscale
            
    def train(self):
        """
        Train the Gaussian Process model to optimize hyperparameters.
        """
        
        if self.joint_temporal_optimization:
            self.train_joint_temporal()
            return
        
        # Ensure y_train is 2D (time-series data handling)
        if self.y_train.dim() == 1:
            self.y_train = self.y_train[:, None]
        
        # The code expects a 2D time-series array so each model needs to be stored for every single time-instance
        self.models = []
        self.likelihoods = []
        self.lengthscales = []

        # Loop over time instances
        for t in range(self.y_train.shape[1]):
            
            # Initialize loop termination criteria
            patience = 20  # Number of iterations to wait for improvement
            tolerance = 1e-4  # Minimum change to consider as improvement
            no_improvement = 0
            previous_loss = None

            # Initialize the model for the first time instance and reuse for subsequent instances
            self.initialize_model(time=t)
            self.training_metrics = self.initialize_training_metrics()
            
            if t > 0:  # Use the previous model's parameters
                print(f'Denoising time instance {t}')
                # self._set_lengthscale(self.lengthscales[-1])
                # self.likelihood.noise = self.models[-1].likelihood.noise.clone()
                # previous_mean = self.models[-1](self.x_train).mean
                # self.model.mean_module.initialize(constant=previous_mean.mean().item())
    
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_computations(log_prob=False, solves=False, covar_root_decomposition=False):
                self.model.train()
                self.likelihood.train()
                
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                if self.mll == 'LeaveOneOutPseudoLikelihood':                
                    mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(self.likelihood, self.model).to(device=self.device, dtype=self.dtype)      
                elif self.mll == 'ExactMarginalLogLikelihood':
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                else:
                    print(f'Unknown MLL {self.mll}, defaulting to ExactMarginalLogLikelihood')
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
                    
                for i in range(self.training_iter):
                    optimizer.zero_grad()
                    output = self.model(self.x_train)
                    loss = -mll(output, self.y_train[:, t])
                    loss.backward()
                    optimizer.step()

                    if self.dont_update_lengthscale and i < self.dont_update_lengthscale_iters:
                        self._set_lengthscale(self.lengthscale)

                    # Store metrics
                    self.training_metrics.lengthscale[i] = self._get_lengthscale() if self.num_kernels == 1 else self._get_lengthscale()[0]
                    self.training_metrics.loss[i] = loss.item()
                    if self.likelihood_model == 'FixedNoiseGaussianLikelihood':
                        self.training_metrics.noise[i] = self.likelihood.noise.mean()

                    if self.verbose >= 2:
                        print(f'Iter {i+1}/{self.training_iter} - lengthscale: {self._get_lengthscale() if self.num_kernels == 1 else self._get_lengthscale()[0]:.6f} - Loss: {loss.item():.6f} - Noise: {self.likelihood.noise.mean():.6f}')

                    # Check for convergence
                    if previous_loss is not None and self.auto_stop_training:
                        if abs(loss.item() - previous_loss) < tolerance:
                            no_improvement += 1
                        else:
                            no_improvement = 0

                        if no_improvement >= patience:
                            if self.verbose: print(f'Converged after {i+1} iterations.')
                            break  # Exit inner loop but not the time loop
                    previous_loss = loss.item()

                # Store outputs for this time instance
                self.models.append(copy.deepcopy(self.model))
                self.likelihoods.append(copy.deepcopy(self.likelihood))
                self.lengthscales.append(copy.deepcopy(self._get_lengthscale()))
    
    def predict(self):
        """
        Make predictions using the trained Gaussian Process model.

        Returns:
            tuple: Predicted mean values, errors, and noise estimates.
        """
        
        # the code is designed to accept time-series (2D) data
        y_mean = torch.zeros_like(self.noise_predict)
        y_error = torch.zeros_like(self.noise_predict)
        y_noise = torch.zeros_like(self.noise_predict)
        
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            for t in range(self.y_train.shape[1]):
                
                # extract the mode for a given time instance
                self.model = self.models[t]
                self.likelihood = self.likelihoods[t]
                
                # inference on large datasets can be too memory intensive
                if self.x_predict.shape[0] > 40000:
                    self.model = self.model.to('cpu')
                    self.likelihood = self.likelihood.to('cpu')
                    self.x_predict = self.x_predict.to('cpu')
                    self.noise_predict = self.noise_predict.to('cpu')
                
                self.model.eval()
                self.likelihood.eval()
                
                # update model lengtscale
                self.lengthscale = self._get_lengthscale()
                    
                # extract noise, error and mean
                with torch.no_grad():    
                           
                    # confidence_region() gives 2*std bounds by default
                    observed_pred_confidence = self.model(self.x_predict)
                    lower, upper = observed_pred_confidence.confidence_region()
                    y_error[:, t] = (upper - lower) / 4  # Convert 2*std to std
                    
                    # predict the signal based on the likelihood model
                    if self.prediction_mode == 1:
                        observed_pred = self.likelihood(self.model(self.x_predict))
                        
                    elif self.prediction_mode == 2:
                        observed_pred = self.likelihood(self.model(self.x_predict), noise=self.noise_predict[:, t])
                        
                    elif self.prediction_mode == 3:
                        observed_pred = self.model(self.x_predict)
                                
                    # predict mean y signal
                    y_mean[:, t] = observed_pred.mean

                    # noise extraction based on likelihood model and settings
                    if self.likelihood_model == 'GaussianLikelihood':
                        # GaussianLikelihood: learned noise parameter
                        y_noise[:, t] = self.likelihood.noise.sqrt()
                        
                    elif self.likelihood_model == 'FixedNoiseGaussianLikelihood':
                        if self.learn_additional_noise:
                            # Learning additional noise on top of fixed noise
                            # The total observation noise includes both fixed and learned components
                            if hasattr(self.likelihood, 'second_noise'):
                                # Total noise = sqrt(fixed_noise + additional_learned_noise)
                                total_noise_var = self.noise_predict[:, t] + self.likelihood.second_noise
                                y_noise[:, t] = total_noise_var.sqrt()
                            else:
                                # Use the prediction variance which includes all noise sources
                                y_noise[:, t] = observed_pred.variance.sqrt()
                        else:
                            # Using only the fixed noise (no additional learning)
                            y_noise[:, t] = self.noise_predict[:, t].sqrt()
                    
                    if self.x_predict.shape[0] > 40000:
                        self.x_predict = self.x_predict.to(self.device)
                        self.noise_predict = self.noise_predict.to(self.device)
                
        return y_mean, y_error, y_noise

        
    # --------------------------- denoising ---------------------------    
    def denoise_with_downsampling(self, downsampling_pts=None, downsampling_method=None, smoothness=None):
        """
        Perform GP denoising with optional downsampling.

        Args:
            downsampling_pts (int, optional): Number of points to downsample to.
            downsampling_method (str, optional): Downsampling method to use.
            smoothness (np.ndarray, optional): Smoothness values for downsampling.

        Returns:
            tuple: Denoised signal, error, and noise arrays.
        """
        
        if downsampling_pts is None or downsampling_method is None:
            y_denoised, y_error, y_noise = self.denoise()
            return y_denoised, y_error, y_noise
        
        num_samples = np.min([downsampling_pts, len(self.x_train0)]).astype(int)        

        if num_samples == len(self.x_train0):
            y_denoised, y_error, y_noise = self.denoise()
            print(f'GP denoising with downsampling. Using {num_samples} data points out of {len(self.x_train0)}')

        else:
            # refine noise estimation before downsampling
            # if self.refine_noise_estimate:
            #     self.noise_train0 = self.refine_noise_estimate_with_gp()
            #     self.noise_predict0 = interp1d(self.x_train0, self.noise_train0, kind='linear', fill_value='extrapolate', axis=0)(self.x_predict0)
            #     self.refine_noise_estimate = False
    
            print(f'GP denoising with downsampling. Using {num_samples} data points out of {len(self.x_train0)}')

            # store data before downsampling
            self.x_train0_bck, self.y_train0_bck, self.noise_train0_bck, self.weights0_bck, self.noise_window_gp_bck = \
                self.x_train0, self.y_train0, self.noise_train0, self.weights0, self.noise_window_gp 

            # downsample the training data
            self.x_train0, self.y_train0, self.noise_train0, self.weights0 = downsample_data(self.x_train0, self.y_train0, self.noise_train0, self.weights0, 
                                                                                    method=downsampling_method, 
                                                                                    num_samples=num_samples, 
                                                                                    smoothness=smoothness)

            # adjust the noise estimation window size (if it will be used at all)
            self.noise_window_gp = self.noise_window_gp * len(self.x_train0) // len(self.x_train0_bck)

            # denoise the downsampled data
            y_denoised, y_error, y_noise = self.denoise()
            
            # restore the data
            self.x_train0, self.y_train0, self.noise_train0, self.weights0, self.noise_window_gp = \
                self.x_train0_bck, self.y_train0_bck, self.noise_train0_bck, self.weights0_bck, self.noise_window_gp_bck

            # refine the upsampled estimate (this will use all data as well as the refined lengthscale estimate)
            if self.refine_downsampled_estimate:
                # try on the GPU first, if it fails, try on the CPU
                state = self._get_current_state()
                self.training_iter = self.refine_downsampled_estimate_iters
                self.refine_noise_estimate = False
                try:        
                    y_denoised, y_error, y_noise = self.denoise()
                except:
                    self.use_gpu = False            
                    y_denoised, y_error, y_noise = self.denoise()        
                self.restore_state(state)
        
        return y_denoised, y_error, y_noise
        
    def denoise(self):
        """
        Denoise the data using Gaussian Processes.
        
        Returns:
            tuple: Denoise data, error and noise arrays (y_denoised, y_err, y_noise).        
        """
        
        # refine noise estimation using GP
        if self.refine_noise_estimate:
            self.noise_train0 = self.refine_noise_estimate_with_gp()
            self.noise_predict0 = interp1d(self.x_train0, self.noise_train0, kind='linear', fill_value='extrapolate', axis=0)(self.x_predict0)
            
        # perform windowed GP denoising
        if self.denoise_with_windows_flag:
            y_denoised, y_error, y_noise, lengthscales = self.denoise_with_windows(self.window_size, self.overlap_factor)
            return y_denoised, y_error, y_noise
        
        # initialize the training data tensors
        self.initialize_training_data(self.x_train0, self.y_train0, self.noise_train0, self.x_predict0, self.noise_predict0, self.weights0)
        
        # normalize the training data to improve robustness of GP training
        if self.normalize_training_data:
            self.normalize_data()
        
        # estimate the lengthscale if needed  
        if self.estimate_lengthscale:
            self._estimate_lengthscale()           
           
        # train the GP model and predict the denoised data
        print(f'GP denoising. Using {len(self.x_train)} data points for training out of {len(self.x_predict)}')
        self.train()                   
        y_denoised, y_err, y_noise = self.predict()
        y_denoised, y_err, y_noise = y_denoised.cpu().numpy(), y_err.cpu().numpy(), y_noise.cpu().numpy()
        
        # undo the normalization of the training/test data as well as the GP prediction output
        if self.normalize_training_data:
            self.denormalize_data()
            y_denoised, y_err, y_noise = self.denormalize(y_denoised, y_err, y_noise)
          
        # plot the results for comparison
        if self.verbose >= 1:
            self.plot_denoising_results(self.x_train.cpu(), self.y_train.cpu(), self.x_predict.cpu(), y_denoised, y_err)

        self._clean_gpu_memory()
        return y_denoised, y_err, y_noise
    
    def denoise_with_windows(self, window_size, overlap_factor=1):
        """
        Perform GP denoising on overlapping windows with optional parallel processing.

        Args:
            window_size (int): The size of each window for denoising.
            overlap_factor (int): Overlap factor for the windows.

        Returns:
            tuple: Stitched outputs of denoising (y_stitched, error_stitched, noise_stitched).
        """
        
        # Store current denoiser state
        state0 = self._get_current_state()
        
        # If data is small, don't use GPU
        if window_size < 500:
            self.use_gpu = False
        
        # Initialize the training data tensors
        self.initialize_training_data(self.x_train0, self.y_train0, self.noise_train0, 
                                    self.x_predict0, self.noise_predict0, self.weights0)
        
        # Normalize the training data to improve robustness of GP training
        if self.normalize_training_data:
            self.normalize_data()

        # Estimate the lengthscale if needed
        if self.estimate_lengthscale:
            self._estimate_lengthscale()

        # Calculate number of windows
        effective_step = window_size / overlap_factor
        window_num = int(np.ceil((len(self.x_predict) - window_size) / effective_step)) + 1
        
        # Decide on processing method
        if self.parallel_windows and window_num > 1:
            print(f'GP denoising with {window_num} windows using parallel processing ({self.parallel_backend})')
            
            # Prepare window data
            window_data_list = self._prepare_window_data(window_size, overlap_factor)
            
            # Choose parallel processing method
            if self.parallel_backend == 'joblib':
                results = self._process_windows_joblib(window_data_list)
            elif self.parallel_backend == 'multiprocessing':
                results = self._process_windows_multiprocessing(window_data_list)
            elif self.parallel_backend == 'threading':
                results = self._process_windows_threading(window_data_list)
            else:
                print(f"Unknown parallel backend '{self.parallel_backend}', falling back to sequential processing")
                return self._denoise_with_windows_sequential(window_size, overlap_factor, state0)
            
            # Combine results
            return self._combine_window_results(results, state0)
        
        else:
            # Sequential processing (original method)
            return self._denoise_with_windows_sequential(window_size, overlap_factor, state0)
        
    def _process_single_window(self, window_data):
        """
        Process a single window independently - designed for parallel processing.
        
        Args:
            window_data (dict): Contains all data needed for one window
        
        Returns:
            dict: Results from processing this window
        """
        # Extract data from the dictionary
        x_train_window = window_data['x_train']
        y_train_window = window_data['y_train'] 
        noise_train_window = window_data['noise_train']
        x_predict_window = window_data['x_predict']
        noise_predict_window = window_data['noise_predict']
        crop_indices = window_data['crop_indices']
        denoiser_state = window_data['denoiser_state']
        
        # Create a new denoiser instance for this process
        denoiser = GPDenoiser()
        denoiser.restore_state(denoiser_state)
        denoiser.use_gpu = self.use_gpu
        
        try:
            # Initialize data for this window
            denoiser.initialize_training_data(
                x_train_window, y_train_window, noise_train_window,
                x_predict_window, noise_predict_window
            )
            
            # Train and predict
            denoiser.train()
                
            y_denoised, y_error, y_noise = denoiser.predict()
            lengthscale = denoiser._get_lengthscale()
            
            # Clean GPU
            denoiser._clean_gpu_memory()
            
            # Convert to numpy and return results
            return {
                'crop_indices': crop_indices,
                'y_denoised': y_denoised.cpu().numpy() if hasattr(y_denoised, 'cpu') else y_denoised,
                'y_error': y_error.cpu().numpy() if hasattr(y_error, 'cpu') else y_error,
                'y_noise': y_noise.cpu().numpy() if hasattr(y_noise, 'cpu') else y_noise,
                'lengthscale': lengthscale if isinstance(lengthscale, (int, float)) else lengthscale[0],
                'success': True
            }
        except Exception as e:
            print(f"Error processing window {crop_indices}: {e}")
            return {
                'crop_indices': crop_indices,
                'y_denoised': None,
                'y_error': None,
                'y_noise': None,
                'lengthscale': None,
                'success': False,
                'error': str(e)
            }

    def _prepare_window_data(self, window_size, overlap_factor):
        """
        Prepare window data for parallel processing.
        
        Returns:
            list: List of window data dictionaries
        """
        # Calculate windows
        effective_step = window_size / overlap_factor
        window_num = int(np.ceil((len(self.x_predict) - window_size) / effective_step)) + 1
        
        window_data_list = []
        for n in range(window_num):
            step_size = int(window_size / overlap_factor)
            crop_start = max(n * step_size, 0)
            crop_end = min(crop_start + window_size, len(self.x_predict))
            crop = slice(crop_start, crop_end)
            
            # Extract data for this window
            x_predict_window = self.x_predict[crop]
            noise_predict_window = self.noise_predict[crop, :]
            
            # Find corresponding training data
            x_predict_min, x_predict_max = x_predict_window[0].item(), x_predict_window[-1].item()
            crop_train = (self.x_train >= x_predict_min) & (self.x_train <= x_predict_max)
            
            window_data = {
                'x_train': self.x_train[crop_train].cpu().numpy(),
                'y_train': self.y_train[crop_train, :].cpu().numpy(),
                'noise_train': self.noise_train[crop_train, :].cpu().numpy(),
                'x_predict': x_predict_window.cpu().numpy(),
                'noise_predict': noise_predict_window.cpu().numpy(),
                'crop_indices': (crop_start, crop_end),
                'denoiser_state': self._get_current_state()
            }
            window_data_list.append(window_data)
        
        return window_data_list

    def _combine_window_results(self, results, state0):
        """
        Combine results from parallel window processing.
        
        Args:
            results (list): List of result dictionaries from parallel processing
            state0 (dict): Original denoiser state to restore
        
        Returns:
            tuple: Combined results (y_stitched, error_stitched, noise_stitched, lengthscales)
        """
        # Initialize output arrays
        y_stitched = np.zeros([self.x_predict.shape[0], self.y_train0.shape[1]])
        noise_stitched = np.zeros([self.x_predict.shape[0], self.y_train0.shape[1]])
        error_stitched = np.zeros([self.x_predict.shape[0], self.y_train0.shape[1]])
        weights = np.zeros([self.x_predict.shape[0], self.y_train0.shape[1]])
        lengthscales = np.zeros([self.x_predict.shape[0]])
        
        # Combine results
        for result in results:
            if not result['success']:
                print(f"Skipping failed window {result['crop_indices']}: {result.get('error', 'Unknown error')}")
                continue
                
            crop_start, crop_end = result['crop_indices']
            crop = slice(crop_start, crop_end)
            
            y_stitched[crop, :] += result['y_denoised']
            noise_stitched[crop, :] += result['y_noise']
            error_stitched[crop, :] += result['y_error']
            lengthscales[crop] += result['lengthscale']
            weights[crop, :] += 1
        
        # Blend overlapping regions
        y_stitched, noise_stitched, error_stitched = blend_overlapping_windows(
            y_stitched, weights, noise_stitched, error_stitched)
        
        # Undo normalization
        if self.normalize_training_data:
            self.denormalize_data()
            y_stitched, error_stitched, noise_stitched = self.denormalize(
                y_stitched, error_stitched, noise_stitched)
        
        # Fill gaps using interpolation
        missing_mask = weights == 0
        if missing_mask.any():
            x_valid = self.x_predict[~missing_mask].cpu().numpy()
            x_missing = self.x_predict[missing_mask].cpu().numpy()
            
            if len(x_valid) > 0:  # Only interpolate if we have valid data
                f_y = interp1d(x_valid, y_stitched[~missing_mask, :], kind='slinear', 
                            fill_value='extrapolate', bounds_error=False)
                f_noise = interp1d(x_valid, noise_stitched[~missing_mask, :], kind='slinear', 
                                fill_value='extrapolate', bounds_error=False)
                f_error = interp1d(x_valid, error_stitched[~missing_mask, :], kind='slinear', 
                                fill_value='extrapolate', bounds_error=False)
                
                y_stitched[missing_mask, :] = f_y(x_missing)
                noise_stitched[missing_mask, :] = f_noise(x_missing)
                error_stitched[missing_mask, :] = f_error(x_missing)
        
        # Restore state
        self.restore_state(state0)
        
        return y_stitched, error_stitched, noise_stitched, lengthscales

    

    def _process_windows_joblib(self, window_data_list):
        """Process windows using joblib parallel processing."""
        with joblib.parallel_backend('loky', n_jobs=self.max_workers):
            results = Parallel()(
                delayed(self._process_single_window)(window_data) 
                for window_data in window_data_list
            )
        return results

    def _process_windows_multiprocessing(self, window_data_list):
        """Process windows using multiprocessing."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._process_single_window, window_data_list))
        return results

    def _process_windows_threading(self, window_data_list):
        """Process windows using threading (good for I/O bound operations)."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._process_single_window, window_data_list))
        return results



    def _denoise_with_windows_sequential(self, window_size, overlap_factor, state0):
        """Original sequential window processing method."""
        # Calculate number of windows
        effective_step = window_size / overlap_factor
        window_num = int(np.ceil((len(self.x_predict) - window_size) / effective_step)) + 1

        # Arrays to store the stitched output
        y_stitched = np.zeros([self.x_predict.shape[0], self.y_train0.shape[1]])
        noise_stitched = np.zeros([self.x_predict.shape[0], self.y_train0.shape[1]])
        error_stitched = np.zeros([self.x_predict.shape[0], self.y_train0.shape[1]])
        weights = np.zeros([self.x_predict.shape[0], self.y_train0.shape[1]])
        lengthscales = np.zeros([self.x_predict.shape[0]])
        
        for n in range(window_num):
            print(f'GP denoising with windows. Using window size of {window_size}. Window {n+1}/{window_num}')

            # Define crop region for the current window
            step_size = int(window_size / overlap_factor)
            crop_start = n * step_size
            crop_end = crop_start + window_size

            crop_start = int(max(crop_start, 0))
            crop_end = int(min(crop_end, len(self.x_predict)))
            crop = slice(crop_start, crop_end)
                
            state = self._get_current_state()

            # crop data the data
            self.x_predict = self.x_predict[crop]
            self.noise_predict = self.noise_predict[crop, :]
            
            # "predict" is typically not-masked and can be larger than other arrays (due to cropping)
            # adjust the croping range
            x_predict_min, x_predict_max = self.x_predict[0].item(), self.x_predict[-1].item()
            crop_train = (self.x_train >= x_predict_min) & (self.x_train <= x_predict_max)
            self.x_train = self.x_train[crop_train]
            self.y_train = self.y_train[crop_train, :]
            self.noise_train = self.noise_train[crop_train, :]
            self.weights = self.weights[crop_train, :] if self.weights0 is not None else None
            
            # Train the GP model and predict the denoised data
            self.train()   
            y_denoised, y_error, y_noise = self.predict()
            lengthscale = self._get_lengthscale()
            if type(lengthscale) == list and len(lengthscale) > 1:
                lengthscale = lengthscale[0]
            
            self.restore_state(state)

            # Get data onto CPU
            try: crop = crop.cpu().numpy()
            except: pass
            try: y_denoised = y_denoised.cpu().numpy()
            except: pass
            try: y_error = y_error.cpu().numpy()
            except: pass
            try: y_noise = y_noise.cpu().numpy()
            except: pass

            # Store the denoised results
            y_stitched[crop, :] += y_denoised
            noise_stitched[crop, :] += y_noise
            error_stitched[crop, :] += y_error
            lengthscales[crop] += lengthscale
            weights[crop, :] += 1
        
        # Blend overlapping regions
        y_stitched, noise_stitched, error_stitched = blend_overlapping_windows(
            y_stitched, weights, noise_stitched, error_stitched)
        
        # Undo normalization of the training/test data as well as the GP prediction output
        if self.normalize_training_data:
            self.denormalize_data()
            y_stitched, error_stitched, noise_stitched = self.denormalize(y_stitched, error_stitched, noise_stitched)

        # Fill gaps in y_stitched, noise_stitched, and error_stitched using interpolation
        missing_mask = weights == 0
        if missing_mask.any():
            x_valid = self.x_predict[~missing_mask].cpu().numpy()
            x_missing = self.x_predict[missing_mask].cpu().numpy()

            # Interpolation for each stitched array
            f_y = interp1d(x_valid, y_stitched[~missing_mask, :], kind='slinear', 
                        fill_value=(y_stitched[~missing_mask, :][0], y_stitched[~missing_mask, :][-1]), bounds_error=False)
            f_noise = interp1d(x_valid, noise_stitched[~missing_mask, :], kind='slinear', 
                            fill_value=(noise_stitched[~missing_mask, :][0], noise_stitched[~missing_mask, :][-1]), bounds_error=False)
            f_error = interp1d(x_valid, error_stitched[~missing_mask, :], kind='slinear', 
                            fill_value=(error_stitched[~missing_mask, :][0], error_stitched[~missing_mask, :][-1]), bounds_error=False)

            y_stitched[missing_mask, :] = f_y(x_missing)
            noise_stitched[missing_mask, :] = f_noise(x_missing)
            error_stitched[missing_mask, :] = f_error(x_missing)

        # Restore state
        self.restore_state(state0)

        # Plot the results if verbose
        if self.verbose >= 1:
            plt.figure()
            plt.title('GP Denoising with Windows')
            plt.plot(self.x_train0, self.y_train0, label='Original')
            plt.plot(self.x_predict0, y_stitched, label='Denoised')
            plt.xlabel("X values")
            plt.ylabel("Y values")
            plt.legend()
            plt.show()

        self._clean_gpu_memory()
        return y_stitched, error_stitched, noise_stitched, lengthscales
    
    def _estimate_local_noise_from_residuals(self, residuals, window_method='rms'):
        """
        Estimate local noise from residuals using sliding windows.
        
        Args:
            residuals (np.ndarray): Residuals from GP fit (y_true - y_pred)
            window_method (str): Method for estimating noise in each window
                - 'rms': Root mean square (recommended for GP)
                - 'mad': Median absolute deviation (most robust)
                - 'percentile_68': 68th percentile of absolute residuals
                - 'abs_mean': Simple mean of absolute residuals
        
        Returns:
            np.ndarray: Local noise estimates
        """
        residuals = np.asarray(residuals)
        if residuals.ndim > 1:
            residuals = residuals.squeeze()
            
        n_points = len(residuals)
        noise_estimate = np.zeros_like(residuals)
                
        # window_size = max(50, n_points // 100)  # Adaptive window size                
        window_size = 10
        lower_bound = 0.67 # for N = 10 for rmse

        for i in range(n_points):
            start_idx = i
            end_idx = i + window_size
            window_residuals = residuals[start_idx:end_idx]
            
            if window_method == 'rms':
                # Root mean square - best for GP variance estimates
                noise_estimate[i] = np.sqrt(np.mean(window_residuals**2))
                noise_estimate[i] = noise_estimate[i]*lower_bound
                
            elif window_method == 'mad':
                # Median absolute deviation - most robust
                noise_estimate[i] = np.median(np.abs(window_residuals))
                
            elif window_method == 'percentile_68':
                # 68th percentile (1-sigma equivalent for normal distribution)
                noise_estimate[i] = np.percentile(np.abs(window_residuals), 68)
                
            elif window_method == 'abs_mean':
                # Simple mean of absolute residuals
                noise_estimate[i] = np.mean(np.abs(window_residuals))
                
            else:
                raise ValueError(f"Unknown window_method: {window_method}")
        
        return noise_estimate

    # --------------------------- refine noise estimation using sliding GP windows ---------------------------
    def refine_noise_estimate_with_gp(self):
        """
        Refine the noise estimate using GP regression on sliding windows.

        Returns:
            numpy.ndarray: Refined noise estimates.
        """
        # Store current denoiser state so that it can be unset
        state = self._get_current_state()

        if self.refine_noise_model == 'sharp':
            # fix noise likelihood allows us to pass a noise estimate and then refine it
            self.likelihood_model = 'FixedNoiseGaussianLikelihood'  # FixedNoiseGaussianLikelihood
            self.learn_additional_noise = True
        elif self.refine_noise_model == 'smooth':
            # Set specific params for the noise estimation
            self.likelihood_model = 'GaussianLikelihood'  # GaussianLikelihood        
        
        self.mattern_nu = self.noise_est_mattern_nu
        self.refine_noise_estimate = False
        # self.dont_update_lengthscale = True
        # self.dont_update_lengthscale_iters = np.inf
        self.estimate_lengthscale = True
        self.num_kernels = 1
        # self.mll = 'LeaveOneOutPseudoLikelihood' 

        # We want to update the noises of the training data
        self.x_predict0 = self.x_train0
        self.noise_predict0 = self.noise_train0
        
        for n_times in range(self.gp_refinement_epochs):
            y_estimate, err, noise_estimate, lengthscales = self.denoise_with_windows(self.noise_window_gp, overlap_factor=1)
            
            # The noise estimate from the GP is just a single value for a given window.
            # Instead we can estimate the noise fully via
            if self.refine_noise_model == 'sharp':
                residuals = self.y_train0 - y_estimate
                if self.estimate_local_noise_from_residuals:
                    noise_estimate = self._estimate_local_noise_from_residuals(residuals, self.estimate_local_noise_from_residuals_method)
                else:
                    noise_estimate = abs(residuals)
                
            if n_times != self.gp_refinement_epochs-1:
                sz = self.x_predict0.shape[0] // 200       
                noise_estimate = uniform_filter1d(noise_estimate**2, sz, mode="nearest", axis=0)**0.5 
                self.noise_train0 = noise_estimate
                self.noise_predict0 = interp1d(self.x_train0, self.noise_train0, kind='linear', fill_value='extrapolate', axis=0)(self.x_predict0)
        
        # Filter the noise
        if self.filter_refined_noise_estimate:
            # Apply this function to your noise estimate
            noise_estimate_prefilter = noise_estimate
            noise_estimate = uniform_filter1d(noise_estimate**2, len(noise_estimate) // 10, mode="nearest", axis=0)**0.5 

        # Undo noise refinement setting
        self.restore_state(state)
        
        if self.verbose >= 1:
            plt.figure()
            plt.title('GP noise estimation')
            plt.plot(self.x_train0, self.noise_train0, label='Initial estimate')
            if self.filter_refined_noise_estimate:
                plt.plot(self.x_train0, noise_estimate_prefilter, label='GP estimate before filter')
            plt.plot(self.x_train0, noise_estimate, label='GP estimate')
            plt.xlabel("X values")
            plt.legend()
            plt.show()

            plt.figure()
            plt.title('Data estimate (during GP noise estimation)')
            plt.plot(self.x_train0, self.y_train0, label='Initial estimate')
            plt.plot(self.x_train0, y_estimate, label='GP estimate')
            plt.xlabel("X values")
            plt.legend()
            plt.show()
            
        self._clean_gpu_memory()
        return noise_estimate

    # --------------------------- plotting ---------------------------
    def plot_denoising_results(self, X_train, y_train, X, mean_prediction, std_prediction=None, y_ref=None, fig_size=(12, 8)):
        """
        Plot the results of Gaussian Process denoising.

        Args:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training output data.
            X (np.ndarray): Test input data.
            mean_prediction (np.ndarray): Predicted mean values.
            std_prediction (np.ndarray, optional): Predicted standard deviation values.
            y_ref (np.ndarray, optional): Reference data for comparison.
            fig_size (tuple): Size of the plot (default: (12, 8)).
        """
        
        plt.figure(figsize=fig_size)

        plt.subplot(3, 2, 1)
        lengthscale = self.lengthscale if self.num_kernels == 1 else self.lengthscale[0]
        plt.title(f'Learned lengthscale {lengthscale:.2f}')
        plt.plot(self.training_metrics.lengthscale.cpu().numpy(), label='Lengthscale')
        plt.legend()
        plt.ylabel('Lengthscale')
        plt.xlabel('Training iteration')

        plt.subplot(3, 2, 2)
        loss = self.training_metrics.loss.cpu().numpy()
        plt.title(f'Final loss {loss[-1]:.2f}')
        plt.plot(loss, label='Loss')
        plt.legend()
        plt.ylabel('Training loss')
        plt.xlabel('Training iteration')

        plt.subplot(3, 1, 2)
        plt.title('Gaussian Process Denoising')
        labels = [None] * y_train.shape[1]
        labels[0] = "Observations"
        plt.plot(X_train, y_train, linestyle="None", marker="o", markersize=1, color="tab:blue", alpha=0.3, label=labels)

        if std_prediction is not None:
            plt.fill_between(X.ravel(),
                            np.mean(mean_prediction - std_prediction, axis=1),
                            np.mean(mean_prediction + std_prediction, axis=1),
                            color="tab:orange", alpha=0.3, label=r"95% confidence interval")
        if y_ref is not None:
            labels = [None] * y_train.shape[1]
            labels[0] = "Reference data"
            plt.plot(X, y_ref, label=labels, color="tab:green")
        labels = [None] * y_train.shape[1]
        labels[0] = "Denoised signal"
        plt.plot(X, mean_prediction, label=labels, color="tab:orange")
        plt.legend()
        plt.tight_layout() 
        if (X<0).sum() > 0:
            plt.xlabel('Wavenumber (kspace)')
        else:
            plt.xlabel("Energy")
        plt.ylabel("Absorption")
        
        plt.show()

if TORCH_AVAILABLE:
    class GPModel(gpytorch.models.ExactGP):
        """
        Exact Gaussian Process model using a Matern kernel.
        """
        
        def __init__(self, train_x, train_y, likelihood, lengthscale_prior, nu, num_kernels, feature_extractor=None):
            super().__init__(train_x, train_y, likelihood)
            
            self.mean_module = gpytorch.means.ConstantMean()
            self.feature_extractor = feature_extractor  # Neural network feature transformation
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

            if lengthscale_prior:
                prior=gpytorch.priors.NormalPrior(loc=20,scale=0.5)
            else:
                prior = None
            
            self.covar_module = ScaleKernel(MaternKernel(nu=nu, lengthscale_prior=prior))  
            if num_kernels > 1:
                for i in range(num_kernels-1):
                    self.covar_module += ScaleKernel(MaternKernel(nu=nu, lengthscale_prior=prior))

            # if num_kernels > 1:
            #     self.covar_module += ScaleKernel(RBFKernel())
            # if num_kernels > 2:
            #     self.covar_module += ScaleKernel(PeriodicKernel())
            # if num_kernels > 3:
            #     for i in range(num_kernels-3):
            #         self.covar_module += ScaleKernel(MaternKernel(nu=nu, lengthscale_prior=prior))
                    
                    
            # self.covar_module += ScaleKernel(RBFKernel()) + ScaleKernel(PeriodicKernel()) # no improvement
            # self.covar_module = RBFKernel() + MaternKernel(nu=nu) + PeriodicKernel() # good
            # self.covar_module = ScaleKernel(RBFKernel()) + ScaleKernel(MaternKernel(nu=nu)) + ScaleKernel(PeriodicKernel()) # great
            

        def forward(self, x):
            """
            Forward pass for the Exact Gaussian Process model.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                gpytorch.distributions.MultivariateNormal: Predicted distribution.
            """
            if self.feature_extractor is not None:
                x = self.feature_extractor(x)
                x = self.scale_to_bounds(x) 

            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 
        
    class LargeFeatureExtractor(torch.nn.Sequential):
        def __init__(self, data_dim):
            super(LargeFeatureExtractor, self).__init__()
            self.add_module('linear1', torch.nn.Linear(1, 1000))
            self.add_module('relu1', torch.nn.ReLU())
            self.add_module('linear2', torch.nn.Linear(1000, 500))
            self.add_module('relu2', torch.nn.ReLU())
            self.add_module('linear3', torch.nn.Linear(500, 50))
            self.add_module('relu3', torch.nn.ReLU())
            self.add_module('linear4', torch.nn.Linear(50, data_dim))


            
    class TrainingMetrics:
        """
        Container class for storing training metrics during GP optimization.
        
        Attributes:
            lengthscale (torch.Tensor): Lengthscale values during training
            noise (torch.Tensor): Noise values during training
            loss (torch.Tensor): Loss values during training
        """
        pass