"""
Denoising utilities for XAS data processing.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def downsample_data(*args, **kwargs):
    """
    Downsample the data for GP denoising.

    Args:
        *args: Arrays to be downsampled (e.g., x, y, noise).
        **kwargs: Arbitrary keyword arguments.
            method (str): Downsampling method ('smoothness' or 'uniform').
            num_samples (int): Number of samples to draw.
            smoothness (array, optional): Array of smoothness values (used if method='smoothness').

    Returns:
        tuple: Downsampled arrays corresponding to the input arrays in *args.
    """

    method = kwargs.get('method')
    num_samples = kwargs.get('num_samples')
    smoothness = kwargs.get('smoothness')

    # Validate inputs
    if num_samples is None:
        # If not provided, default to the length of the first array
        num_samples = len(args[0])
    
    # downsample data 
    if (method == 'smoothness' or method == 'kspace') and num_samples <= len(args[0]):
        # Assuming smoothness is a 1D array for now
        smoothness = np.squeeze(smoothness)
        assert smoothness.ndim == 1, "Smoothness must be 1D"
        
        downsampled = proportional_downsampling(*args, smoothness=smoothness, num_samples=num_samples)
    elif method == 'uniform' and num_samples <= len(args[0]):
        downsampled = uniform_downsampling(*args, num_samples=num_samples)
    else:
        # no downsampling
        downsampled = args
        
    return downsampled

def proportional_downsampling(*args, **kwargs):
    """
    Perform proportional downsampling based on the smoothness array.

    Args:
        *args: Arrays to be downsampled.
        **kwargs: Arbitrary keyword arguments.
            smoothness (np.ndarray): Array of smoothness values.
            num_samples (int): Number of samples to draw.
            base (float, optional): Base probability for sampling. Defaults to 0.005.

    Returns:
        tuple: Downsampled arrays corresponding to the input arrays in *args.
    """

    smoothness = kwargs.get('smoothness')
    num_samples = kwargs.get('num_samples')
    base = kwargs.get('base', 0.005)
        
    sampling_dist = get_downsampling_distribution(smoothness, num_samples, base)
    sampled_indices = sampling_dist()
    sampled_indices = np.sort(sampled_indices)

    downsampled_args = tuple(arg[sampled_indices] if arg is not None else None for arg in args)
    return downsampled_args

def uniform_downsampling(*args, **kwargs):
    """
    Perform uniform downsampling.

    Args:
        *args: Arrays to be downsampled.
        **kwargs: Arbitrary keyword arguments.
            num_samples (int): Number of samples to draw.

    Returns:
        tuple: Downsampled arrays corresponding to the input arrays in *args.
    """

    num_samples = kwargs.get('num_samples')
    
    if num_samples is None:
        raise ValueError("num_samples must be provided in kwargs")

    sampled_indices = np.random.choice(args[0].shape[0], num_samples, replace=False)
    sampled_indices = np.sort(sampled_indices)
    
    downsampled_args = tuple(arg[sampled_indices] if arg is not None else None for arg in args)
    return downsampled_args

def get_downsampling_factors(smoothness):
    """
    Calculate downsampling factors based on smoothness values.

    Args:
        smoothness (np.ndarray): Array of smoothness values.

    Returns:
        np.ndarray: Normalized downsampling factors.
    """

    smo = smoothness**0.5
    return smo / np.max(smo) 

def get_downsampling_distribution(smoothness, num_samples=1000, base=0.005):
    """
    Use the smoothness array to generate a probabilistic distribution for sampling indices.

    Args:
        smoothness (np.ndarray): Array of smoothness values.
        num_samples (int, optional): Number of samples to draw. Defaults to 1000.
        base (float, optional): Base probability for sampling. Defaults to 0.005.

    Returns:
        Callable: A function that generates sampling indices based on the distribution.
    """

    factors = get_downsampling_factors(smoothness)
    total_size = len(smoothness)

    # Normalize the factors to make their sum equal to `num_samples`
    probabilities = np.maximum(base, factors)
    probabilities = probabilities / np.sum(probabilities) * num_samples

    # Clamp probabilities to [0, 1]
    probabilities = np.clip(probabilities, 0, 1)

    def sampler():
        # Generate samples based on probabilities
        selected_indices = np.where(np.random.binomial(n=1, p=probabilities) == 1)[0]

        # Adjust the number of samples to match `num_samples` as closely as possible
        if len(selected_indices) > num_samples:
            # Randomly downsample to reduce the count
            selected_indices = np.random.choice(selected_indices, size=num_samples, replace=False)
        elif len(selected_indices) < num_samples:
            # Add missing samples by prioritizing higher probabilities
            remaining_indices = np.setdiff1d(np.arange(total_size), selected_indices)
            additional_indices = remaining_indices[
                np.argsort(probabilities[remaining_indices])[-(num_samples - len(selected_indices)):]
            ]
            selected_indices = np.concatenate([selected_indices, additional_indices])

        return selected_indices

    return sampler

def energy_to_wavenumber(energy, edge, mode=None):
    """
    Transform energy values to wavenumber (k-space) values.
    
    Args:
        energy (np.ndarray): Array of energy values.
        edge (float): Edge energy.
        mode (str, optional): Special handling modes.
        
    Returns:
        np.ndarray: K-space values.
    """
    # Calculate energy difference relative to the edge
    dE = energy - edge
    
    # Apply standard k-space transformation
    k = np.sign(dE) * (np.sqrt(0.2625 * np.abs(dE)))
    
    # Apply special handling for different modes if specified
    if mode == 'linear_pre_edge':
        # Apply linear transformation to the pre-edge region
        idx_pre = np.where(energy < edge)[0]
        k[idx_pre] = np.linspace(k[idx_pre].min(), k[idx_pre].max(), len(idx_pre))
    return k

def kspace_downsampling_factors(energy, edge):
    """
    Perform data warping using the k**2 scaling factor used in XAS data analysis.
    
    Args:
        energy (np.ndarray): Array of energy values.
        edge (float): Edge energy.
        
    Returns:
        np.ndarray: K-space downsampling factors.
    """
    k = energy_to_wavenumber(energy, edge)
    return 1/(1+abs(k))**0.5

def inducing_indices(smoothness, num_samples, base):
    """
    Get inducing indices for sparse data using a smoothness-based distribution.

    Args:
        smoothness (np.ndarray): Array of smoothness values.
        num_samples (int): Number of samples to draw.
        base (float): Base probability for sampling.

    Returns:
        np.ndarray: Selected indices for sparse GP inducing points.
    """

    sampling_dist = get_downsampling_distribution(smoothness, num_samples, base)
    sampled_indices = sampling_dist()

    return sampled_indices

def estimate_noise(x, y):
    """
    Estimate the standard deviation of residuals from a linear regression fit.

    Args:
        x (np.ndarray): Array of x values (e.g., energy).
        y (np.ndarray): Array of y values (e.g., spectrum).

    Returns:
        np.ndarray: Estimated standard deviations of the residuals.
    """

    # fit a line on the noise region
    coef = np.polyfit(x, y, deg=1)
    model = coef[0]*x[:,None] + coef[1]
    
    # deviation between the data and the linear model
    if y.ndim > 1:
        std = np.std(y - model,axis=0)
    else:
        std = np.std(y - model)
    return std

def estimate_noise_std_sliding_window(x, y, window=21):
    """
    Estimate noise standard deviation using a sliding window approach.

    Args:
        x (np.ndarray): Array of x values (e.g., energy).
        y (np.ndarray): Array of y values (e.g., spectrum).
        window (int, optional): Size of the sliding window. Must be odd. Defaults to 21.

    Returns:
        np.ndarray: Estimated noise variances across the spectrum.
    """

    ndim = y.ndim
    if ndim == 1: y = y[:,None]
        
    std_windows_arr = np.zeros_like(y)
    for t in range(y.shape[1]):
        if window % 2 == 0:
            window = window + 1
            
        # Get sliding window views of the data
        y_windows = sliding_window_view(y[:,t], window, axis=0).T
        x_windows = np.arange(window)
        
        std_windows = estimate_noise(x_windows, y_windows)
            
        pad = y.shape[0] - std_windows.shape[0]
        std_windows = np.pad(std_windows, (pad // 2, pad // 2), constant_values=(std_windows[0], std_windows[-1]))
        
        std_windows_arr[:,t] = std_windows
        
    if ndim == 1: std_windows_arr = std_windows_arr[:,0]       
    return std_windows_arr

def blend_overlapping_windows(y_stitched, weights, noise_stitched=None, error_stitched=None):
    """
    Blend overlapping signal windows.
    
    Args:
        y_stitched: The combined signal array with overlaps
        weights: Weight array tracking number of overlaps
        noise_stitched: Optional noise estimates array
        error_stitched: Optional error estimates array
        
    Returns:
        Tuple of blended arrays (y_blended, noise_blended, error_blended)
    """
    y_blended = y_stitched / np.maximum(weights, 1)
    
    # Process noise and error if provided
    noise_blended = None
    error_blended = None
    
    if noise_stitched is not None:
        noise_blended = noise_stitched / np.maximum(weights, 1)
        
    if error_stitched is not None:
        error_blended = error_stitched / np.maximum(weights, 1)
    
    return y_blended, noise_blended, error_blended

def ev2points(x, win_eV):
    """
    Convert energy values to points for window lengths.

    Args:
        x (np.ndarray): Energy values.
        win_eV (int): Window length in eV.
        
    Returns:
        int: Window length in points.
    """
    
    win = np.argmin(np.cumsum(abs(np.diff(x))) <= win_eV)
    win = win // 2 * 2 + 1
    win = np.maximum(win, 3) # at least 3 points long
    return win