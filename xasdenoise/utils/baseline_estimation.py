"""
Functions used to estimate the baseline of the spectrum. 

In most cases these functions should return a step-like function to obtain a zero mean signal.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from xasdenoise.utils import normalization
from scipy.optimize import curve_fit, least_squares


def estimate_polynomial_baseline(x, y, glitch_mask=None):
    """
    Estimate the smooth baseline of the spectrum using polynomial fitting.

    Args:
        x (np.ndarray): The energy array.
        y (np.ndarray): The spectrum array.
        glitch_mask (np.ndarray, optional): Boolean array indicating artifact data points. Defaults to None.

    Returns:
        np.ndarray: The smooth baseline of the spectrum.
    """
    from pybaselines import Baseline
    baseline_fitter = Baseline(x_data=x)
    baseline, params = baseline_fitter.poly(y, weights=glitch_mask, poly_order=3)
    return baseline


def estimate_smooth_baseline(x, y, glitch_mask=None, lam=1e3):
    """
    Estimate the smooth baseline of the spectrum using iterative asymmetric least squares.

    Args:
        x (np.ndarray): The energy array.
        y (np.ndarray): The spectrum array.
        glitch_mask (np.ndarray, optional): Boolean array indicating artifact data points. Defaults to None.
        lam (float, optional): Smoothing parameter. Defaults to 1e3.

    Returns:
        np.ndarray: The smooth baseline of the spectrum.
    """
    from pybaselines import Baseline
    baseline_fitter = Baseline(x_data=x)
    baseline, params = baseline_fitter.iasls(y, weights=glitch_mask, lam=lam, p=0.01)
    return baseline


def find_rising_edge_midpoint(spectrum):
    """
    Find rising edge mid-point.
    
    Args:
        spectrum (object): Spectrum object.
        
    Returns:
        float: Estimated edge energy.
    """
    # use the mid-point of the rising absorptin edge. Easiest to use the precomputed background
    x = spectrum.energy
    background = spectrum.background
    if background is None:
        background = spectrum.compute_background()
    midpoint = x[np.argmin(np.abs(background - (background.max() - background.min())/2))]
    return midpoint


def fit_edge_step_not_normalized(x, y, edge, pre_edge_idx, post_edge_idx, 
                           fitting_funcs=['V','V'], downsample=0, beta=10,
                           robust_metric='mad'):
    """
    Estimate the step background function of an XAS spectrum using pre-edge and post-edge fits.

    Args:
        x (np.ndarray): The energy array.
        y (np.ndarray): The spectrum array, 2D (energy, time).
        edge (float): The edge energy.
        pre_edge_idx (list): The indices of the pre-edge region.
        post_edge_idx (list): The indices of the post-edge region.
        fitting_funcs (list, optional): A list of fitting functions for the pre-edge and post-edge regions. Defaults to ['V','V'].
        downsample (int, optional): The downsampling factor for fitting. Defaults to 0.
        beta (float, optional): Controls the sharpness of the transition between pre-edge and post-edge regions. Defaults to 10.
        robust_metric (str, optional): Robust error metric. Options: 'mad', 'percentile', 'mse'. Defaults to 'mad'.

    Returns:
        np.ndarray: Estimated background for the spectrum.
    """
    def tanh_fun(x, beta):
        """Hyperbolic tangent transition function"""
        return 0.5 * (1 + np.tanh(beta * (2 * x - 1)))
    
    def robust_error_metric(residuals, metric='mad'):
        """
        Compute robust error metrics that are less sensitive to outliers.
        """
        if metric == 'mse':
            return np.mean(residuals**2)
       
        elif metric == 'mad':
            # Median Absolute Deviation - very robust to outliers
            return np.median(np.abs(residuals - np.median(residuals)))
        
        elif metric == 'percentile':
            # Use 75th percentile of squared errors to ignore worst outliers
            return np.percentile(residuals**2, 75)
        
        else:
            raise ValueError(f"Unknown robust metric: {metric}")
    
    # reduce noise
    # y = gaussian_filter1d(y, 10)
    win_len_eV = 10
    win = np.argmin(np.cumsum(abs(np.diff(x))) <= win_len_eV)
    win = win // 2 * 2 + 1
    y = uniform_filter1d(y, win, axis=0)
    
    # Initialize normalization object
    normalise = normalization.NormFit()
    normalise.downsample = downsample
    normalise.e0 = edge
    
    # Use a rough guess for the pre/post edge fitting parameters
    pre_edge_fit_params = ([x[pre_edge_idx[0]] - edge, 
                            x[pre_edge_idx[-1]] - edge, 
                            fitting_funcs[0]])
    post_edge_fit_params = ([x[post_edge_idx[0]] - edge, 
                            x[post_edge_idx[-1]] - edge, 
                            fitting_funcs[-1]])
    
    # Fit pre-edge and post-edge baselines
    pre_edge_baseline = normalise._fit_edge(x, y, pre_edge_fit_params, 'pre-edge')[0]
    post_edge_baseline = normalise._fit_edge(x, y, post_edge_fit_params, 'post-edge')[0]

    # Determine blending region
    width = 50
    min_error = np.inf
    edge_shift = 0
    
    for width_shift in range(-50,50):
        for edge_shift in range(-20,20):
            idx_pre = x < (edge + edge_shift) - (width + width_shift)
            idx_post = x > (edge + edge_shift) + (width + width_shift)
            idx_blend = (x >= (edge + edge_shift) - (width + width_shift)) & (x <= (edge + edge_shift) + (width + width_shift))

            # Assign pre-edge and post-edge values
            background = np.zeros_like(x)
            background[idx_pre] = pre_edge_baseline[idx_pre]
            background[idx_post] = post_edge_baseline[idx_post]

            # Blend using the selected transition function
            coefs = tanh_fun((x[idx_blend] - (edge + edge_shift) + (width + width_shift)) / (2 * (width + width_shift) + 1e-4), beta)
            blended_baseline = (1 - coefs) * pre_edge_baseline[idx_blend] + coefs * post_edge_baseline[idx_blend]
            background[idx_blend] = blended_baseline

            # Calculate robust error metric
            residuals = background - y
            
            error = robust_error_metric(residuals, robust_metric)
            
            if error < min_error:
                min_error = error
                best_width = width + width_shift
                best_edge_shift = edge + edge_shift
                best_background = background                

    background = best_background
    return background

def fit_edge_step(x, y, edge_guess=None,  
                  robust_fit=True, max_iter=100):
    """
    Fit a smooth step-like baseline function directly to a normalized XAS spectrum.
    
    This is a simpler alternative to the complex pre/post-edge fitting approach,
    suitable for normalized spectra where we expect a step from ~0 to ~1.
    
    Args:
        x (np.ndarray): Energy array
        y (np.ndarray): Normalized spectrum array
        edge_guess (float, optional): Initial guess for edge position. If None, uses midpoint.
        robust_fit (bool): Use robust fitting to handle outliers
        max_iter (int): Maximum iterations for fitting
        
    Returns:
        tuple: (baseline, fit_params) where baseline is the fitted step function
               and fit_params contains the fitted parameters
    """
    
    def tanh_step(x, x0, width):
        """Hyperbolic tangent transition function"""
        return 1 * 0.5 * (1 + np.tanh((x - x0) / width))
    
    def huber_loss(residuals, delta=1.0):
        """Huber loss function"""
        abs_res = np.abs(residuals)
        return np.where(abs_res <= delta, 
                        0.5 * residuals**2,
                        delta * (abs_res - 0.5 * delta))
            
    def loss_function(params):
        """Loss function for robust fitting"""
        res = tanh_step(x, *params) - y
        return huber_loss(res, delta=np.std(res))
    
    # Initial parameter guess
    if edge_guess is None:
        # Find approximate edge position from derivative
        dx = np.diff(x)
        dy = np.diff(y)
        edge_guess = x[np.argmax(dy/dx)]
    
    width_guess = (x.max() - x.min()) * 0.05  # 5% of energy range
    
    initial_params = [edge_guess, width_guess]#, y_min, y_max]
    
    # Parameter bounds
    bounds = (
        [x.min(), 0], # Lower bounds
        [x.max(), np.inf]# Upper bounds
    )
    
    if robust_fit:        
        try:
            result = least_squares(loss_function, initial_params, 
                                 bounds=bounds, max_nfev=max_iter*len(initial_params))
            fit_params = result.x
            success = result.success
        except:
            success = False
    else:
        # Standard least squares fitting
        try:
            fit_params, _ = curve_fit(tanh_step, x, y, p0=initial_params, 
                                    bounds=bounds, maxfev=max_iter)
            success = True
        except:
            success = False
    
    if not success:
        # Fallback to initial guess
        print("Warning: Fitting failed, using initial parameter guess")
        fit_params = initial_params
    
    # Generate baseline
    baseline = tanh_step(x, *fit_params)
    return baseline