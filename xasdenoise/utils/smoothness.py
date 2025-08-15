"""
Functions used to estimate the smoothness of a 1D signal.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.fftpack import fft, ifft, fftshift


def estimate_smoothness(x, y, polyfit_len=51, smooth_len=20):
    """
    Estimate the smoothness of a signal using sliding window polynomial fitting.

    Args:
        x (np.ndarray): Array of x values (e.g., energy).
        y (np.ndarray): Array of y values (e.g., spectrum).
        polyfit_len (int, optional): Length of the sliding window for polynomial fitting. Defaults to 51.
        smooth_len (int, optional): Length of the sliding window for smoothing. Defaults to 20.

    Returns:
        tuple: Hessian (second derivative) and smoothness estimate.
    """
    # start_time = time.time()
    if polyfit_len % 2 == 0:
        polyfit_len += 1

    y = np.mean(y, axis=1) if y.ndim > 1 else y
    
    y_windows = sliding_window_view(y, polyfit_len, axis=0).T
    x_windows = np.arange(polyfit_len)
    
    poly_deg = np.max([np.min([2, polyfit_len-1]), 1])
    fit_coeff = np.polyfit(x_windows, y_windows, deg=poly_deg)
    hessian = fit_coeff[0, :]
    
    win_len = np.max([np.min([5, polyfit_len]), 2])
    hessian = median_filter(hessian, win_len)

    pad = x.shape[0] - hessian.shape[0]
    hessian = np.pad(hessian, (pad // 2, pad // 2), mode='edge')
    smoothness_estimate = maximum_filter1d(np.abs(hessian), smooth_len, mode='nearest')
    # print(f"Time taken for fast smoothness estimation: {(time.time() - start_time)*1e3:.2f} ms")
    return hessian, smoothness_estimate

def high_pass_filter(spectrum, sigma=0.05):
    """
    Apply a high-pass filter to estimate the baseline of a signal.

    Args:
        spectrum (np.ndarray): Input signal.
        sigma (float, optional): Standard deviation for the high-pass filter. Defaults to 0.05.

    Returns:
        np.ndarray: High-pass filtered signal.
    """
    padding = spectrum.shape[0] // 2

    if padding > 0:
        spectrum = (
            np.pad(spectrum, (padding, padding), mode='edge')
            if spectrum.ndim == 1 else
            np.pad(spectrum, ((padding, padding), (0, 0)), mode='edge')
        )

    Npix = spectrum.shape[0]
    spectrum = fft(spectrum, axis=0)

    x = np.arange(-Npix // 2, Npix // 2) / Npix + 1e-10
    sigma = 256 / (Npix - 2 * padding) * sigma

    spectral_filter = (
        2j * np.pi * (fftshift(np.arange(Npix) / Npix) - 0.5)
        if sigma == 0 else
        fftshift(np.exp(1. / (-(x ** 2) / (sigma ** 2))))
    )

    spectrum *= spectral_filter[:, None] if spectrum.ndim > 1 else spectral_filter
    spectrum = np.real(ifft(spectrum, axis=0))

    return spectrum[padding:-padding] if padding > 0 else spectrum
