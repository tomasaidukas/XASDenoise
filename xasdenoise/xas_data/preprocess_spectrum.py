"""
Functions used to process the Spectrum object.
"""

import numpy as np
from xasdenoise.utils import artefacts
from xasdenoise.utils import baseline_estimation
from xasdenoise.utils import normalization
from xasdenoise.utils import baseline_estimation
from matplotlib import pyplot as plt


def time_average_spectrum(data, size=None):
    """
    Compute the time-averaged spectrum and update the spectrum data.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        size (int, optional): The size of the time-averaged spectrum. If None, the full spectrum is used for averaging.
    
    Returns:
        None
    """
    if size is None:
        data.spectrum = data.time_averaged_spectrum[:, None]
    elif size is not None:
        data.bin_time_instances(size)
    else:
        raise ValueError("Invalid size parameter. Must be None or an integer.")    
    

def compute_mu(data):
    """
    Compute the absorption spectrum for the provided Spectrum object.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.

    Returns:
        None
    """
    data.compute_mu()

def remove_bad_time_instances(data):
    """
    Identify and remove time instances with bad data (e.g., high noise).

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.

    Returns:
        None
    """
    y = data.spectrum
    if y.shape[1] <= 1:
        print('remove_bad_time_instances() requires a time series spectrum with more than 1 time instance. Skipping.')
        return

    I = np.sum(abs(y) ** 2, axis=0)
    thresh = np.median(I) * 1.2
    indices = np.where(I > thresh)[0]

    if indices.size > 0:
        print(f"Deleting time indices {indices[0]} to {indices[-1]} out of {y.shape[1]} for {data.metadata['compound']}")
        data.delete_time_indices(indices)

def remove_nans(data):
    """
    Remove NaN values from the spectrum data.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.

    Returns:
        None
    """
    indices = np.isnan(np.sum(data.spectrum, axis=1))
    if np.sum(indices) > 0:
        data.delete_energy_indices(indices)

def remove_indices(data, indices):
    """
    Remove specific indices from the spectrum data.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        indices (np.ndarray): Indices to remove from the spectrum data.

    Returns:
        None
    """
    data.delete_energy_indices(indices)

def show_nans(data):
    """
    Check if NaNs are present in the spectrum data.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.

    Returns:
        bool: True if NaNs are present, False otherwise.
    """
    return np.isnan(np.sum(data.spectrum))

def find_glitches(data, threshold=95, glitch_matching=True, split_data=False, glitch_refinement_fit=False, 
                  glitch_fit_window=None, glitch_width_scaling=1, group_glitches=False, plot=False, 
                  glitch_fit_max_error=np.inf, glitch_fit_max_width=1000,
                  glitch_fit_models=None):
    """
    Identify glitches in the spectrum using various detection techniques.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        threshold (int, optional): Percentile threshold for glitch detection. Defaults to 95.
        glitch_matching (bool, optional): Use glitch matching algorithm. Defaults to True.
        split_data (bool, optional): Split spectrum into regions for glitch detection. Defaults to False.
        glitch_refinement_fit (bool, optional): Refine glitches using fitting models. Defaults to False.
        glitch_fit_window (int, optional): Window size for fitting glitches. Defaults to None.
        glitch_width_scaling (int, optional): Scale factor for glitch width. Defaults to 1.
        group_glitches (bool, optional): Group similar glitches. Defaults to False.
        plot (bool, optional): Generate plots for visualization. Defaults to False.
        glitch_fit_max_error (float, optional): Maximum error for glitch fitting. Defaults to np.inf.
        glitch_fit_max_width (int, optional): Maximum width for fitting glitches. Defaults to 1000.
        glitch_fit_models (list, optional): List of fitting models. Defaults to None.

    Returns:
        np.ndarray: Boolean mask indicating glitch positions.
    """
    if glitch_fit_models is None:
        glitch_fit_models = ['asymmetric_gaussians', 'asymmetric_lorentzians', 'asymmetric_skewed_gaussians', 'asymmetric_skewed_lorentzians',
                             'asymmetric_gaussians_flip', 'asymmetric_skewed_gaussians_flip', 'asymmetric_skewed_lorentzians_flip',
                             'gaussian', 'skewed_gaussian', 'shifted_gaussian_tails', 'lorentzian', 'skewed_lorentzian','voigt']

    # initialize the data
    x = data.energy
    y = data.time_averaged_spectrum
    I0 = data.time_averaged_I0
    I1 = data.time_averaged_I1

    glitch_mask_best = np.zeros(x.shape[0], dtype=bool)
    
    # remove baseline form the absorption spectrum for easier processing
    step_baseline = baseline_estimation.fit_edge_step_not_normalized(x, y, data.edge, data.pre_edge_region_indices, data.post_edge_region_indices)
    y_ref = y-step_baseline
    
    # split data into XANES and EXAFS regions and find glitches for each region
    if split_data:
        # crop = data.pre_edge_region_indices
        crop_xanes = np.where(x < data.first_minima_before_edge)[0]
        
        # crop = data.post_edge_region_indices
        crop_exafs = np.where(x > (data.first_maxima_after_edge+20))[0]
        crop_arr  [crop_xanes, crop_exafs]
    else:
        # no crop, take all data
        crop_arr = [np.ones_like(x, dtype=bool)]
  
    # find glitches for each region    
    for crop in crop_arr:
        mask = np.ones_like(x, dtype=bool)
        mask[crop] = False

        # detect glitches within the data using the I0 signal, the glitches are returned as boolean mask 
        # this method uses the scipy peak finder to estimate the glitch positions and widths
        if glitch_matching:
            glitch_positions, glitch_widths, glitch_mask = artefacts.detect_matching_glitches(x, y_ref, I0, mask, threshold, glitch_width_scaling, group_glitches=group_glitches, tol=20, plot=plot>=2)
        else:
            glitch_positions, glitch_widths, glitch_mask = artefacts.detect_glitches(x, I0, mask, threshold, glitch_width_scaling, group_glitches, plot=plot>=2)
        glitch_mask_best[glitch_mask] = True
        
        # refine the glitch mask by fitting glitch models to the absorption spectrum (not I0)   
        # this will replace the mask generated by scipy peak finder 
        if glitch_refinement_fit:   
            glitch_mask_fit = artefacts.estimate_glitch_mask_fit(x, y_ref, glitch_positions, glitch_widths, glitch_width_scaling,
                                                                glitch_fit_window, glitch_fit_max_width, glitch_fit_max_error, glitch_fit_models)

            # combine both glitch masks into one
            # glitch_mask_best[glitch_mask_fit] = True
            # overwrite (but keep the edges, fitting does not deal well at the edge regions)
            glitch_mask_best = glitch_mask_fit
            glitch_mask_best[:100] = glitch_mask[:100] | glitch_mask_fit[:100]
            glitch_mask_best[-100:] = glitch_mask[-100:] | glitch_mask_fit[-100:]
    
    # exclude region close to the edge
    exclude = (x > data.first_minima_before_edge) * (x < (data.first_maxima_after_edge+20))
    glitch_mask_best[exclude] = False
    
    # deglitching progress
    n_idx = (glitch_mask_best==True).sum()
    ratio = round(n_idx / x.shape[0] * 100)
    print("Number of artefact indiced found: {} ({}% of data)".format(n_idx,ratio))
    
    if plot and n_idx > 0:
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('Detected glitch locations')
        plt.plot(x[glitch_mask_best], y[glitch_mask_best], 'r.', alpha=0.3, label='glitches')    
        plt.plot(x, y, label='spectrum')
        plt.xlabel('Energy (eV)')
        plt.legend()
        plt.subplot(2,1,2)
        x_removed, y_removed= artefacts.remove_glitches(x, y, glitch_mask_best, plot=False, mode='interp', crop=True)
        plt.plot(x_removed, y_removed, label='deglitched spectrum')
        plt.xlabel('Energy (eV)')
        plt.legend()
        plt.show()
    return glitch_mask_best

def remove_glitches(data, glitch_mask0, glitch_fill='interp', exclude_edge=False, crop_edges=False):
    """
    Remove glitches from the spectrum data based on a mask.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        glitch_mask (np.ndarray): Boolean mask indicating glitch positions.
        glitch_fill (str, optional): Method for filling glitches ('interp', 'delete', etc.). Defaults to 'interp'.
        exclude_edge (bool, optional): Exclude the edge region from glitch removal. Defaults to False.
        crop_edges (bool, optional): If there are glitches at the ends of the spectrum, crop them. Defaults to False.

    Returns:
        None
    """
    x = data.energy
    glitch_mask = glitch_mask0.copy()
    if exclude_edge:
        exclude = (x > data.first_minima_before_edge) & (x < data.first_maxima_after_edge + 20)
        glitch_mask[exclude] = False

    # if glitches are at the start/end of the spectrum, interpolation is not well defined, better to just crop the data
    if crop_edges:
        glitch_mask_tmp = np.zeros_like(glitch_mask, dtype=bool)
        
        # check if there are glitches at the start of the spectrum for around 10eV
        idx = np.where((x - x[0]) < 10)[0]
        if np.sum(glitch_mask[idx]) > 0:
            glitch_mask_tmp[idx] = glitch_mask[idx]
            glitch_mask[idx] = False
            
        # check if there are glitches at the end of the spectrum for around 10eV
        idx = np.where((x[-1] - x) < 10)[0]
        if np.sum(glitch_mask[idx]) > 0:
            glitch_mask_tmp[idx] = glitch_mask[idx]
            glitch_mask[idx] = False
            
        # crop data by deleting energy indices
        if np.sum(glitch_mask_tmp) > 0:
            data_mask = ~glitch_mask_tmp
            data.delete_energy_indices(glitch_mask_tmp)            
            glitch_mask = data.glitch_mask
                
    # perform glitch removal/interpolation        
    if glitch_mask is not None and np.sum(glitch_mask) > 0:     
        print('Removing glitches')   
        x0 = data.energy
        for attr in data.y_arrays:
            y = getattr(data, attr)
            if y is not None:
                # for glitch mask only peform deletion, otherwise do not touch
                if attr == 'glitch_mask':
                    if glitch_fill == 'delete':
                        x, y = artefacts.remove_glitches(x0, y, glitch_mask, plot=False, mode='delete')
                        setattr(data, attr, y)
                    else:
                        # Skip glitch removal for other modes when attr is 'glitch_mask'
                        pass
                else:
                    x, y = artefacts.remove_glitches(x0, y, glitch_mask, plot=False, mode=glitch_fill)
                    setattr(data, attr, y)
        setattr(data, 'energy', x)
        
def normalize_spectrum(data, fitting_funcs=['1','V'], fit_individual=True, downsample=0):
    """
    Normalize the spectrum by fitting a model to pre-edge and post-edge regions.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        fitting_funcs (list, optional): A list of fitting functions for the pre-edge and post-edge regions.
        fit_individual (bool, optional): If True, fits each time instance individually.
        downsample (int, optional): The downsampling factor for normalization.

    Returns:
        None
    """
    # Initialize data
    x_fit = data.energy
    y_fit = data.time_averaged_spectrum
    y = data.spectrum
    edge = data.edge
    normalise = normalization.NormFit()
    normalise.downsample = downsample
    normalise.e0 = edge
    
    # Use a rough guess for the pre/post edge fitting parameters
    pre_edge_fit_params = ([x_fit[data.pre_edge_region_indices[0]] - edge, 
                            x_fit[data.pre_edge_region_indices[-1]] - edge, 
                            data.metadata.get('fitting_func_pre_edge', fitting_funcs[0])])
    post_edge_fit_params = ([x_fit[data.post_edge_region_indices[0]] - edge, 
                            x_fit[data.post_edge_region_indices[-1]] - edge, 
                            data.metadata.get('fitting_func_post_edge', fitting_funcs[-1])])
        
    # Normalize spectrum using the best pre/post fits
    if fit_individual:
        for time in range(y.shape[1]):
            data.spectrum[:, time] = normalise.norm(
                x_fit, y[:, time], y[:, time], pre_edge_fit_params, post_edge_fit_params, edge
            )
    else:
        data.spectrum = normalise.norm(x_fit, y_fit, y, pre_edge_fit_params, post_edge_fit_params, edge)
        # data.spectrum = y / np.mean(y, axis=0) * np.mean(data.spectrum, axis=0)

def estimate_background(data):
    """
    Estimate the background of an XAS spectrum using pre-edge and post-edge fits.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        
    Returns:
        np.ndarray: Estimated background for the spectrum.
    """
    x_fit = data.energy + 1e-3
    y_fit = data.time_averaged_spectrum
    edge = data.edge # instead of the tabulated edge position
    pre_edge_idx = data.pre_edge_region_indices
    post_edge_idx = data.post_edge_region_indices
    
    from xasdenoise.utils import baseline_estimation
    background = baseline_estimation.fit_edge_step_not_normalized(x_fit, y_fit, edge, pre_edge_idx, post_edge_idx)
    data.background = background
    
    return background

def downsample_spectrum(data, size=None, factor=None):
    """
    Downsample the spectrum by averaging neighbouring points.
    
    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        size (int, optional): The size of the downsampled spectrum.
        factor (int, optional): The downsampling factor for the spectrum.
    """
    data.downsample_spectrum(size, factor)

def crop_spectrum(data, pre_edge=50, post_edge=50):
    """
    Crop the spectrum to within a specified pre-edge and post-edge region.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        pre_edge (int): The energy range before the edge to crop. Defaults to 50.
        post_edge (int): The energy range after the edge to crop. Defaults to 50.

    Returns:
        None
    """
    x = data.energy
    edge = data.edge

    pre_edge_idx = np.where(x < edge - pre_edge)[0]
    pre_edge_idx = pre_edge_idx[-1] if len(pre_edge_idx) > 0 else 0

    post_edge_idx = np.where(x > edge + post_edge)[0]
    post_edge_idx = post_edge_idx[0] if len(post_edge_idx) > 0 else len(x)

    crop = slice(pre_edge_idx, post_edge_idx)
    data.crop_energy_indices(crop)


def crop_spectrum_energy(data, energy_idx):
    """
    Crop the spectrum based on specified energy indices.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        energy_idx (np.ndarray): The energy indices to crop the spectrum to.

    Returns:
        None
    """
    data.crop_energy_indices(energy_idx)


def crop_spectrum_time(data, time_idx):
    """
    Crop the spectrum based on specified time indices.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.
        time_idx (np.ndarray): The time indices to crop the spectrum to.

    Returns:
        None
    """
    data.crop_time_indices(time_idx)

def crop_other_edges(data):
    """
    Crop the spectrum to remove additional edges (e.g., L-edges) other than the main edge.

    Args:
        data (Spectrum): A Spectrum object containing the spectrum data.

    Returns:
        None
    """
    x = data.energy
    edge = data.edge
    all_edges = data.metadata['all_edges']

    edges_in_spectrum = [
        value for value in all_edges if data.energy.min() < value < data.energy.max()
    ]
    edges_in_spectrum = np.array(edges_in_spectrum)

    if len(edges_in_spectrum) > 1:
        idx = np.argmin(np.abs(edges_in_spectrum - edge))
        edges_in_spectrum = np.delete(edges_in_spectrum, idx)

        min_idx = np.where(edges_in_spectrum < edge)[0]
        max_idx = np.where(edges_in_spectrum > edge)[0]

        min_crop = 0 if len(min_idx) == 0 else np.argmin(x < (np.max(edges_in_spectrum[min_idx]) + 50))
        max_crop = len(x) if len(max_idx) == 0 else np.argmax(x > (np.min(edges_in_spectrum[max_idx]) - 50))

        crop = slice(min_crop, max_crop)
        data.crop_energy_indices(crop)
