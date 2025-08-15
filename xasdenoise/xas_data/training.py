"""
Helper functions for autoencoder training.
"""

from xasdenoise.xas_data import preprocess_spectrum, preprocess_spectrum_list
from sklearn.model_selection import train_test_split
import numpy as np


def create_training_tensors(spectrum_list_train, spectrum_list_target,
                           remove_background=True, warp=True, scale=True, 
                           simulate_training_data=False, num_times=None, noise2clean=True,
                           noise2noise=False, normalize_data=True, randomize_times=False):
    """
    Create training tensors for the encoder model.
    
    Args:
        spectrum_list_train (list): List of Spectrum objects containing the training data
        spectrum_list_target (list): List of Spectrum objects containing the target data
        compounds_for_test (list): List of compounds for testing set 
        align (bool): Whether to align the spectra
        remove_background (bool): Whether to remove the step function from data
        warp (bool): Whether to warp the data
        scale (bool): Whether to scale the data
        simulate_training_data (bool): Whether to simulate noisy training data from target data
        num_times (int): Number of time points to use for each compound
        
    Returns:
        tuple: (x0, y_train0, y_target0, background0, data_mask0, compounds0, elements0)
    """                            
    # Create copies of the spectrum lists
    spectrum_list_target = preprocess_spectrum_list.copy_spectra(spectrum_list_target)
    spectrum_list_train = preprocess_spectrum_list.copy_spectra(spectrum_list_train)
    
    # Remove background if requested
    if remove_background:
        _remove_background(spectrum_list_train, spectrum_list_target)
        
    # # Pad the spectra such that all of them are of the same length
    # preprocess_spectrum_list.pad_to_max_length(spectrum_list_target)
    # preprocess_spectrum_list.pad_to_max_length(spectrum_list_train)
    
    # Extract datasets
    if noise2noise and noise2clean:
        x0, y_train0, y_target0, glitch_mask0, data_mask0, compounds0, elements0, edges0 = _extract_spectrum_data_mixed(
            spectrum_list_train, spectrum_list_target, num_times, randomize_times)    
    elif noise2noise:
        x0, y_train0, y_target0, glitch_mask0, data_mask0, compounds0, elements0, edges0 = _extract_spectrum_data_noise2noise(
            spectrum_list_train, spectrum_list_target, num_times, randomize_times)
    elif noise2clean:
        x0, y_train0, y_target0, glitch_mask0, data_mask0, compounds0, elements0, edges0 = _extract_spectrum_data(
            spectrum_list_train, spectrum_list_target, num_times)
    
    # Adjust mean of target data to match training data
    if not noise2noise and normalize_data:
        y_train0 = _adjust_means(y_train0, y_target0)
        y_train0 = _adjust_poly_baseline(y_train0, y_target0, x0, edges0, degree=2)
    
    # Simulate noisy training data from target data if requested
    if simulate_training_data:
        x0, y_train0, y_target0, glitch_mask0, data_mask0, compounds0, elements0, edges0 = create_training_data_from_target_data(x0, y_train0, y_target0, glitch_mask0, data_mask0, compounds0, elements0, edges0, stdev=0.05)

    # Scale data if requested
    if scale:
        y_train0, y_target0 = _scale_data(x0, y_train0, y_target0, edges0)
        
    # Warp data if requested
    if warp:
        x0, y_train0, y_target0, data_mask0, glitch_mask0 = _warp_data(x0, y_train0, y_target0, edges0, data_mask0, glitch_mask0)

    # Pad arrays to the same size
    x0, y_train0, y_target0, glitch_mask0, data_mask0, edges0  = _pad_arrays_to_same_length(
        x0, y_train0, y_target0, glitch_mask0, data_mask0, edges0, warp)
    
    # Convert the lists to numpy arrays
    x0, y_train0, y_target0, glitch_mask0, data_mask0, edges0 = _convert_to_numpy_arrays(
        x0, y_train0, y_target0, glitch_mask0, data_mask0, edges0)
    
    
    try:
        # Create data mask to avoid training on corrupted data
        glitch_mask0 = glitch_mask0.astype(bool)
        data_mask0 = data_mask0.astype(bool)
        data_mask0 = data_mask0 & ~glitch_mask0
    except:
        pass
    
    return x0, y_train0, y_target0, data_mask0, compounds0, elements0, edges0


def create_training_data_from_target_data(x, y_train, y_target, glitch_mask, data_mask, compounds, elements, edges, stdev=0.05):
    """
    Create training data by taking ideal ground truth target data and adding random noise.
    
    Args:
        y_train (list): list of Spectrum arrays containing the training data
        y_target (list): list of Spectrum arrays containing the target data
        
    Returns:
        spectrum_list_train (list): list of Spectrum objects containing the training data
    """
    num_samples = len(y_train)

    for i in range(num_samples):
        # get a random stdev value and use it to create a random noise array
        stdev_rand = abs(np.random.normal(stdev, stdev))
        noise = np.random.normal(0, stdev_rand, size=y_target[i].shape)
        # add noise to the training data
        y_train.append(y_target[i] + noise)
        y_target.append(y_target[i])
        x.append(x[i])
        glitch_mask.append(glitch_mask[i])
        data_mask.append(data_mask[i])
        compounds.append(compounds[i])
        elements.append(elements[i])
        edges.append(edges[i])
                
    return x, y_train, y_target, glitch_mask, data_mask, compounds, elements, edges,

def _remove_background(spectrum_list_train, spectrum_list_target):
    """Remove background from spectra."""
    for train, target in zip(spectrum_list_train, spectrum_list_target): 
        train.spectrum = train.spectrum - train.background
        target.spectrum = target.spectrum - target.background

def _extract_spectrum_data_mixed(spectrum_list_train, spectrum_list_target, num_times=None, randomize_times=False):
    """
    Extract data for noise2noise and noise2clean training using different time points as independent noise realizations.
    """
    x0, y_train0, y_target0 = [], [], []
    data_mask0, compounds0, elements0, edges0, glitch_mask0 = [], [], [], [], []
    
    for s, s_target in zip(spectrum_list_train, spectrum_list_target):
        # Ensure we have at least 2 time points for noise2noise
        if s.spectrum.shape[1] < 2:
            continue
            
        time_range = range(min(num_times, s.spectrum.shape[1])) if num_times is not None else range(s.spectrum.shape[1])
        
        # Create pairs from different time points
        for t in time_range[:-1]:  # Skip last time point to ensure pairs
            x0.append(np.squeeze(s.energy))
            if np.random.rand() < 0.5:  # Randomly choose between noise2noise and noise2clean
                y_train0.append(np.squeeze(s.spectrum[:,t]))      # Time point t
                
                if randomize_times:
                    # Randomly select a target time point different from t
                    t_target = np.random.choice([i for i in time_range if i != t])
                    y_target0.append(np.squeeze(s.spectrum[:,t_target]))
                else:
                    y_target0.append(np.squeeze(s.spectrum[:,t+1]))   # Time point t+1
            else:
                y_train0.append(np.squeeze(s.spectrum[:,t]))
                y_target0.append(np.squeeze(s_target.spectrum[:,t]))  # Time point t from target
            glitch_mask0.append(s.glitch_mask)
            data_mask0.append(s.data_mask)
            compounds0.append(s.metadata['compound'])
            elements0.append(s.metadata['element'])
            edges0.append(s.edge)
    
    return x0, y_train0, y_target0, glitch_mask0, data_mask0, compounds0, elements0, edges0

def _extract_spectrum_data_noise2noise(spectrum_list_train, spectrum_list_target, num_times=None, randomize_times=False):
    """
    Extract data for noise2noise training using different time points as independent noise realizations.
    """
    x0, y_train0, y_target0 = [], [], []
    data_mask0, compounds0, elements0, edges0, glitch_mask0 = [], [], [], [], []
    
    for s, s_target in zip(spectrum_list_train, spectrum_list_target):
        # Ensure we have at least 2 time points for noise2noise
        if s.spectrum.shape[1] < 2:
            continue
            
        time_range = range(min(num_times, s.spectrum.shape[1])) if num_times is not None else range(s.spectrum.shape[1])
        
        # Create pairs from different time points
        for t in time_range[:-1]:  # Skip last time point to ensure pairs
            x0.append(np.squeeze(s.energy))
            y_train0.append(np.squeeze(s.spectrum[:,t]))      # Time point t
            
            if randomize_times:
                # Randomly select a target time point different from t
                t_target = np.random.choice([i for i in time_range if i != t])
                y_target0.append(np.squeeze(s.spectrum[:,t_target]))
            else:
                y_target0.append(np.squeeze(s.spectrum[:,t+1]))   # Time point t+1
                
            glitch_mask0.append(s.glitch_mask)
            data_mask0.append(s.data_mask)
            compounds0.append(s.metadata['compound'])
            elements0.append(s.metadata['element'])
            edges0.append(s.edge)
    
    return x0, y_train0, y_target0, glitch_mask0, data_mask0, compounds0, elements0, edges0

def _extract_spectrum_data(spectrum_list_train, spectrum_list_target, num_times=None):
    """Extract data from spectrum objects into arrays."""
    x0, y_train0, y_target0 = [], [], []
    data_mask0, compounds0, elements0, edges0, glitch_mask0 = [], [], [], [], []
    
    for s, s_target in zip(spectrum_list_train, spectrum_list_target):
        time_range = range(min(num_times, s.spectrum.shape[1])) if num_times is not None else range(s.spectrum.shape[1])
            
        for t in time_range:
            x0.append(np.squeeze(s.energy))
            y_train0.append(np.squeeze(s.spectrum[:,t]))
            y_target0.append(np.squeeze(s_target.spectrum[:,t]))
            glitch_mask0.append(s.glitch_mask)
            data_mask0.append(s.data_mask)
            compounds0.append(s.metadata['compound'])
            elements0.append(s.metadata['element'])
            edges0.append(s.edge)
    
    # Return arrays and lists separately
    return x0, y_train0, y_target0, glitch_mask0, data_mask0, compounds0, elements0, edges0


def _scale_data(x0, y_train0, y_target0, edges0):
    """Scale data using k-space scaling."""    
    scaled_target = []
    scaled_train = []
    
    for i in range(len(x0)):
        # Apply data scaling
        dE = x0[i] - edges0[i]
        exafs = x0[i] > edges0[i]
        xanes = x0[i] <= edges0[i]
        
        weights = np.sign(dE) * np.sqrt(0.2625 * abs(dE)) 
        weights = (1+abs(weights))**2
        weights[xanes] = 1
        
        scaled_train.append(y_train0[i] * weights)
        scaled_target.append(y_target0[i] * weights)
    return scaled_train, scaled_target


def _warp_data(x0, y_train0, y_target0, edges0, data_mask0, glitch_mask0):
    """Warp data using k-space warping."""
    from xasdenoise.warping.warping import DataWarper
    
    # some spectra are smaller in length than others, so they were padded by zeros
    # however, such padding will cause issues with warping. Hence, before warping, 
    # we need to remove the padded zeros and then add them back after warping.
    warped_x = []
    warped_y_train =[]
    warped_y_target = []
    warped_mask = []
    warped_glitch_mask = []
    
    for i in range(len(x0)):
        # Remove padded zeros from the spectra
        mask = data_mask0[i]
        mask_idx = np.where(mask)[0]       
        
        # print(f'Warping data for index {i+1}/{len(x0)}')
        warper = DataWarper(input_warping_method='kspace', warping_interpolation_method='same', 
                          output_warping_interpolation='linear', verbose=0)
        
        # Initialize once and reuse the same warper
        warper.initialize_warping(x0[i][mask_idx], edges0[i])
        
        # Warp target data
        _, warped_y_target_arr = warper.warp(y_target0[i][mask_idx])
        warped_y_target.append(warped_y_target_arr)
        
        # Keep the original x values
        warped_x.append(x0[i])  
        
        # Warp training data
        _, arr = warper.warp(y_train0[i][mask_idx])
        warped_y_train.append(arr)
        
        # Warp mask
        _, arr = warper.warp(data_mask0[i][mask_idx])
        warped_mask.append(arr)
        
        # Warp glitch mask
        _, arr = warper.warp(glitch_mask0[i][mask_idx])
        warped_glitch_mask.append(arr)
        
    return warped_x, warped_y_train, warped_y_target, warped_mask, warped_glitch_mask

def _adjust_means(y_train0, y_target0):
    """Adjust means of target data to match training data."""
    adjusted = []
    for i in range(len(y_train0)):
        adjusted.append(y_train0[i] - np.mean(y_train0[i]) + np.mean(y_target0[i]))
    return adjusted

def _adjust_poly_baseline(y_train0, y_target0, x0, edges0, degree=2):
    """Adjust target spectra by removing polynomial baseline and matching train baseline."""
    adjusted = []
    ctr = 0
    for yt, ytr in zip(y_target0, y_train0):
        
        idx_pre_edge = np.where(x0[ctr] < edges0[ctr])[0]
        idx_post_edge = np.where(x0[ctr] >= edges0[ctr])[0]
        
        adjusted_ytr = ytr.copy()
        
        for crop in [idx_pre_edge, idx_post_edge]:
            x = np.arange(y_train0[ctr][crop].shape[-1])
            # Fit polynomial baselines to both
            poly_target = np.polynomial.Polynomial.fit(x, yt[crop], degree)
            poly_train = np.polynomial.Polynomial.fit(x, ytr[crop], degree)
            # Remove target baseline, add train baseline
            adjusted_ytr[crop] = ytr[crop] - poly_train(x) + poly_target(x) 
            
        adjusted.append(adjusted_ytr)
        ctr += 1
    return adjusted

def _pad_arrays_to_same_length(x0, y_train0, y_target0, glitch_mask0, data_mask0, edges0, warping):
    # pad the arrays to the same length
    max_len = max(len(arr) for arr in x0)
    
    for i in range(len(y_train0)):
        # find pre/post edge regions
        pre_edge_len = np.sum(x0[i] < edges0[i])
        post_edge_len = np.sum(x0[i] >= edges0[i])

        total_len = pre_edge_len + post_edge_len
        pad_len = max_len - total_len
        
        if pad_len > 0:
            # use padding as a fraction of the pre-edge and post-edge lengths
            pre_edge_pad = int(pad_len / total_len * pre_edge_len)
            post_edge_pad = int(pad_len / total_len * post_edge_len)
            
            # Ensure total padding is correct
            post_edge_pad += pad_len - pre_edge_pad - post_edge_pad 
                       
            x0[i] = np.pad(x0[i], (pre_edge_pad, post_edge_pad), mode='edge')
            y_train0[i] = np.pad(y_train0[i], (pre_edge_pad, post_edge_pad), mode='edge')
            y_target0[i] = np.pad(y_target0[i], (pre_edge_pad, post_edge_pad), mode='edge')
            glitch_mask0[i] = np.pad(glitch_mask0[i], (pre_edge_pad, post_edge_pad), mode='constant', constant_values=False)
            data_mask0[i] = np.pad(data_mask0[i], (pre_edge_pad, post_edge_pad), mode='constant', constant_values=False)        
            
    return x0, y_train0, y_target0, glitch_mask0, data_mask0, edges0

def _convert_to_numpy_arrays(x0, y_train0, y_target0, glitch_mask0, data_mask0, edges0):
    # Explicitly convert everything to numpy arrays for numerical operations
    x0 = np.array(x0)
    y_train0 = np.array(y_train0)
    y_target0 = np.array(y_target0)
    glitch_mask0 = np.array(glitch_mask0)
    data_mask0 = np.array(data_mask0)
    edges0 = np.array(edges0)
    
    return x0, y_train0, y_target0, glitch_mask0, data_mask0, edges0

def split_training_data(x0, y_train0, y_target0, data_mask0, compounds0, elements0, 
               train_indices=[], test_indices=[], val_indices=[]):
    """
    Split data into train, validation, and test sets by compounds.
    
    Args:
        x0 (np.ndarray): Energy grid.
        y_train0 (np.ndarray): Training data (e.g., noisy spectra).
        y_targets0 (np.ndarray): Target data (e.g., clean spectra).
        data_mask0 (np.ndarray): Data mask.
        compounds0 (list): List of compound names corresponding to each sample.
        elements0 (list): List of element names corresponding to each sample.
        compounds_for_test (list): List of explicit compounds to use for testing.
        
    Returns:
        tuple: (x_train, y_train, y_train_target, mask_train, compound_train, element_train,
                x_val, y_val, y_val_target, mask_val, compound_val, element_val,
                x_test, y_test, y_test_target, mask_test, compound_test, element_test)
    """                          
    # Perform the split    
    # train_indices, val_indices, test_indices = split_by_compounds(
    #     compounds0, compounds_for_test, train_frac=train_split, val_frac=val_split, test_frac=test_split, random_state=42
    # )

    # Use the indices to create splits
    x_train, y_train, y_train_target, mask_train, compound_train, element_train = \
    x0[train_indices], y_train0[train_indices], y_target0[train_indices], data_mask0[train_indices], list(np.array(compounds0)[train_indices]), list(np.array(elements0)[train_indices])
    x_test, y_test, y_test_target, mask_test, compound_test, element_test = \
    x0[test_indices], y_train0[test_indices], y_target0[test_indices], data_mask0[test_indices], list(np.array(compounds0)[test_indices]), list(np.array(elements0)[test_indices])

    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")

    # Convert lists of compounds to sets
    train_compounds_set = set(np.unique(np.array(compounds0)[train_indices]))
    test_compounds_set = set(np.unique(np.array(compounds0)[test_indices]))
    
    # Check for intersections
    if train_compounds_set.isdisjoint(test_compounds_set):
        print("No compounds are shared between training and test sets.")
    else:
        print("Some compounds are shared between training and test sets.")

    train_compounds = " ".join(np.unique(np.array(compounds0)[train_indices]))
    test_compounds = " ".join(np.unique(np.array(compounds0)[test_indices]))

    print(f"Training compounds: {train_compounds}")
    print(f"Test compounds: {test_compounds}")
    
    
    if len(val_indices) > 0:
        x_val, y_val, y_val_target, mask_val, compound_val, element_val = \
            x0[val_indices], y_train0[val_indices], y_target0[val_indices], data_mask0[val_indices], list(np.array(compounds0)[val_indices]), list(np.array(elements0)[val_indices])
        print(f"Validation samples: {len(val_indices)}")

        val_compounds_set = set(np.unique(np.array(compounds0)[val_indices]))
        if val_compounds_set.isdisjoint(test_compounds_set):
            print("No compounds are shared between validation and test sets.")
        else:
            print("Some compounds are shared between validation and test sets.")

        if train_compounds_set.isdisjoint(val_compounds_set):
            print("No compounds are shared between training and validation sets.")
        else:
            print("Some compounds are shared between training and validation sets.")
        val_compounds = " ".join(np.unique(np.array(compounds0)[val_indices]))
        print(f"Validation compounds: {val_compounds}")
    else:
        x_val, y_val, y_val_target, mask_val, compound_val, element_val = None, None, None, None, None, None
        
    return x_train, y_train, y_train_target, mask_train, compound_train, element_train, \
           x_val, y_val, y_val_target, mask_val, compound_val, element_val, \
           x_test, y_test, y_test_target, mask_test, compound_test, element_test
           
def split_by_compounds(compounds, compounds_for_test=[], compounds_to_exclude=[], train_frac=0.9, val_frac=0.05, test_frac=0.05, random_state=42):
    """
    Split data into train, validation, and test sets by compounds.
    
    Args:
        y_train (torch.Tensor): Training data (e.g., noisy spectra).
        y_targets (torch.Tensor): Target data (e.g., clean spectra).
        compounds (list): List of compound names corresponding to each sample.
        compounds_for_test (list): List of explicit compounds to use for testing.
        compounds_to_exclude (list): List of compounds to exclude from the training set.
        train_frac (float): Fraction of compounds to use for training.
        val_frac (float): Fraction of compounds to use for validation.
        test_frac (float): Fraction of compounds to use for testing.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    # Get unique compounds
    unique_compounds = np.unique(compounds)
    
    # If test compounds are provided, create the test compound list
    # and adjust the split ratios to avoid creating too many or too little
    # entries in one list or another
    if len(compounds_for_test) > 0:
        N = len(compounds_for_test)
        N_train = len(unique_compounds) * train_frac
        N_val = len(unique_compounds) * val_frac
        N_test = len(unique_compounds) * test_frac - N
        
        print(f"Initial split ratios: train={train_frac}, val={val_frac}, test={test_frac}")
        test_frac = int(np.maximum(N_test / (N_train + N_val + N_test), 0))
        train_frac = int(np.minimum(N_train / (N_train + N_val + N_test), 1))
        val_frac = int(np.maximum(N_val / (N_train + N_val + N_test), 0))
        print(f"Adjusted split ratios: train={train_frac}, val={val_frac}, test={test_frac}")
    
    # Split compounds into train/val/test
    if val_frac == 0 or val_frac is None:
        train_compounds, test_compounds = train_test_split(
            unique_compounds, train_size=train_frac, random_state=random_state
        )
        val_compounds = []
    else:
        train_compounds, temp_compounds = train_test_split(
            unique_compounds, train_size=train_frac, random_state=random_state
        )
        val_compounds, test_compounds = train_test_split(
            temp_compounds, test_size=(test_frac / (val_frac + test_frac)), random_state=random_state
        )
    
    # if test compounds are provided, include them into the test set
    if compounds_for_test is not None and len(compounds_for_test) > 0:
        print(f"Compounds explicitly included into the test set:")
        print(list(compounds_for_test))
        for c in compounds_for_test:
            if c in train_compounds:
                test_compounds = np.append(test_compounds, c)
                train_compounds = np.delete(train_compounds, np.where(train_compounds == c))
            if c in val_compounds:
                test_compounds = np.append(test_compounds, c)
                val_compounds = np.delete(val_compounds, np.where(val_compounds == c))

    # exclude compounds from the splits
    print(f"Compounds explicitly excluded from the splits:")
    print(list(compounds_to_exclude))
    for c in compounds_to_exclude:
        if c in train_compounds:
            train_compounds = np.delete(train_compounds, np.where(train_compounds == c))
        if c in val_compounds:
            val_compounds = np.delete(val_compounds, np.where(val_compounds == c))
        if c in test_compounds:
            test_compounds = np.delete(test_compounds, np.where(test_compounds == c))
    
    # Get indices for each split
    train_indices = [i for i, c in enumerate(compounds) if c in train_compounds]
    val_indices = [i for i, c in enumerate(compounds) if c in val_compounds]
    test_indices = [i for i, c in enumerate(compounds) if c in test_compounds]
    
    # Convert indices into numpy arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # For testing take just one time instance for each compound
    test_indices = np.unique(np.array([np.where(np.array(compounds) == c)[0][0] for c in test_compounds]))
    
    print(f"Number of unique compounds: {len(unique_compounds)}")
    print(f"Training compounds: {len(train_compounds)}")
    print(f"Validation compounds: {len(val_compounds)}")
    print(f"Test compounds: {len(test_compounds)}")
    
    return train_indices, val_indices, test_indices

def align_spectra(reference, spectra, max_shift=np.inf, method='correlation'):
    """
    Align spectra to a reference spectrum using cross-correlation.
    
    Args:
        reference (np.ndarray): Reference spectrum for alignment
        spectra (np.ndarray): Array of spectra to align [n_samples, n_features]
        max_shift (int): Maximum allowed shift in either direction
        method (str): Method for alignment ('correlation', 'baseline', or 'peak')
        
    Returns:
        numpy.ndarray: Aligned spectra [n_samples, n_features]
    """
    aligned_spectra = np.zeros_like(spectra)
    shifts = np.zeros(len(spectra)).astype(int)
    
    for i, spectrum in enumerate(spectra):
        if method == 'correlation':
            # Cross-correlation based alignment
            corr = np.correlate(reference, spectrum, mode='full')
            shift = np.argmax(corr) - (len(reference) - 1)
            # Limit shift to max_shift
            shift = np.clip(shift, -max_shift, max_shift)
        elif method == 'peak':
            # Peak-based alignment
            ref_peak = np.argmax(reference)
            spec_peak = np.argmax(spectrum)
            shift = ref_peak - spec_peak
            shift = np.clip(shift, -max_shift, max_shift)
        elif method == 'baseline':
            # use the middle of the rising edge of the baselines
            ref_edge = np.argmin(np.abs(reference - (reference.max() - reference.min())/2))
            spec_edge = np.argmin(np.abs(spectrum - (spectrum.max() - spectrum.min())/2))
            shift = ref_edge - spec_edge
            shift = np.clip(shift, -max_shift, max_shift)
            
        shift = int(shift)
        
        # Apply the shift and store
        shifted = spectrum.copy()
        if shift > 0:
            shifted[shift:] = spectrum[:-shift]
        elif shift < 0:
            shifted[:shift] = spectrum[-shift:]
        else:
            shifted = spectrum
            
        aligned_spectra[i] = shifted
        shifts[i] = shift
        
    return aligned_spectra, shifts

def apply_shifts(spectra, shifts):
    """
    Apply the calculated shifts to the spectra.
    
    Args:
        spectra (np.ndarray): Array of spectra to align [n_samples, n_features]
        shifts (np.ndarray): Array of shifts for each spectrum
        
    Returns:
        numpy.ndarray: Aligned spectra [n_samples, n_features]
    """
    aligned_spectra = np.zeros_like(spectra)
    for i, spectrum in enumerate(spectra):
        shift = shifts[i]
        shifted = spectrum.copy()
        if shift > 0:
            shifted[shift:] = spectrum[:-shift]
        elif shift < 0:
            shifted[:shift] = spectrum[-shift:]
        else:
            shifted = spectrum
            
        aligned_spectra[i] = shifted
        
    return aligned_spectra
    
    