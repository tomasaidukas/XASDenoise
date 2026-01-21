"""
Prepare training data for the autoencoder model.
"""

import numpy as np
from xasdenoise.denoising_pipeline import PipelineConfig, DenoisingPipeline
from xasdenoise.denoising_methods.denoisers import AutoencoderDenoiser
from sklearn.model_selection import train_test_split
import tqdm

def process_training_data(spectrum_obj, config=None, denoiser=None):
    """
    Create training tensors for the autoencoder model.
    
    Args:
        spectrum_obj (Spectrum): A Spectrum object containing the spectrum data.
        config (PipelineConfig): Pipeline configuration object
        denoiser (Denoiser): Denoiser object to use for data preprocessing. If None, no denoising will be used.
        
    Returns:
        dict: A dictionary containing the processed arrays.
    """     

    # Data processing configuration used to pre-process training data
    if config is None:
        # Create a default pipeline configuration
        config = PipelineConfig(
            verbose=0,
        )
        
    # Create and run pipeline with method chaining
    pipeline = DenoisingPipeline(config)
    pipeline.load_data(spectrum_obj)
    
    if denoiser is not None:
        preprocessed_data = pipeline.process(denoiser, return_denoised_data=True)
        preprocessed_data['y'] = preprocessed_data['y_denoised']
    else:
        preprocessed_data = pipeline.preprocess_data()
    
    return preprocessed_data

def create_noise2noise_data(spectrum_obj, config=None, denoiser=None, 
                           pairing_strategy='adjacent', pair_window_size=1):
    """
    Create noise2noise training data from the preprocessed data.

    Here we assume that the input spectrum is a 2D array of (energy, noisy observations). 
    One option is to use average the noisy observatiobns and obtain an estimate of the clean spectrum.
    This option is used for noise2clean training data.
    
    For noise2noise training data, we create pairs of noisy observations.
    The pairing strategy can be:
        - 'adjacent': Only pair consecutive observations
        - 'window': Pair each observation with neighbors within pair_window_size
        - 'all_pairs': All unique pairwise combinations
    
    Args:
        spectrum_obj: Spectrum object with multiple noisy observations (energy, noisy observations)
        config: Pipeline configuration object
        denoiser: Optional denoiser for preprocessing
        pairing_strategy: 'adjacent', 'window', or 'all_pairs'
            - 'adjacent': Only pair consecutive observations
            - 'window': Pair each observation with neighbors within pair_window_size
            - 'all_pairs': All unique pairwise combinations (for temporally stable data)
        pair_window_size: The number of observations to be paired within a given window (default=1 for adjacent)
    
    Returns:
        dict: Processed data with y_train and y_target pairs
    """
    
    processed_data = process_training_data(spectrum_obj, config, denoiser)
    
    y_data = processed_data.get('y')  # Shape: (n_energy, n_pairs)
    n_energy, n_pairs0 = y_data.shape
    
    if pair_window_size > 1:
        pair_window_size = pair_window_size // 2
        
    # Create pairs based on strategy
    if pairing_strategy == 'adjacent':
        # Pair adjacent observations only
        y_train_list = []
        y_target_list = []
        y_pair_indices = []
        
        for t in range(n_pairs0 - 1):
            # Pair sequential observations
            y_train_list.append(y_data[:, t])
            y_target_list.append(y_data[:, t + 1])
            y_pair_indices.append(t)

            # Make sure A = denoise(B) and B = denoise(A)
            y_target_list.append(y_data[:, t])
            y_train_list.append(y_data[:, t + 1])
            y_pair_indices.append(t)
    
    elif pairing_strategy == 'sequential':
        # Pair sequential observations only
        # This is similar to 'adjacent' but does not create reverse pairs
        # This is useful for sequential data where we want to predict the next observation
        y_train_list = []
        y_target_list = []
        y_pair_indices = []
        
        for t in range(n_pairs0 - 1):
            y_train_list.append(y_data[:, t])
            y_target_list.append(y_data[:, t + 1])
            y_pair_indices.append(t)
              
    elif pairing_strategy == 'window_sequential':
        # Pair each observation with neighbors within a window
        y_train_list = []
        y_target_list = []
        y_pair_indices = []

        for t in range(n_pairs0):
            # Define window boundaries
            t_min = max(0, t - pair_window_size)
            t_max = min(n_pairs0, t + pair_window_size + 1)

            # Pair with all neighbors in window (excluding self)
            for t_neighbor in range(t_min, t_max):
                if t_neighbor != t:  # Exclude self-pairing
                    y_train_list.append(y_data[:, t])
                    y_target_list.append(y_data[:, t_neighbor])
                    # Keep track of original observation index such that we can identify
                    # which pairs belong together
                    y_pair_indices.append(t) 
    
    elif pairing_strategy == 'window':
        # Pair each observation with neighbors within a window
        y_train_list = []
        y_target_list = []
        y_pair_indices = []

        for t in range(n_pairs0):
            # Define window boundaries
            t_min = max(0, t - pair_window_size)
            t_max = min(n_pairs0, t + pair_window_size + 1)

            # Pair with all neighbors in window (excluding self)
            for t_neighbor in range(t_min, t_max):
                if t_neighbor != t:  # Exclude self-pairing
                    y_train_list.append(y_data[:, t])
                    y_target_list.append(y_data[:, t_neighbor])
                    # Keep track of original observation index such that we can identify
                    # which pairs belong together
                    y_pair_indices.append(t) 
                    
                    # Also add the reverse pair for symmetry
                    # Make sure A = denoise(B) and B = denoise(A)
                    y_train_list.append(y_data[:, t_neighbor])
                    y_target_list.append(y_data[:, t])
                    # Keep track of original observation index such that we can identify
                    # which pairs belong together
                    y_pair_indices.append(t) 
                            
    elif pairing_strategy == 'all_pairs':
        # All unique pairwise combinations (for temporally stable ex-situ data)
        y_train_list = []
        y_target_list = []
        y_pair_indices = []
        
        for i in range(n_pairs0):
            for j in range(i + 1, n_pairs0):
                # Add pair (i, j)
                y_train_list.append(y_data[:, i])
                y_target_list.append(y_data[:, j])
                y_pair_indices.append(i)
                        
                # Add reverse pair (j, i) for symmetry
                y_train_list.append(y_data[:, j])
                y_target_list.append(y_data[:, i])
                y_pair_indices.append(j)
    else:
        raise ValueError(f"Invalid pairing_strategy: {pairing_strategy}. "
                        f"Choose 'adjacent', 'window', or 'all_pairs'.")
    
    n_pairs = len(y_train_list)

    # Convert lists to arrays
    processed_data['y_train'] = np.array(y_train_list)  # Shape: (n_pairs, n_energy)
    processed_data['y_target'] = np.array(y_target_list)  # Shape: (n_pairs, n_energy)
    processed_data['y_pair_indices'] = np.array(y_pair_indices)
    
    # Replicate other arrays to match number of pairs
    processed_data['x'] = np.repeat(processed_data.get('x')[None, :], n_pairs, axis=0)
    
    if processed_data.get('data_mask') is not None:
        processed_data['data_mask'] = np.repeat(processed_data.get('data_mask')[None, :], n_pairs, axis=0)
    
    try:
        processed_data['compounds'] = np.repeat([spectrum_obj.metadata['compound']], n_pairs, axis=0)
    except:
        pass
    
    try:
        processed_data['elements'] = np.repeat([spectrum_obj.metadata['element']], n_pairs, axis=0)
    except:
        pass
    
    try:
        processed_data['edges'] = np.repeat([spectrum_obj.edge], n_pairs, axis=0)
    except:
        pass
    
    # Print statistics
    print(f"Pairing strategy: {pairing_strategy}")
    if pairing_strategy == 'window':
        print(f"  Pairing window size: {pair_window_size}")
    print(f"  Number of observations: {n_pairs0}")
    print(f"  Training pairs: {n_pairs}")
    print(f"  Data augmentation factor: {n_pairs / n_pairs0:.1f}x")
    
    return processed_data

def create_noise2clean(spectrum_obj, config=None, denoiser=None):
    """
    Create noise2clean training data from the preprocessed data.

    Here we assume that the input spectrum contains many noisy 
    observations of the same underlying signal.
    """
    
    # In noise2clean, target is the clean spectrum. Here we assume that for every spectrum 
    # we have multiple noisy observations passed as a 2D array containing (energy, pairs). 
    # Therefore, the clean spectrum is approximated as an average of all noisy observations.
    processed_data = process_training_data(spectrum_obj, config)
    y_train = processed_data.get('y')

    if denoiser is not None:
        processed_data = process_training_data(spectrum_obj, config, denoiser)
    y_target = processed_data.get('y')
    
    # Target dataset will be the the average of all noisy observations within a given spectrum
    processed_data['y_train'] = y_train.T
    processed_data['y_target'] = np.repeat(np.mean(y_target, axis=1)[:, None], y_train.shape[1], axis=1).T
    
    # Create the other arrays through replication for training
    num_pairs = processed_data['y_train'].shape[0]
    
    processed_data['x'] = np.repeat(processed_data.get('x')[None, :], num_pairs, axis=0)
    if processed_data.get('data_mask') is not None:
        processed_data['data_mask'] = np.repeat(processed_data.get('data_mask')[None, :], num_pairs, axis=0)
    
    try:
        processed_data['compounds'] = np.repeat([spectrum_obj.metadata['compound']], num_pairs, axis=0)
    except:
        pass
    
    try:
        processed_data['elements'] = np.repeat([spectrum_obj.metadata['element']], num_pairs, axis=0)
    except:
        pass
    
    try:
        processed_data['edges'] = np.repeat([spectrum_obj.edge], num_pairs, axis=0)
    except:
        pass
    
    return processed_data

def prepare_training_data(spectrum_obj_list, config=None, denoiser=None, 
                          method='noise2noise', pairing_strategy='adjacent', pair_window_size=1):
    """
    Prepare training data for the autoencoder model. 
    
    The input is either a list of Spectrum objects or a single Spectrum object.
    
    Args:
        spectrum_obj (Spectrum): A Spectrum object containing the spectrum data.
        config (PipelineConfig): Pipeline configuration object for data processing.
        denoiser (Denoiser): Denoiser object for denoising the data.
        method (str): The method to use for training data preparation ('noise2noise' or 'noise2clean').
        pairing_strategy (str): Pairing strategy for noise2noise ('adjacent', 'window', 'all_pairs').
        pair_window_size (int): Maximum index gap for 'window' pairing strategy.
    """
    valid_keys = ['x', 'y_train', 'y_target', 'y_pair_indices', 'data_mask', 'compounds', 'elements', 'edges']
    processed_data_lists = {}
    
    if not isinstance(spectrum_obj_list, list):
        spectrum_obj_list = [spectrum_obj_list]

    if method == 'noise2clean' and (pairing_strategy != 'adjacent' or pair_window_size != 1):
        print("Warning: 'pairing_strategy' and 'pair_window_size' are ignored in 'noise2clean' method.")
        
    # First, collect all arrays in lists
    tqdm_iterator = tqdm.tqdm(spectrum_obj_list, desc="Processing spectra for training data")
    for idx, spectrum_obj in enumerate(tqdm_iterator):
        if method == 'noise2noise':
            processed_data = create_noise2noise_data(spectrum_obj, config, denoiser, pairing_strategy, pair_window_size)
        elif method == 'noise2clean':
            processed_data = create_noise2clean(spectrum_obj, config, denoiser)
        else:
            raise ValueError("Invalid method. Choose either 'noise2noise' or 'noise2clean'.")
       
        processed_data_lists = _collect_arrays_in_lists(processed_data_lists, processed_data, valid_keys)
    
    # Pad arrays to same length and then concatenate
    processed_data_all = _pad_and_concatenate_arrays(processed_data_lists, valid_keys)
    
    return processed_data_all
       
def _collect_arrays_in_lists(processed_data_lists, new_data, keys):
    """
    Collect arrays in lists instead of concatenating immediately.
    
    Args:
        processed_data_lists (dict): Dictionary containing lists of arrays.
        new_data (dict): New data to be added to the lists.
        keys (list): Valid keys to process.
        
    Returns:
        dict: Updated dictionary with arrays collected in lists.
    """
    for key, value in new_data.items():
        if key in keys:
            if key not in processed_data_lists:
                processed_data_lists[key] = []
            processed_data_lists[key].append(value)
                
    return processed_data_lists

def _pad_and_concatenate_arrays(processed_data_lists, keys):
    """
    Pad arrays to same length and concatenate them.
    
    Args:
        processed_data_lists (dict): Dictionary containing lists of arrays.
        keys (list): Valid keys to process.
        
    Returns:
        dict: Dictionary with padded and concatenated arrays.
    """
    processed_data_all = {}
    
    # First, convert lists to arrays and collect all samples
    all_arrays = {}
    for key in keys:
        if key in processed_data_lists:
            # Concatenate along the first axis (samples)
            all_arrays[key] = []
            for array_batch in processed_data_lists[key]:
                for i in range(array_batch.shape[0]):
                    all_arrays[key].append(array_batch[i])
    
    # Check if we need padding by looking at array lengths
    needs_padding = False
    if 'x' in all_arrays and all_arrays['x']:
        lengths = [len(arr) for arr in all_arrays['x']]
        needs_padding = len(set(lengths)) > 1
    
    if needs_padding:
        # Apply padding to make all arrays the same length
        padded_x, padded_y_train, padded_y_target, padded_data_mask = _pad_arrays_to_same_length(
            all_arrays.get('x', []),
            all_arrays.get('y_train', []),
            all_arrays.get('y_target', []),
            all_arrays.get('data_mask', []),
            all_arrays.get('edges', [])
        )
        
        # Update the arrays with padded versions
        if 'x' in all_arrays:
            all_arrays['x'] = padded_x
        if 'y_train' in all_arrays:
            all_arrays['y_train'] = padded_y_train
        if 'y_target' in all_arrays:
            all_arrays['y_target'] = padded_y_target
        if 'data_mask' in all_arrays:
            all_arrays['data_mask'] = padded_data_mask
    
    # Now concatenate all arrays
    for key in keys:
        if key in all_arrays:
            processed_data_all[key] = np.array(all_arrays[key])
    
    return processed_data_all

def _pad_arrays_to_same_length(x0, y_train0, y_target0, data_mask0, edges0):
    """
    Pad arrays to the same length - updated version for the new data structure.
    
    Args:
        x0 (list): List of energy arrays
        y_train0 (list): List of training spectra
        y_target0 (list): List of target spectra  
        data_mask0 (list): List of data masks
        edges0 (list): List of edge positions
        
    Returns:
        tuple: Padded arrays as lists
    """
    if not x0:  # Empty list
        return x0, y_train0, y_target0, data_mask0
    
    # Find the maximum length
    max_len = max(len(arr) for arr in x0)
    
    # Pad each array
    for i in range(len(x0)):
        current_len = len(x0[i])
        
        if current_len < max_len:
            # Find pre/post edge regions
            edge_pos = edges0[i] if edges0 and i < len(edges0) else np.mean(x0[i])
            pre_edge_len = np.sum(x0[i] < edge_pos)
            post_edge_len = np.sum(x0[i] >= edge_pos)
            
            total_len = pre_edge_len + post_edge_len
            pad_len = max_len - total_len
            
            if pad_len > 0:
                # Use padding as a fraction of the pre-edge and post-edge lengths
                pre_edge_pad = int(pad_len / total_len * pre_edge_len) if total_len > 0 else pad_len // 2
                post_edge_pad = pad_len - pre_edge_pad
                
                # Pad the arrays
                x0[i] = np.pad(x0[i], (pre_edge_pad, post_edge_pad), mode='edge')
                y_train0[i] = np.pad(y_train0[i], (pre_edge_pad, post_edge_pad), mode='edge')
                y_target0[i] = np.pad(y_target0[i], (pre_edge_pad, post_edge_pad), mode='edge')
                
                # Handle data_mask which might be None or contain None values
                if data_mask0 and i < len(data_mask0) and data_mask0[i] is not None:
                    data_mask0[i] = np.pad(data_mask0[i], (pre_edge_pad, post_edge_pad), mode='constant', constant_values=False)
                elif data_mask0 and i < len(data_mask0):
                    # Create a default mask if it was None
                    data_mask0[i] = np.ones(max_len, dtype=bool)
    
    return x0, y_train0, y_target0, data_mask0

def train_autoencoder_model(spectrum_obj_list, config=None, denoiser_preproc=None, method='noise2noise',
                        pairing_strategy='adjacent', pair_window_size=1,
                        model_params=None, training_params=None):
    """
    Prepare training data and train the autoencoder model.
    
    Args:
        spectrum_obj (Spectrum or list): A Spectrum object or a list of Spectrum objects containing the spectrum data.
        config (PipelineConfig): Pipeline configuration object for data processing.
        denoiser_preproc (Denoiser): Denoiser object for denoising the data during preprocessing.
        method (str): The method to use for training data preparation ('noise2noise' or 'noise2clean').
        pairing_strategy (str): Pairing strategy for noise2noise ('adjacent', 'window', 'all_pairs').
        pair_window_size (int): Window size within which the data are paired.
        model_path (str): Path to save the trained model.
        training_params (dict): Dictionary of training parameters.
        
    Returns:
        EncoderModel: Trained autoencoder model.
    """
    if config is None:
        # Create a default pipeline configuration
        config = PipelineConfig(
            verbose=0,
        )
        
    processed_data = prepare_training_data(spectrum_obj_list, config, denoiser_preproc, method=method,
                                           pairing_strategy=pairing_strategy, pair_window_size=pair_window_size)

    # Parameter which determines if data should be shuffled
    if pairing_strategy == 'window':
        dont_shuffle = True
    else:
        dont_shuffle = False
        
    # Model initialization parameters with defaults
    model_params = {
        'model_type': model_params.get('model_type', 'conv'),
        'device': model_params.get('device', 'auto'),
        'gpu_index': model_params.get('gpu_index', None),
        'num_layers': model_params.get('num_layers', 4),
        'kernel_size': model_params.get('kernel_size', 11),
        'dropout_rate': model_params.get('dropout_rate', 0),
        'channels': model_params.get('channels', None),
        'normalization_method': model_params.get('normalization_method', None),
        'bias': model_params.get('bias', False),
        'output_mode': model_params.get('output_mode', 'direct'),
        'output_nonnegativity': model_params.get('output_nonnegativity', False),
        'transpose_input': model_params.get('transpose_input', False)
    }
    
    # Training parameters with defaults
    training_params = {
        'batch_size': training_params.get('batch_size', 16),
        'num_epochs': training_params.get('num_epochs', 100),
        'learning_rate': training_params.get('learning_rate', 1e-4),
        'augment_data': training_params.get('augment_data', False),
        'remove_padded_regions': training_params.get('remove_padded_regions', False),
        'save_path': training_params.get('save_path', None),
        'early_stopping_patience': training_params.get('early_stopping_patience', 50),
        'weight_decay': training_params.get('weight_decay', 1e-5),
        'loss_weights': training_params.get('loss_weights', None),
        'dont_shuffle': training_params.get('dont_shuffle', dont_shuffle)
    }

    # Define model type and initialize the denoiser
    denoiser = AutoencoderDenoiser(**model_params)

    # Train the model
    training_metrics = denoiser.train_model(
        y_train=processed_data['y_train'],  # Noisy spectra as input
        y_target=processed_data['y_target'],  # Clean spectra as target
        mask_train=processed_data['data_mask'],  # Data mask
        **training_params
    )

    return denoiser, training_metrics


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
        
        if N_test < 0:
            print(f"Warning: Too many compounds are explicitly included into the test set.")
            print(f"Adjusting the split ratios to avoid negative number of test compounds.")
            N_train = N_train + abs(N_test)
            N_test = 0
        
        print(f"Initial split ratios: train={train_frac}, val={val_frac}, test={test_frac}")
        test_frac = float(np.maximum(N_test / (N_train + N_val + N_test), 0))
        train_frac = float(np.minimum(N_train / (N_train + N_val + N_test), 1))
        val_frac = float(np.maximum(N_val / (N_train + N_val + N_test), 0))
        print(f"Adjusted split ratios: train={train_frac}, val={val_frac}, test={test_frac}")
    
    # Split compounds into train/val/test
    if train_frac == 1.0:
        train_compounds = unique_compounds
        val_compounds = []
        test_compounds = []
    elif val_frac == 0 or val_frac is None:
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
    
    # For testing take just one observation for each compound
    test_indices = np.unique(np.array([np.where(np.array(compounds) == c)[0][0] for c in test_compounds]))
    
    print(f"Number of unique compounds: {len(unique_compounds)}")
    print(f"Training compounds: {len(train_compounds)}")
    print(f"Validation compounds: {len(val_compounds)}")
    print(f"Test compounds: {len(test_compounds)}")
    
    return train_indices, val_indices, test_indices