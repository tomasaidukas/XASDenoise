"""
Prepare training data for the encoder model.
"""

import numpy as np
from xasdenoise.denoising_pipeline import PipelineConfig, DenoisingPipeline
from xasdenoise.denoising_methods.denoisers import EncoderDenoiser
from sklearn.model_selection import train_test_split
import tqdm

def process_training_data(spectrum_obj, config=None, denoiser=None):
    """
    Create training tensors for the encoder model.
    
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
        preprocessed_data['y'] = preprocessed_data['denoised']
    else:
        preprocessed_data = pipeline.preprocess_data()
    
    return preprocessed_data

def create_noise2noise_data(spectrum_obj, config=None, denoiser=None):
    """
    Create noise2noise training data from the preprocessed data.

    Here we assume that the input spectrum time series contains many noisy observations of the same underlying signal.
    """
    
    processed_data = process_training_data(spectrum_obj, config, denoiser)
    # processed_data = {}
    # processed_data['y'] = spectrum_obj.spectrum
    # processed_data['x'] = spectrum_obj.energy
    # processed_data['data_mask'] = spectrum_obj.data_mask if hasattr(spectrum_obj, 'data_mask') else None
    
    # In noise2noise, target is the neighboring time point
    y_train = processed_data.get('y')
    y_target = processed_data.get('y')
    y_target = np.roll(y_target, shift=1, axis=1) 
    
    # Remove the first time point from training and last from target to avoid circular shift
    processed_data['y_train'] = y_train[:,1:].T
    processed_data['y_target'] = y_target[:,:-1].T
    
    # Create the other arrays through replication for training
    num_times = processed_data['y_train'].shape[0]
    
    processed_data['x'] = np.repeat(processed_data.get('x')[None, :], num_times, axis=0)
    if processed_data.get('data_mask') is not None:
        processed_data['data_mask'] = np.repeat(processed_data.get('data_mask')[None, :], num_times, axis=0)
    processed_data['compounds'] = np.repeat([spectrum_obj.metadata['compound']], num_times, axis=0)
    processed_data['elements'] = np.repeat([spectrum_obj.metadata['element']], num_times, axis=0)
    processed_data['edges'] = np.repeat([spectrum_obj.edge], num_times, axis=0)

    return processed_data

def create_noise2clean(spectrum_obj, config=None, denoiser=None):
    """
    Create noise2clean training data from the preprocessed data.

    Here we assume that the input spectrum time series contains many noisy observations of the same underlying signal.
    """
    
    # In noise2clean, target is the clean spectrum, which will be taken as the time-averaged spectrum
    # and can also be pre-denoised. Do not denoise the training data (it must be noisy)
    processed_data = process_training_data(spectrum_obj, config)
    y_train = processed_data.get('y')

    if denoiser is not None:
        processed_data = process_training_data(spectrum_obj, config, denoiser)
    y_target = processed_data.get('y')
    
    # Target dataset will be the the time-averaged spectrum
    processed_data['y_train'] = y_train.T
    processed_data['y_target'] = np.repeat(np.mean(y_target, axis=1)[:, None], y_train.shape[1], axis=1).T
    
    # Create the other arrays through replication for training
    num_times = processed_data['y_train'].shape[0]
    
    processed_data['x'] = np.repeat(processed_data.get('x')[None, :], num_times, axis=0)
    if processed_data.get('data_mask') is not None:
        processed_data['data_mask'] = np.repeat(processed_data.get('data_mask')[None, :], num_times, axis=0)
    processed_data['compounds'] = np.repeat([spectrum_obj.metadata['compound']], num_times, axis=0)
    processed_data['elements'] = np.repeat([spectrum_obj.metadata['element']], num_times, axis=0)
    processed_data['edges'] = np.repeat([spectrum_obj.edge], num_times, axis=0)

    return processed_data

def prepare_training_data(spectrum_obj_list, config=None, denoiser=None, method='noise2noise'):
    """
    Prepare training data for the encoder model. 
    
    The input is either a list of Spectrum objects or a single Spectrum object.
    
    Args:
        spectrum_obj (Spectrum): A Spectrum object containing the spectrum data.
        config (PipelineConfig): Pipeline configuration object for data processing.
        denoiser (Denoiser): Denoiser object for denoising the data.
        method (str): The method to use for training data preparation ('noise2noise' or 'noise2clean').
    """
    valid_keys = ['x', 'y_train', 'y_target', 'data_mask', 'compounds', 'elements', 'edges']
    processed_data_all = {}
    
    if not isinstance(spectrum_obj_list, list):
        spectrum_obj_list = [spectrum_obj_list]

    tqdm_iterator = tqdm.tqdm(spectrum_obj_list, desc="Processing spectra for training data")
    for idx, spectrum_obj in enumerate(tqdm_iterator):
        if method == 'noise2noise':
            processed_data = create_noise2noise_data(spectrum_obj, config, denoiser)
        elif method == 'noise2clean':
            processed_data = create_noise2clean(spectrum_obj, config, denoiser)
        else:
            raise ValueError("Invalid method. Choose either 'noise2noise' or 'noise2clean'.")
       
        processed_data_all = _extend_dictionary(processed_data_all, processed_data, valid_keys)
    
    return processed_data_all
       
def _extend_dictionary(processed_data, new_data, keys):
    """
    Extend the processed data dictionary with new data by concatenating arrays.
    
    Args:
        processed_data (dict): Existing processed data dictionary.
        new_data (dict): New data to be added to the processed data.
        
    Returns:
        dict: Updated processed data dictionary with concatenated arrays.
    """
    for key, value in new_data.items():
        if key in keys:
            if key in processed_data:
                processed_data[key] = np.concatenate((processed_data[key], value), axis=0)
            else:
                processed_data[key] = value
                
    return processed_data
    
def train_encoder_model(spectrum_obj_list, config=None, denoiser_preproc=None, method='noise2noise',
                        model_path=None, **encoder_kwargs):
    """
    Prepare training data and train the encoder model.
    
    Args:
        spectrum_obj (Spectrum or list): A Spectrum object or a list of Spectrum objects containing the spectrum data.
        config (PipelineConfig): Pipeline configuration object for data processing.
        denoiser_preproc (Denoiser): Denoiser object for denoising the data during preprocessing.
        method (str): The method to use for training data preparation ('noise2noise' or 'noise2clean').
        model_path (str): Path to save the trained model.
        **encoder_kwargs: Additional arguments for EncoderDenoiser and training. Supported keys:
            - model_type (str): Default 'conv'
            - device (str): Default 'auto'
            - gpu_index (int): Default None
            - num_layers (int): Default 4
            - kernel_size (int): Default 11
            - dropout_rate (float): Default 0
            - channels (int): Default None
            - batch_size (int): Default 16
            - epochs (int): Default 100
            - learning_rate (float): Default 1e-4
            - remove_padded_regions (bool): Default False
            - augment_data (bool): Default False
        
    Returns:
        EncoderModel: Trained encoder model.
    """
    if config is None:
        # Create a default pipeline configuration
        config = PipelineConfig(
            verbose=0,
        )
        
    processed_data = prepare_training_data(spectrum_obj_list, config, denoiser_preproc, method=method)

    # Model initialization parameters with defaults
    model_params = {
        'model_type': encoder_kwargs.get('model_type', 'conv'),
        'device': encoder_kwargs.get('device', 'auto'),
        'gpu_index': encoder_kwargs.get('gpu_index', None),
        'num_layers': encoder_kwargs.get('num_layers', 4),
        'kernel_size': encoder_kwargs.get('kernel_size', 11),
        'dropout_rate': encoder_kwargs.get('dropout_rate', 0),
        'channels': encoder_kwargs.get('channels', None),
    }
    
    # Training parameters with defaults
    training_params = {
        'batch_size': encoder_kwargs.get('batch_size', 16),
        'num_epochs': encoder_kwargs.get('epochs', 100),
        'learning_rate': encoder_kwargs.get('learning_rate', 1e-4),
        'augment_data': encoder_kwargs.get('augment_data', False),
        'remove_padded_regions': encoder_kwargs.get('remove_padded_regions', False),
        'save_path': model_path,
        'early_stopping_patience': encoder_kwargs.get('early_stopping_patience', 50),
        'weight_decay': encoder_kwargs.get('weight_decay', 1e-5),
    }

    # Define model type and initialize the denoiser
    denoiser = EncoderDenoiser(**model_params)

    # Train the model
    training_metrics = denoiser.train_model(
        y_train=processed_data['y_train'],  # Noisy spectra as input
        y_target=processed_data['y_target'],  # Clean spectra as target
        mask_train=processed_data['data_mask'],  # Data mask
        **training_params
    )

    return denoiser


# def split_training_data(x0, y_train0, y_target0, data_mask0, compounds0, elements0, 
#                train_indices=[], test_indices=[], val_indices=[]):
#     """
#     Split data into train, validation, and test sets by compounds.
    
#     Args:
#         x0 (np.ndarray): Energy grid.
#         y_train0 (np.ndarray): Training data (e.g., noisy spectra).
#         y_targets0 (np.ndarray): Target data (e.g., clean spectra).
#         data_mask0 (np.ndarray): Data mask.
#         compounds0 (list): List of compound names corresponding to each sample.
#         elements0 (list): List of element names corresponding to each sample.
#         compounds_for_test (list): List of explicit compounds to use for testing.
        
#     Returns:
#         tuple: (x_train, y_train, y_train_target, mask_train, compound_train, element_train,
#                 x_val, y_val, y_val_target, mask_val, compound_val, element_val,
#                 x_test, y_test, y_test_target, mask_test, compound_test, element_test)
#     """                          
#     # Perform the split    
#     # train_indices, val_indices, test_indices = split_by_compounds(
#     #     compounds0, compounds_for_test, train_frac=train_split, val_frac=val_split, test_frac=test_split, random_state=42
#     # )

#     # Use the indices to create splits
#     x_train, y_train, y_train_target, mask_train, compound_train, element_train = \
#     x0[train_indices], y_train0[train_indices], y_target0[train_indices], data_mask0[train_indices], list(np.array(compounds0)[train_indices]), list(np.array(elements0)[train_indices])
#     x_test, y_test, y_test_target, mask_test, compound_test, element_test = \
#     x0[test_indices], y_train0[test_indices], y_target0[test_indices], data_mask0[test_indices], list(np.array(compounds0)[test_indices]), list(np.array(elements0)[test_indices])

#     print(f"Training samples: {len(train_indices)}")
#     print(f"Test samples: {len(test_indices)}")

#     # Convert lists of compounds to sets
#     train_compounds_set = set(np.unique(np.array(compounds0)[train_indices]))
#     test_compounds_set = set(np.unique(np.array(compounds0)[test_indices]))
    
#     # Check for intersections
#     if train_compounds_set.isdisjoint(test_compounds_set):
#         print("No compounds are shared between training and test sets.")
#     else:
#         print("Some compounds are shared between training and test sets.")

#     train_compounds = " ".join(np.unique(np.array(compounds0)[train_indices]))
#     test_compounds = " ".join(np.unique(np.array(compounds0)[test_indices]))

#     print(f"Training compounds: {train_compounds}")
#     print(f"Test compounds: {test_compounds}")
    
    
#     if len(val_indices) > 0:
#         x_val, y_val, y_val_target, mask_val, compound_val, element_val = \
#             x0[val_indices], y_train0[val_indices], y_target0[val_indices], data_mask0[val_indices], list(np.array(compounds0)[val_indices]), list(np.array(elements0)[val_indices])
#         print(f"Validation samples: {len(val_indices)}")

#         val_compounds_set = set(np.unique(np.array(compounds0)[val_indices]))
#         if val_compounds_set.isdisjoint(test_compounds_set):
#             print("No compounds are shared between validation and test sets.")
#         else:
#             print("Some compounds are shared between validation and test sets.")

#         if train_compounds_set.isdisjoint(val_compounds_set):
#             print("No compounds are shared between training and validation sets.")
#         else:
#             print("Some compounds are shared between training and validation sets.")
#         val_compounds = " ".join(np.unique(np.array(compounds0)[val_indices]))
#         print(f"Validation compounds: {val_compounds}")
#     else:
#         x_val, y_val, y_val_target, mask_val, compound_val, element_val = None, None, None, None, None, None
        
#     return x_train, y_train, y_train_target, mask_train, compound_train, element_train, \
#            x_val, y_val, y_val_target, mask_val, compound_val, element_val, \
#            x_test, y_test, y_test_target, mask_test, compound_test, element_test
           
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
    
    # For testing take just one time instance for each compound
    test_indices = np.unique(np.array([np.where(np.array(compounds) == c)[0][0] for c in test_compounds]))
    
    print(f"Number of unique compounds: {len(unique_compounds)}")
    print(f"Training compounds: {len(train_compounds)}")
    print(f"Validation compounds: {len(val_compounds)}")
    print(f"Test compounds: {len(test_compounds)}")
    
    return train_indices, val_indices, test_indices