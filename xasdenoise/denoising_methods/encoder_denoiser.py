"""
Autoencoder denoiser class.
"""

import numpy as np
import gc
from scipy import interpolate
from xasdenoise.denoising_methods.denoising_utils import downsample_data
from tqdm import tqdm

# Conditional imports for torch and gpytorch
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    # warnings.warn("PyTorch not available - GPDenoiser will not be functional")
    


# Make an encoder denoising class which takes a loaded encoder model and denoises the data
class EncoderDenoiser:
    """
    A class to perform signal denoising using an encoder model.
    
    Note: Requires PyTorch. If not available, raises ImportError on initialization.
    """
    def __init__(self, model_type='conv', signal_length=None, device='auto', gpu_index=0, num_layers=4, kernel_size=7, channels=None, dropout_rate=0):
        """
        Initialize the EncoderDenoiser with a specified autoencoder architecture.

        Args:
            model_type (str): Type of autoencoder ('linear', 'conv', 'improved_conv', 'unet', 'transformer'). Defaults to 'conv'.
            signal_length (int, optional): Length of the input signal (required for 'linear' model). Defaults to None.
            device (str): Device to use for training and inference. Options: 'auto', 'cpu', 'cuda', 'mps'. Defaults to 'auto'.
            gpu_index (int): GPU index to use when device is 'cuda'. Defaults to 0.
            num_layers (int): Number of layers in the autoencoder. Defaults to 4.
            kernel_size (int): Kernel size for convolutional layers. Defaults to 7.
            channels (list, optional): Number of channels in each layer. Defaults to None.
            dropout_rate (float): Dropout rate for regularization. Defaults to 0.
        """
        # Check dependencies before initialization
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GPDenoiser but not available. "
                "Install with: pip install torch"
            )
        
        self.model_type = model_type
        self.norm_params = {}
        self.verbose = True
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.gpu_index = gpu_index
        
        # Initialize device - auto-detect if 'auto' is specified
        if device == 'auto':
            self.device = self._get_best_device()
        else:
            self.device = device
            
        # Initialize the computation device
        self.gpu_device = self._initialize_device()
        
        if model_type == 'linear':
            if signal_length is None:
                raise ValueError("For 'linear' model, 'signal_length' must be provided.")
            self.encoder_model = DenoisingAutoencoder(signal_length, dropout_rate=dropout_rate).to(self.device)
        elif model_type == 'conv':
            self.encoder_model = ConvDenoisingAutoencoder(num_layers=self.num_layers, kernel_size=self.kernel_size, channels=channels, dropout_rate=dropout_rate).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'linear', 'conv'.")

        self.encoder_model.train()  # Set the model to training mode
      
    def _get_best_device(self):
        """
        Automatically detect and return the best available device.

        Returns:
            str: Best available device ('cuda', 'mps', or 'cpu').
        """
        if torch.cuda.is_available():
            return f'cuda:{self.gpu_index}' if hasattr(self, 'gpu_index') else 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _initialize_device(self):
        """
        Initialize and return the computation device (GPU or CPU).

        Returns:
            torch.device: The computation device to use.
        """
        
        if self.device.startswith('cuda') and torch.cuda.is_available():
            device = torch.device(self.device)
            if ':' in self.device:
                torch.cuda.set_device(device)  # Specify your GPU device index
            torch.empty(1, device=device)  # Initialize CUDA context on the target GPU
            self._clean_gpu_memory()

            if self.verbose:
                print(f"Using CUDA device: {device}, {torch.cuda.get_device_name(device)}")
        elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            if self.verbose:
                print("Using Apple Silicon MPS device.")
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
        
        if self.device in ['cuda', 'mps']:
            try:
                # Delete class attributes that might hold GPU memory
                for attr in ["x_train", "y_train", "noise_train", "x_predict", "noise_redict", "model", "likelihood"]:
                    if hasattr(self, attr):
                        setattr(self, attr, None)

                # Call garbage collector and clear cache
                gc.collect()
                
                # CUDA-specific memory cleanup
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                
                # MPS-specific memory cleanup
                elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()

                # if self.verbose:
                #     print("GPU memory has been successfully cleaned.")
                    
            except Exception as e:
                print(f"Error occurred while cleaning GPU memory: {e}")
    
    def _clean_model(self):
        """
        Clean the model by deleting its attributes and clearing GPU memory.
        """
        
        if hasattr(self, 'encoder_model'):
            setattr(self, 'encoder_model', None)

         # Now systematically search for and clean any tensor attributes
        for attr_name in list(vars(self).keys()):
            attr_val = getattr(self, attr_name)
            
            # Clean PyTorch tensors on any GPU device
            if isinstance(attr_val, torch.Tensor) and attr_val.device in ['cuda', 'mps']:
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
                    
    def to_tensor(self, arr):
        """
        Convert the input array to a PyTorch tensor.
        
        Args:
            arr (np.ndarray): Input array to convert.
            
        Returns:
            torch.Tensor: Converted PyTorch tensor.
        """
        if arr is not None:
            if isinstance(arr, bool):
                return torch.tensor(arr, dtype=torch.bool).to(self.device)
            else:
                return torch.tensor(arr, dtype=torch.float32).to(self.device)
        else:
            return None
        
    def normalize_data(self, y, compute_norm=True):
        """
        Normalize the training data to the range [-1, 1].

        Args:
            y (torch.Tensor): Training data.
            compute_norm (bool): Whether to compute normalization parameters. Defaults to True.

        Returns:
            torch.Tensor: Normalized training data.
        """
        if compute_norm:
            y_max = y.max(axis=1).values[:, None]
            y_min = y.min(axis=1).values[:, None]
            self.norm_params['y'] = (y_min, y_max)
        y_min, y_max = self.norm_params['y']
        return 2 * (y - y_min) / (y_max - y_min) - 1
    
    def denormalize_data(self, y):
        """
        Denormalize the denoised data using the stored normalization parameters.

        Args:
            y (torch.Tensor): Denoised data.

        Returns:
            torch.Tensor: Denormalized denoised data.
        """
        y_min, y_max = self.norm_params['y']
        return (y + 1) * (y_max - y_min) / 2 + y_min
    
    def train_model(self, y_train, y_target, mask_train=None, y_val=None, y_val_target=None, mask_val=None,
            batch_size=32, num_epochs=1000, learning_rate=1e-4, save_path=None, augment_data=False, noise2noise=False,
            remove_padded_regions=True, randomized_masking=False, kweighted_loss=False, loss_weights=None,
            early_stopping_patience=50, dropout_rate=0.1, weight_decay=1e-5):
        """
        Train the encoder model using the given training data with optional masking.

        Args:
            y_train (torch.Tensor): Noisy spectra (input for training).
            y_target (torch.Tensor): Clean spectra (target for training).
            mask_train (torch.Tensor, optional): Binary mask (1 for valid, 0 for padded values). Defaults to None.
            y_val (torch.Tensor, optional): Noisy validation spectra. Defaults to None.
            y_val_target (torch.Tensor, optional): Clean validation spectra. Defaults to None.
            mask_val (torch.Tensor, optional): Mask for validation data. Defaults to None.
            batch_size (int): Batch size for training. Defaults to 32.
            num_epochs (int): Number of epochs for training. Defaults to 1000.
            learning_rate (float): Learning rate for the optimizer. Defaults to 1e-4.
            save_path (str, optional): Path to save the trained model. Defaults to None.
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping. Defaults to 50.
            dropout_rate (float): Dropout rate for regularization. Defaults to 0.1.
            weight_decay (float): L2 regularization strength. Defaults to 1e-5.
        """
        # Convert to tensors and initialize arrays
        y_train, y_target, mask_train, y_val, y_val_target, mask_val = self.to_tensor(y_train), self.to_tensor(y_target), self.to_tensor(mask_train), self.to_tensor(y_val), self.to_tensor(y_val_target), self.to_tensor(mask_val)
        if mask_train is None:
            mask_train = torch.ones_like(y_train)
        if y_val is not None and mask_val is None:
            mask_val = torch.ones_like(y_val)
        if loss_weights is not None:
            loss_weights = self.to_tensor(loss_weights)
            loss_weights0 = loss_weights.clone()
            
        # Normalize data
        if y_val is not None:
            y_val = self.normalize_data(y_val)
            y_val_target = self.normalize_data(y_val_target, compute_norm=False)
            
        y_train = self.normalize_data(y_train)
        y_target = self.normalize_data(y_target, compute_norm=False)

        # Prepare datasets
        train_dataset = TensorDataset(y_train, y_target, mask_train) if mask_train is not None else TensorDataset(y_train, y_target)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if y_val is not None and y_val_target is not None:
            val_dataset = TensorDataset(y_val, y_val_target, mask_val) if mask_val is not None else TensorDataset(y_val, y_val_target)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        # Define loss function and optimizer                
        criterion = nn.MSELoss(reduction='none')  # Compute per-element loss        
        # criterion = nn.L1Loss(reduction='none') 
        
        # Add weight decay
        optimizer = optim.Adam(self.encoder_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # scheduler = StepLR(optimizer, step_size=int(0.6 * num_epochs), gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Dictionary to store metrics
        metrics = {
            'loss': 0.0,
            'running_loss': [],
            'best_loss': float('inf'),
            'val_loss': [],  # Add validation loss tracking
            'epoch_losses': [],  # Add epoch loss tracking
            'val_losses': []  # Add validation epoch losses
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        update_freq = 5
        
        # Training loop
        for epoch in range(num_epochs):
            # Create progress bar for this epoch with custom formatting
            with tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{num_epochs}",
                miniters=update_freq,  # Only update progress every N iterations
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',  # Customize bar format
                position=0, 
                leave=True
            ) as pbar:
                    
                self.encoder_model.train()
                epoch_loss = 0.0
                for batch in train_loader:
                    y_batch, y_target_batch, mask_batch = batch
                        
                    # The spectra here are not of uniform lengths and are instead padded with zeros.
                    # Use the mask to crop the padded values. This avoids having zeros in the CNN layers and
                    # random cropping of all spectra within the batch acts as a data augmentation technique.
                    if remove_padded_regions and mask_batch is not None:
                        nonzero = mask_batch.all(dim=0)
                        y_batch = y_batch[:, nonzero]
                        y_target_batch = y_target_batch[:, nonzero]
                        mask_batch = mask_batch[:, nonzero]
                        if loss_weights is not None:
                            loss_weights = loss_weights0[:, nonzero]


                    # Mask data or the loss randomly with 50-50 chance
                    if randomized_masking and mask_batch is not None:
                        if torch.rand(1).item() < 0.5:
                            nonzero = mask_batch.all(dim=0)
                            y_batch = y_batch[:, nonzero]
                            y_target_batch = y_target_batch[:, nonzero]
                            mask_batch = mask_batch[:, nonzero]  
                                
                    # Data augmentation for XAS (apply more aggressively to prevent overfitting)
                    if augment_data and epoch < num_epochs * 0.8:  # Apply for 80% of training
                        if noise2noise:
                            aug_params = self._get_random_aug_params(y_batch.shape, device=y_batch.device)
                            y_batch = self._augment_xas_data_noise2noise(y_batch, aug_params)
                            y_target_batch = self._augment_xas_data_noise2noise(y_target_batch, aug_params)
                        else:
                            y_batch = self._augment_xas_data(y_batch)
                                    
                    # Forward pass
                    outputs = self.encoder_model(y_batch)
                    
                    # Compute masked loss
                    loss = criterion(outputs, y_target_batch)  # Per-element loss
                    if loss_weights is not None:
                        loss = loss * loss_weights
                    loss = (loss * mask_batch).sum() / mask_batch.sum()  # Compute mean over valid values only

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.encoder_model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Update metrics
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    metrics['loss'] = batch_loss
                    metrics['running_loss'].append(batch_loss)
                    
                    # Compute running average over last 20 batches
                    running_avg = np.mean(metrics['running_loss'][-20:])
                    
                    # Update progress bar with current metrics
                    postfix_dict = {
                        'loss': f"{batch_loss:.8f}",
                        'avg_loss': f"{running_avg:.8f}",
                        'best': f"{metrics['best_loss']:.8f}"
                    }
                    
                    # Add validation loss to progress bar if available
                    if len(metrics['val_losses']) > 0:
                        postfix_dict['val_loss'] = f"{metrics['val_losses'][-1]:.8f}"
                    
                    pbar.set_postfix(postfix_dict)
                    pbar.update()
                
                # Validation loop
                val_loss = None
                if val_loader is not None:
                    self.encoder_model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch in val_loader:
                            y_batch, y_target_batch, mask_batch = batch
                            
                            nonzero = mask_batch.all(dim=0)
                            y_batch = y_batch[:, nonzero]
                            y_target_batch = y_target_batch[:, nonzero]
                            mask_batch = mask_batch[:, nonzero]
                            if loss_weights is not None:
                                loss_weights = loss_weights0[:, nonzero]
                                
                            outputs = self.encoder_model(y_batch)
                            
                            loss = criterion(outputs, y_target_batch)  # Per-element loss
                            if loss_weights is not None:
                                loss = loss * loss_weights
                            loss = (loss * mask_batch).sum() / mask_batch.sum()  # Compute mean over valid values only

                            val_loss += loss.item()
                    val_loss /= len(val_loader)                
                    metrics['val_losses'].append(val_loss)  # Store validation loss

                # End of epoch processing
                avg_epoch_loss = epoch_loss / len(train_loader)
                metrics['epoch_losses'].append(avg_epoch_loss)  # Store epoch loss
                
                if avg_epoch_loss < metrics['best_loss']:
                    metrics['best_loss'] = avg_epoch_loss
                
                # Early stopping logic
                if val_loss is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = self.encoder_model.state_dict().copy()
                        if self.verbose:
                            print(f"\nNew best validation loss: {val_loss:.8f}")
                    else:
                        patience_counter += 1
                        if self.verbose and patience_counter % 10 == 0:
                            print(f"\nNo improvement for {patience_counter} epochs")
                    
                    # Early stopping check
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch+1}! No improvement for {early_stopping_patience} epochs.")
                        print(f"Best validation loss: {best_val_loss:.8f}")
                        # Restore best model
                        if best_model_state is not None:
                            self.encoder_model.load_state_dict(best_model_state)
                        break
                
                # Final update for this epoch with validation loss
                postfix_dict = {
                    'epoch_loss': f"{avg_epoch_loss:.8f}", 
                    'best_loss': f"{metrics['best_loss']:.8f}"
                }
                
                if val_loss is not None:
                    postfix_dict['val_loss'] = f"{val_loss:.8f}"
                    postfix_dict['patience'] = f"{patience_counter}"
                
                pbar.set_postfix(postfix_dict)
                
            # Save the trained model
            if save_path is not None and (epoch+1) % 50 == 0:
                temp_path = save_path.replace('.pth', f'_checkpoint.pth')
                torch.save(self.encoder_model.state_dict(), temp_path)
                print(f"Model saved to {temp_path}")
            
            scheduler.step(avg_epoch_loss)
    
        # Save the trained model
        if save_path is not None:
            # Save the best model if we have it, otherwise save current
            if best_model_state is not None:
                torch.save(best_model_state, save_path)
                print(f"Best model saved to {save_path} (val_loss: {best_val_loss:.8f})")
            else:
                torch.save(self.encoder_model.state_dict(), save_path)
                print(f"Final model saved to {save_path}")
        
        return metrics
    
    def _augment_xas_data(self, y_batch):
        """XAS-specific data augmentation."""
        augmented = y_batch.clone()
                
        # Random noise
        noise_level = 0.01 * torch.rand(y_batch.shape[0], 1, device=y_batch.device)
        noise = torch.randn_like(y_batch) * noise_level
        augmented = augmented + noise
        
        # Random baseline shift
        baseline_shift = 0.02 * (torch.rand(y_batch.shape[0], 1, device=y_batch.device) - 0.5)
        augmented = augmented + baseline_shift
        
        # Random scaling (intensity variations)
        scale = 1.0 + 0.05 * (torch.rand(y_batch.shape[0], 1, device=y_batch.device) - 0.5)
        augmented = augmented * scale
        
        return augmented
    
    def _get_edge_crop(self, num_layers, kernel_size):
        rf = 1 + (kernel_size - 1) * num_layers
        crop = (rf - 1) // 2
        return crop

    def _get_random_aug_params(self, batch_shape, device=None):
        """Return random parameters for batch augmentation (scaling, baseline shift, crop)."""
        B, L = batch_shape
        device = device or 'cpu'
        # Scaling factor: e.g., [0.95, 1.05]
        scale = 0.95 + 0.1 * torch.rand(B, 1, device=device)
        # Baseline shift: e.g., [-0.02, 0.02]
        baseline = 0.04 * (torch.rand(B, 1, device=device) - 0.5)
        # Random crop: pick a start index so that crop fits in length L
        crop_size = int(0.9 * L)  # e.g., keep 90% of the spectrum
        crop_starts = torch.randint(0, L - crop_size + 1, (B,), device=device)
        return dict(scale=scale, baseline=baseline, crop_starts=crop_starts, crop_size=crop_size)

    def _augment_xas_data_noise2noise(self, batch, aug_params):
        """Apply identical augmentation (scaling, baseline, crop) to batch."""
        out = batch * aug_params['scale'] + aug_params['baseline']
        # Apply random crop
        crop_starts = aug_params['crop_starts']
        crop_size = aug_params['crop_size']
        out = out[:, crop_starts[:, None]:(crop_starts[:, None] + crop_size)]
        return out

    def save_model(self, path):
        """
        Save the encoder model to a file.

        Args:
            path (str): Path to save the model file.
        """
        torch.save(self.encoder_model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """
        Load the encoder model from a saved file.

        Args:
            path (str): Path to the saved model file.
        """
        obj = torch.load(path, map_location=self.device)
        self.encoder_model.load_state_dict(obj)
        self.encoder_model.to(self.device)
        self.encoder_model.eval()
        print(f"Model loaded from {path}")
        
    def initialize_denoiser(self, **kwargs):
        """
        Initialize data which will be used for subsequent denoising methods.

        Args:
            **kwargs: Dictionary containing keys 'x' (input values), 'y' (output values), and 'x_predict' (optional).
            
        Raises:
            ValueError: If required keys 'x' or 'y' are not provided.
        """

        self.x_train0 = kwargs.get("x", None)
        self.y_train0 = kwargs.get("y", None)
        self.x_predict0 = kwargs.get("x_predict", self.x_train0)
        self.y_reference0 = kwargs.get("y_reference", None)

        if self.x_train0 is None or self.y_train0 is None:
            raise ValueError("Missing required arguments: x or y.")
        
    def denoise_with_downsampling(self, downsampling_pts=None, downsampling_method=None, smoothness=None):
        """
        Denoise the data with optional downsampling.

        Args:
            downsampling_pts (int, optional): Number of points to downsample to. Defaults to None.
            downsampling_method (str, optional): Downsampling method to use. Defaults to None.
            smoothness (np.ndarray, optional): Smoothness values for downsampling. Defaults to None.

        Returns:
            tuple: Denoised signal, error estimates, and noise estimates.
        """

        if downsampling_pts is None or downsampling_method is None:
            y_denoised, y_error, y_noise = self.denoise()
            return y_denoised, y_error, y_noise
        
        num_samples = np.min([downsampling_pts, len(self.x_train0)]).astype(int)             
        print(f'Denoising with downsampling. Using {num_samples} data points out of {len(self.x_train0)}')   
        
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
        Perform denoising using the encoder model and optionally interpolate onto a new grid.

        Returns:
            tuple: Denoised signal, error estimates, and noise estimates.
        """

        # Extract required arguments
        x = self.x_train0
        y = self.y_train0
        x_predict = self.x_predict0
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # swap x and time axes
        y = y.transpose(0, 1)
        
        self.encoder_model.to(self.device)
        
        # Perform inference using the encoder model
        self.encoder_model.eval()
        with torch.no_grad():
            # y_denoised = self.encoder_model(x, y)  
            y = self.normalize_data(y)          
            y_denoised = self.encoder_model(y)
            y_denoised = self.denormalize_data(y_denoised)        
            
            # swap back to time and x axes
            y_denoised = y_denoised.transpose(1, 0)
            y_denoised = y_denoised.cpu().numpy()
                
        
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
            
        
        y_error = np.zeros_like(y_denoised)
        y_noise = np.zeros_like(y_denoised)

        return y_denoised, y_error, y_noise

if TORCH_AVAILABLE:
    class DenoisingAutoencoder(nn.Module):
        def __init__(self, signal_length, dropout_rate=0):
            super().__init__()
            self.layer_dim = 64
            
            # Encoder with dropout
            self.encoder = nn.Sequential(
                nn.Linear(signal_length, self.layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(self.layer_dim, self.layer_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(self.layer_dim // 2, self.layer_dim // 4)
            )
            
            # Decoder with dropout
            self.decoder = nn.Sequential(
                nn.Linear(self.layer_dim // 4, self.layer_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(self.layer_dim // 2, self.layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(self.layer_dim, signal_length)
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    class ConvDenoisingAutoencoder(nn.Module):
        def __init__(self, num_layers=4, kernel_size=7, channels=None, dropout_rate=0):
            super().__init__()
            if channels is None:
                # Channels: 16, 32, 64, ..., 16 * 2**(num_layers-1)
                channels = [16 * (2 ** i) for i in range(num_layers)]
            else:
                num_layers = len(channels)
                
            self.num_layers = num_layers
            self.kernel_size = kernel_size
            self.dropout_rate = dropout_rate

            # Encoder with dropout
            encoder_layers = []
            in_c = 1
            for i, out_c in enumerate(channels):
                encoder_layers.append(
                    nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2, bias=False, padding_mode='reflect')
                )
                encoder_layers.append(nn.ReLU())
                # Add dropout after ReLU, but not after the last layer of encoder
                if i < len(channels) - 1 and dropout_rate > 0:
                    encoder_layers.append(nn.Dropout1d(dropout_rate))
                in_c = out_c
            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder with dropout
            decoder_layers = []
            rev_channels = list(reversed(channels))
            for i in range(len(rev_channels) - 1):
                decoder_layers.append(
                    nn.ConvTranspose1d(rev_channels[i], rev_channels[i+1], kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
                )
                decoder_layers.append(nn.ReLU())
                # Add dropout after ReLU in decoder layers
                if dropout_rate > 0:
                    decoder_layers.append(nn.Dropout1d(dropout_rate))
            decoder_layers.append(
                nn.ConvTranspose1d(rev_channels[-1], 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            )
            self.decoder = nn.Sequential(*decoder_layers)

        def _get_edge_crop(self):
            # for 4 layers and kernel size 9 we will get
            # crop = 1 + 8 * 4 = 33 -> crop = (33 - 1) // 2 = 16
            rf = 1 + (self.kernel_size - 1) * self.num_layers
            return (rf - 1) // 2

        def forward(self, x):
            x = x.unsqueeze(1)
            crop = self._get_edge_crop()
            # Pad to avoid edge artefacts
            x = nn.functional.pad(x, (crop, crop), mode='reflect')
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            # Crop the padded values
            if crop > 0:
                decoded = decoded[..., crop:-crop]
            return decoded.squeeze(1)