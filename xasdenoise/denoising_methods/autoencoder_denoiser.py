"""
Autoencoder denoiser class.
"""

import numpy as np
import gc
from scipy import interpolate
from xasdenoise.denoising_methods.denoising_utils import downsample_data
from tqdm import tqdm
import pickle

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
    


# Make an autoencoder denoising class which takes a loaded autoencoder model and denoises the data
class AutoencoderDenoiser:
    """
    A class to perform signal denoising using an autoencoder model.
    
    Note: Requires PyTorch. If not available, raises ImportError on initialization.
    """
    def __init__(self, model_type='conv', device='auto', gpu_index=0, num_layers=4, kernel_size=7, 
                 channels=None, dropout_rate=0, normalization_method=None, bias=False,
                 output_mode='direct', output_nonnegativity=False,
                 transpose_input=False):
        """
        Initialize the AutoencoderDenoiser with a specified autoencoder architecture.

        Args:
            model_type (str): Type of autoencoder. Options: 'conv' (1D), 'conv2d_temporal' (2D), 'separable_conv1d' (separable 1D). Defaults to 'conv'.
            device (str): Device to use for training and inference. Options: 'auto', 'cpu', 'cuda', 'mps'. Defaults to 'auto'.
            gpu_index (int): GPU index to use when device is 'cuda'. Defaults to 0.
            num_layers (int): Number of layers in the autoencoder. Defaults to 4.
            kernel_size (int): Kernel size for convolutional layers (energy dimension for 2D). Defaults to 7.
            channels (list, optional): Number of channels in each layer. Defaults to None.
            dropout_rate (float): Dropout rate for regularization. Defaults to 0.
            normalization_method: None, 'zscore', 'minmax', 'l2norm'.
            bias (bool): Whether to include bias terms in layers. Defaults to False.
            output_mode (str): Output mode of the autoencoder. Options: 'direct', 'residual'. Defaults to 'direct'.
            output_nonnegativity (bool): If True and output_mode is 'direct', applies softplus to ensure non-negative outputs. Defaults to False.
            transpose_input (str): If True, transposes the input data before processing. Defaults to False.
        """
        
        # Check dependencies before initialization
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for AutoencoderDenoiser but not available. "
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
        self.normalization_method = normalization_method
        self.bias = bias
        self.output_mode = output_mode
        self.output_nonnegativity = output_nonnegativity
        self.transpose_input = transpose_input
        
        # Initialize device - auto-detect if 'auto' is specified
        if device == 'auto':
            self.device = self._get_best_device()
        else:
            self.device = device
            
        # Initialize the computation device
        self.gpu_device = self._initialize_device()
        
        if model_type == 'conv':
            self.autoencoder_model = ConvDenoisingAutoencoder(num_layers=self.num_layers, 
                                                          kernel_size=self.kernel_size, 
                                                          channels=self.channels, 
                                                          dropout_rate=self.dropout_rate,
                                                          bias=self.bias,
                                                          output_mode=self.output_mode,
                                                          output_nonnegativity=self.output_nonnegativity
                                                          ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: 'conv'.")

        self.autoencoder_model.train()  # Set the model to training mode
      
    def _get_best_device(self):
        """
        Automatically detect and return the best available device.

        Returns:
            str: Best available device ('cuda', 'mps', or 'cpu').
        """
         # Set CUDA device BEFORE any torch.cuda calls to prevent GPU 0 initialization
        if TORCH_AVAILABLE and self.gpu_index is not None and self.gpu_index >= 0:
            try:
                torch.cuda.set_device(self.gpu_index)
            except Exception:
                pass
            
        if TORCH_AVAILABLE and torch.cuda.is_available():
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
        
        # Set CUDA device BEFORE any torch.cuda calls to prevent GPU 0 initialization
        if TORCH_AVAILABLE and self.gpu_index is not None and self.gpu_index >= 0:
            try:
                torch.cuda.set_device(self.gpu_index)
            except Exception:
                pass
            
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
        
    def normalize_data(self, y, compute_norm=True, method='l2norm'):
        """
        Normalize the training data.

        Args:
            y (torch.Tensor): Training data.
            compute_norm (bool): Whether to compute normalization parameters. Defaults to True.
            method (str): Normalization method. Options: 'minmax', 'l2norm', 'zscore'. Defaults to 'l2norm'.

        Returns:
            torch.Tensor: Normalized training data.
        """
        if method == None:
            return y
        
        if method == 'minmax':
            if compute_norm:
                y_max = y.max(axis=1).values[:, None]
                y_min = y.min(axis=1).values[:, None]
                self.norm_params['y'] = (y_min, y_max)
            y_min, y_max = self.norm_params['y']
            return 2 * (y - y_min) / (y_max - y_min) - 1
        
        elif method == 'percentile':
            low, high = 1, 99
            if compute_norm:
                y_low = torch.quantile(y, low / 100.0, dim=1, keepdim=True)
                y_high = torch.quantile(y, high / 100.0, dim=1, keepdim=True)
                self.norm_params['y'] = (y_low, y_high)
            y_low, y_high = self.norm_params['y']
            return 2 * (y - y_low) / (y_high - y_low + 1e-8) - 1
        
        elif method == 'l2norm':
            if compute_norm:
                y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
                self.norm_params['y'] = y_norm
            y_norm = self.norm_params['y']
            return y / (y_norm + 1e-8)  # Avoid division by zero
        
        elif method == 'l2norm_global':
            if compute_norm:
                y_norm = torch.norm(y)
                self.norm_params['y'] = y_norm
            y_norm = self.norm_params['y']
            return y / (y_norm + 1e-8)  # Avoid division by zero
        
        elif method == 'zscore':
            if compute_norm:
                y_mean = y.mean(axis=1, keepdim=True)
                y_std = y.std(axis=1, keepdim=True)
                self.norm_params['y'] = (y_mean, y_std)
            y_mean, y_std = self.norm_params['y']
            return (y - y_mean) / (y_std + 1e-8)  # Avoid division by zero
        
        else:
            raise ValueError(f"Unknown normalization method: {method}. Supported: 'minmax', 'l2norm', 'zscore'.")
    
    def denormalize_data(self, y, method='l2norm'):
        """
        Denormalize the denoised data using the stored normalization parameters.

        Args:
            y (torch.Tensor): Denoised data.
            method (str): Normalization method used during training. Options: 'minmax', 'l2norm', 'zscore'. Defaults to 'l2norm'.

        Returns:
            torch.Tensor: Denormalized denoised data.
        """
        if method == None:
            return y
        
        if method == 'minmax':
            y_min, y_max = self.norm_params['y']
            return (y + 1) * (y_max - y_min) / 2 + y_min
        elif method == 'percentile':
            y_low, y_high = self.norm_params['y']
            return (y + 1) * (y_high - y_low) / 2 + y_low
        elif method == 'l2norm':
            y_norm = self.norm_params['y']
            return y * (y_norm + 1e-8)  # Avoid division by zero
        elif method == 'l2norm_global':
            y_norm = self.norm_params['y']
            return y * (y_norm + 1e-8)  # Avoid division by zero
        elif method == 'zscore':
            y_mean, y_std = self.norm_params['y']
            return y * (y_std + 1e-8) + y_mean  # Avoid division by zero
        else:
            raise ValueError(f"Unknown normalization method: {method}. Supported: 'minmax', 'l2norm', 'zscore'.")
    
    def train_model(self, y_train, y_target, mask_train=None, y_val=None, y_val_target=None, mask_val=None,
            batch_size=32, num_epochs=1000, learning_rate=1e-4, save_path=None, augment_data=False,
            remove_padded_regions=True, randomized_masking=False, loss_weights=None,
            early_stopping_patience=50, weight_decay=1e-5, dont_shuffle=False):
        """
        Train the autoencoder model using the given training data with optional masking.

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
            augment_data (bool): Whether to apply data augmentation. Defaults to False.
            remove_padded_regions (bool): Whether to remove padded regions using the mask. Defaults to True.
            randomized_masking (bool): Whether to randomly mask data during training. Defaults to False.
            loss_weights (torch.Tensor, optional): Weights for each data point in the loss function. Defaults to None.
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping. Defaults to 50.
            weight_decay (float): L2 regularization strength. Defaults to 1e-5.
            dont_shuffle (bool): Whether to NOT shuffle the data within each batch. Default is False.
        """
        
        # Prepare data and loaders
        train_loader, val_loader, loss_weights_dict = self._prepare_training_data(
            y_train, y_target, mask_train, y_val, y_val_target, mask_val,
            batch_size, loss_weights, dont_shuffle
        )
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(learning_rate, weight_decay)
        
        # Initialize training state
        training_state = self._initialize_training_state(early_stopping_patience)
        
        # Train the model
        metrics = self._train_model(
            train_loader, val_loader, optimizer, scheduler, training_state,
            num_epochs, augment_data, remove_padded_regions,
            randomized_masking, loss_weights_dict, save_path
        )
        
        # Save final models
        self._save_final_models(save_path, training_state, metrics)
        
        return metrics
    
    def _prepare_training_data(self, y_train, y_target, mask_train, 
                                y_val, y_val_target, mask_val, batch_size, loss_weights,
                                dont_shuffle):
        """Prepare and normalize training data, create data loaders."""
        
        # Convert to tensors and initialize arrays
        y_train, y_target, mask_train, y_val, y_val_target, mask_val = \
            self.to_tensor(y_train), self.to_tensor(y_target), self.to_tensor(mask_train), \
            self.to_tensor(y_val), self.to_tensor(y_val_target), self.to_tensor(mask_val)
            
        if mask_train is None:
            mask_train = torch.ones_like(y_train)
        if y_val is not None and mask_val is None:
            mask_val = torch.ones_like(y_val)
        if loss_weights is not None:
            loss_weights = self.to_tensor(loss_weights)
            loss_weights0 = loss_weights.clone()
            
        # Normalize data
        if y_val is not None:
            y_val = self.normalize_data(y_val, method=self.normalization_method)
            y_val_target = self.normalize_data(y_val_target, compute_norm=False, method=self.normalization_method)

        y_train = self.normalize_data(y_train, method=self.normalization_method)
        y_target = self.normalize_data(y_target, compute_norm=False, method=self.normalization_method)
        
        if self.transpose_input:
            y_train = y_train.T
            y_target = y_target.T
            mask_train = mask_train.T
            if y_val is not None:
                y_val = y_val.T
                y_val_target = y_val_target.T
                mask_val = mask_val.T
            if loss_weights is not None:
                loss_weights = loss_weights.T
                loss_weights0 = loss_weights0.T
                
        # Prepare datasets
        shuffle = not dont_shuffle

        train_dataset = TensorDataset(y_train, y_target, mask_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        val_loader = None
        if y_val is not None and y_val_target is not None:
            val_dataset = TensorDataset(y_val, y_val_target, mask_val) if mask_val is not None else TensorDataset(y_val, y_val_target)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        loss_weights_dict = {
            'loss_weights': loss_weights if loss_weights is not None else None,
            'loss_weights0': loss_weights0 if loss_weights is not None else None,
        }
        
        return train_loader, val_loader, loss_weights_dict
    
    def _setup_optimizer_and_scheduler(self, learning_rate, weight_decay):
        """Setup optimizer and learning rate scheduler."""
        optimizer = optim.Adam(self.autoencoder_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return optimizer, scheduler
    
    def _initialize_training_state(self, early_stopping_patience):
        """Initialize training state variables."""
        return {
            'metrics': {
                'loss': 0.0,
                'running_loss': [],
                'best_loss': float('inf'),
                'val_loss': [],
                'epoch_losses': [],
                'val_losses': []
            },
            'best_val_loss': float('inf'),
            'patience_counter': 0,
            'best_model_state': None,
            'early_stopping_patience': early_stopping_patience
        }
    
    def _train_model(self, train_loader, val_loader, optimizer, scheduler, training_state,
                       num_epochs, augment_data, remove_padded_regions,
                       randomized_masking, loss_weights_dict, save_path):
        """Standard training loop."""
        criterion = nn.MSELoss(reduction='none')
        metrics = training_state['metrics']
        update_freq = 5
        
        for epoch in range(num_epochs):
            epoch_loss = self._run_training_epoch(
                train_loader, optimizer, criterion, metrics, epoch, num_epochs,
                augment_data, remove_padded_regions, randomized_masking,
                loss_weights_dict, update_freq
            )
            
            # Validation
            val_loss = self._validate_epoch(val_loader, criterion, loss_weights_dict)
            if val_loss is not None:
                metrics['val_losses'].append(val_loss)
            
            # Update training state
            self._update_training_state(training_state, epoch_loss, val_loss, epoch, num_epochs)
            
            # Checkpoint saving
            if save_path and (epoch + 1) % 50 == 0:
                self._save_checkpoint(save_path)
            
            # Early stopping check
            if self._check_early_stopping(training_state, epoch):
                break
            
            scheduler.step(epoch_loss)
        
        return metrics
    
    def _run_training_epoch(self, train_loader, optimizer, criterion, metrics,
                                     epoch, num_epochs, augment_data,
                                     remove_padded_regions, randomized_masking,
                                     loss_weights_dict, update_freq):
        """Run one training epoch of training."""
        self.autoencoder_model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}",
                 miniters=update_freq, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                 position=0, leave=True) as pbar:
            
            for batch in train_loader:
                y_batch, y_target_batch, mask_batch = batch
                
                # Apply preprocessing
                y_batch, y_target_batch, mask_batch = self._preprocess_batch(
                    y_batch, y_target_batch, mask_batch, remove_padded_regions,
                    randomized_masking, loss_weights_dict
                )
                
                # Apply augmentation
                if augment_data and epoch < num_epochs * 0.8:
                    y_batch, y_target_batch = self._apply_data_augmentation(
                        y_batch, y_target_batch
                    )
                
                # Forward pass
                outputs = self.autoencoder_model(y_batch)
                
                # Compute loss
                loss = criterion(outputs, y_target_batch)
                
                if loss_weights_dict['loss_weights'] is not None:
                    loss = loss * loss_weights_dict['loss_weights']
                
                reconstruction_loss = (loss * mask_batch).sum() / mask_batch.sum()
                
                # Backward pass
                optimizer.zero_grad()
                reconstruction_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update metrics
                batch_loss = reconstruction_loss.item()
                epoch_loss += batch_loss
                self._update_progress_bar(pbar, batch_loss, metrics)
        
        return epoch_loss / len(train_loader)
    
    def _preprocess_batch(self, y_batch, y_target_batch, mask_batch,
                         remove_padded_regions, randomized_masking, loss_weights_dict):
        """Preprocess batch: remove padding, apply random masking."""
        if remove_padded_regions and mask_batch is not None:
            nonzero = mask_batch.all(dim=0)
            y_batch = y_batch[:, nonzero]
            y_target_batch = y_target_batch[:, nonzero]
            mask_batch = mask_batch[:, nonzero]
        
        if randomized_masking and mask_batch is not None:
            if torch.rand(1).item() < 0.5:
                nonzero = mask_batch.all(dim=0)
                y_batch = y_batch[:, nonzero]
                y_target_batch = y_target_batch[:, nonzero]
                mask_batch = mask_batch[:, nonzero]
        
        return y_batch, y_target_batch, mask_batch
    
    def _apply_data_augmentation(self, y_batch, y_target_batch):
        """Apply data augmentation to batch."""
        # no augmentation for now, placeholder for future methods
        return y_batch, y_target_batch
    
    def _validate_epoch(self, val_loader, criterion, loss_weights_dict):
        """Run validation and return validation loss."""
        if val_loader is None:
            return None
        
        self.autoencoder_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                y_batch, y_target_batch, mask_batch = batch
                
                nonzero = mask_batch.all(dim=0)
                y_batch = y_batch[:, nonzero]
                y_target_batch = y_target_batch[:, nonzero]
                mask_batch = mask_batch[:, nonzero]
                
                outputs = self.autoencoder_model(y_batch)
                
                # Reconstruction loss
                loss = criterion(outputs, y_target_batch)
                if loss_weights_dict['loss_weights'] is not None:
                    loss = loss * loss_weights_dict['loss_weights']
                reconstruction_loss = (loss * mask_batch).sum() / mask_batch.sum()
                
                val_loss += reconstruction_loss.item()
        
        return val_loss / len(val_loader)
    
    def _update_progress_bar(self, pbar, batch_loss, metrics):
        """Update progress bar with current metrics."""
        metrics['loss'] = batch_loss
        metrics['running_loss'].append(batch_loss)
        
        running_avg = np.mean(metrics['running_loss'][-20:])
        
        postfix_dict = {
            'loss': f"{batch_loss:.8f}",
            'avg_loss': f"{running_avg:.8f}",
            'best': f"{metrics['best_loss']:.8f}"
        }
        
        if len(metrics['val_losses']) > 0:
            postfix_dict['val_loss'] = f"{metrics['val_losses'][-1]:.8f}"
        
        pbar.set_postfix(postfix_dict)
        pbar.update()
    
    def _update_training_state(self, training_state, epoch_loss, val_loss, epoch, num_epochs):
        """Update training state after epoch."""
        metrics = training_state['metrics']
        metrics['epoch_losses'].append(epoch_loss)
        
        if epoch_loss < metrics['best_loss']:
            metrics['best_loss'] = epoch_loss
            training_state['best_model_state'] = self.autoencoder_model.state_dict().copy()
        
        if val_loss is not None:
            if val_loss < training_state['best_val_loss']:
                training_state['best_val_loss'] = val_loss
                training_state['patience_counter'] = 0
                if self.verbose:
                    print(f"\nNew best validation loss: {val_loss:.8f}")
            else:
                training_state['patience_counter'] += 1
                if self.verbose and training_state['patience_counter'] % 10 == 0:
                    print(f"\nNo improvement for {training_state['patience_counter']} epochs")
    
    def _check_early_stopping(self, training_state, epoch):
        """Check if early stopping criteria is met."""
        if training_state['patience_counter'] >= training_state['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch+1}! No improvement for {training_state['early_stopping_patience']} epochs.")
            print(f"Best validation loss: {training_state['best_val_loss']:.8f}")
            return True
        return False
    
    def _save_checkpoint(self, save_path):
        """Save training checkpoint."""
        temp_path = save_path.replace('.pth', '_checkpoint.pth')
        torch.save(self.autoencoder_model.state_dict(), temp_path)
        print(f"Model saved to {temp_path}")
    
    def _save_final_models(self, save_path, training_state, metrics):
        """Save final and best models."""
        if save_path is None:
            return
        
        # Save final model
        torch.save(self.autoencoder_model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")
        
        # Save best model
        if training_state['best_model_state'] is not None:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(training_state['best_model_state'], best_path)
            print(f"Best model saved to {best_path} (val_loss: {training_state['best_val_loss']:.8f})")
        
        # Save training metrics
        metrics_path = save_path.replace('.pth', '_training_metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Training metrics saved to {metrics_path}")
    
    def _get_edge_crop(self, num_layers, kernel_size):
        rf = 1 + (kernel_size - 1) * num_layers
        crop = (rf - 1) // 2
        return crop

    def save_model(self, path):
        """
        Save the autoencoder model to a file.

        Args:
            path (str): Path to save the model file.
        """
        torch.save(self.autoencoder_model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """
        Load the autoencoder model from a saved file.

        Args:
            path (str): Path to the saved model file.
        """
        obj = torch.load(path, map_location=self.device)
        
        new_state_dict = {}
        for key, value in obj.items():
            new_state_dict[key] = value
        
        self.autoencoder_model.load_state_dict(new_state_dict)
        self.autoencoder_model.to(self.device)
        self.autoencoder_model.eval()
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
        Perform denoising using the autoencoder model and optionally interpolate onto a new grid.

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
        
        self.autoencoder_model.to(self.device)
        
        # Perform inference using the autoencoder model
        self.autoencoder_model.eval()
        with torch.no_grad():
            # y_denoised = self.autoencoder_model(x, y)  
            y = self.normalize_data(y, method=self.normalization_method)     
            
            if self.transpose_input:     
                y_denoised = self.autoencoder_model(y.T).T
            else:
                y_denoised = self.autoencoder_model(y)
            y_denoised = self.denormalize_data(y_denoised, method=self.normalization_method)        
            
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
    class ConvDenoisingAutoencoder(nn.Module):
        def __init__(self, 
                     num_layers=4, 
                     kernel_size=7, 
                     channels=None, 
                     dropout_rate=0, 
                     bias=False, 
                     output_mode='direct',
                     output_nonnegativity=False):
            super().__init__()
            if channels is None:
                # Channels: 16, 32, 64, ..., 16 * 2**(num_layers-1)
                channels = [16 * (2 ** i) for i in range(num_layers)]
            else:
                num_layers = len(channels)
                
            self.num_layers = num_layers
            self.kernel_size = kernel_size
            self.dropout_rate = dropout_rate
            self.bias = bias
            self.output_mode = output_mode
            self.output_nonnegativity = output_nonnegativity
            
            # Encoder with dropout
            autoencoder_layers = []
            in_c = 1
            for i, out_c in enumerate(channels):
                autoencoder_layers.append(
                    nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2, bias=self.bias, padding_mode='reflect')
                )
                # autoencoder_layers.append(nn.InstanceNorm1d(out_c, affine=True))
                autoencoder_layers.append(nn.ReLU())

                if i < len(channels) - 1 and dropout_rate > 0:
                    autoencoder_layers.append(nn.Dropout1d(dropout_rate))
                in_c = out_c
            self.encoder = nn.Sequential(*autoencoder_layers)

            # Decoder with dropout
            decoder_layers = []
            rev_channels = list(reversed(channels))
            for i in range(len(rev_channels) - 1):
                decoder_layers.append(
                    nn.ConvTranspose1d(rev_channels[i], rev_channels[i+1], kernel_size=kernel_size, padding=kernel_size // 2, bias=self.bias)
                )
                # decoder_layers.append(nn.InstanceNorm1d(rev_channels[i+1], affine=True))
                decoder_layers.append(nn.ReLU())

                if dropout_rate > 0:
                    decoder_layers.append(nn.Dropout1d(dropout_rate))
            decoder_layers.append(
                nn.ConvTranspose1d(rev_channels[-1], 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=self.bias)
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
            x_padded = nn.functional.pad(x, (crop, crop), mode='reflect')

            encoded = self.encoder(x_padded)
            decoded = self.decoder(encoded)

            if crop > 0:
                decoded = decoded[..., crop:-crop]

            # Residual learning: predict noise, subtract from input
            if self.output_mode == 'residual':
                out = x - 1 * decoded # Predict noise
                return out.squeeze(1)
            elif self.output_mode == 'direct':
                if self.output_nonnegativity:
                    decoded = nn.functional.softplus(decoded, beta=0.5) # Ensure non-negativity
                return decoded.squeeze(1)