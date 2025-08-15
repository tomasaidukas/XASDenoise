"""
Data Interpolation Module

This module handles interpolation of data onto uniform grids for denoisers
that require uniformly sampled data.
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from typing import Optional, Tuple, Union


class DataInterpolator:
    """
    Handles interpolation of data onto uniform grids.
    
    This class provides various interpolation strategies (downsampling, upsampling, 
    same resolution) to transform warped non-uniform data onto uniform grids
    required by many denoising algorithms.
    """
    
    def __init__(self, method: Optional[str] = None, 
                 interpolation_kind: str = 'linear',
                 num_points: Optional[int] = None,
                 verbose: int = 0):
        """
        Initialize the data interpolator.
        
        Args:
            method: Interpolation method ('downsample', 'upsample', 'same', 'optimized', None)
            interpolation_kind: Type of interpolation ('linear', 'cubic', 'quadratic')
            num_points: Override number of points for interpolation
            verbose: Verbosity level (0=quiet, 1=basic info, 2=detailed with plots)
        """
        self.method = method
        self.interpolation_kind = interpolation_kind
        self.num_points = num_points
        self.verbose = verbose
        
        # State variables
        self.x_input = None
        self.x_uniform = None
        self.window_width = None
        self.initialized = False
        
    def initialize(self, x_warped: np.ndarray) -> 'DataInterpolator':
        """
        Initialize interpolation grid based on input warped coordinates.
        
        Args:
            x_warped: Warped coordinates to interpolate from
            
        Returns:
            Self for method chaining
        """
        self.x_input = x_warped.copy()
        
        if self.method is None:
            # No interpolation - keep original grid
            self.x_uniform = x_warped.copy()
            target_points = len(x_warped)
            
        elif self.method.lower() == 'downsample':
            # Downsample based on maximum meaningful spacing (excluding gaps)
            max_dist = self._get_max_meaningful_distance(x_warped)
            self.window_width = max_dist
            target_points = int((x_warped.max() - x_warped.min()) / max_dist) if self.num_points is None else self.num_points
            self.x_uniform = np.linspace(x_warped.min(), x_warped.max(), target_points)
            
        elif self.method.lower() == 'upsample':
            # Upsample based on minimum meaningful spacing
            min_dist = self._get_min_meaningful_distance(x_warped)
            self.window_width = min_dist
            target_points = int((x_warped.max() - x_warped.min()) / min_dist) if self.num_points is None else self.num_points
            self.x_uniform = np.linspace(x_warped.min(), x_warped.max(), target_points)
            
        elif self.method.lower() == 'same':
            # Same number of points as input
            target_points = len(x_warped) if self.num_points is None else self.num_points
            self.window_width = 1 / target_points
            self.x_uniform = np.linspace(x_warped.min(), x_warped.max(), target_points)
            
        elif self.method.lower() == 'optimized':
            # Optimized grid based on data distribution
            self.x_uniform = self._create_optimized_grid(x_warped)
            target_points = len(self.x_uniform)
            self.window_width = 1 / target_points
            
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
        
        self.initialized = True
        
        if self.verbose > 0:
            print(f"Interpolation initialized using '{self.method}' method")
            print(f"Input grid points: {len(x_warped)}")
            print(f"Output grid points: {len(self.x_uniform)}")
            print(f"Grid range: [{self.x_uniform.min():.4f}, {self.x_uniform.max():.4f}]")
            
        return self
    
    def interpolate(self, y: np.ndarray) -> np.ndarray:
        """
        Interpolate data onto the uniform grid.
        
        Args:
            y: Data to interpolate (1D or 2D)
            
        Returns:
            Interpolated data on uniform grid
        """
        if not self.initialized:
            raise ValueError("Must call initialize() before interpolating")
            
        if y is None:
            return None
            
        if self.method is None:
            # No interpolation needed
            return y.copy()
        
        # Use binned interpolation for better performance and accuracy
        y_interpolated = self._interpolate_with_bins(self.x_input, y, self.x_uniform)
        
        if self.verbose >= 2:
            self._visualize_interpolation(self.x_input, y, self.x_uniform, y_interpolated)
            
        return y_interpolated
    
    def uninterpolate(self, y_uniform: np.ndarray) -> np.ndarray:
        """
        Uninterpolate data from uniform grid back to original grid.
        
        Args:
            y_uniform: Data on uniform grid
            
        Returns:
            Data interpolated back to original grid
        """
        if not self.initialized:
            raise ValueError("Must call initialize() before uninterpolating")
            
        if y_uniform is None:
            return None
            
        if self.method is None:
            # No interpolation was applied
            return y_uniform.copy()
        
        # Use standard interpolation for unwarping (simpler case)
        y_original = self._interpolate_standard(self.x_uniform, y_uniform, self.x_input)
        
        if self.verbose >= 2:
            self._visualize_interpolation(self.x_uniform, y_uniform, self.x_input, y_original,
                                        title="Uninterpolation Results")
            
        return y_original
    
    def get_uniform_grid(self) -> np.ndarray:
        """Get the uniform interpolation grid."""
        if not self.initialized:
            raise ValueError("Must call initialize() before getting uniform grid")
        return self.x_uniform.copy()
    
    def _get_max_meaningful_distance(self, x: np.ndarray, gap_threshold_factor: float = 3.0) -> float:
        """
        Get the maximum meaningful distance between consecutive points, excluding large gaps.
        
        Args:
            x: Input coordinates
            gap_threshold_factor: Factor to identify gaps (distances larger than this factor 
                                 times the median distance are considered gaps)
        
        Returns:
            Maximum meaningful distance
        """
        distances = np.abs(np.diff(x))
        
        # Calculate median distance as baseline
        median_dist = np.median(distances)
        
        # Identify potential gaps
        gap_threshold = gap_threshold_factor * median_dist
        meaningful_distances = distances[distances <= gap_threshold]
        
        if len(meaningful_distances) == 0:
            # Fallback to using all distances if filtering is too aggressive
            meaningful_distances = distances
            
        # Use 95th percentile of meaningful distances to avoid outliers
        max_meaningful_dist = np.percentile(meaningful_distances, 95)
        
        # if self.verbose > 1:
        #     print(f"Median distance: {median_dist:.6f}")
        #     print(f"Gap threshold: {gap_threshold:.6f}")
        #     print(f"Meaningful distances: {len(meaningful_distances)}/{len(distances)}")
        #     print(f"Max meaningful distance: {max_meaningful_dist:.6f}")
        
        return max_meaningful_dist
    
    def _get_min_meaningful_distance(self, x: np.ndarray, outlier_threshold_factor: float = 0.1) -> float:
        """
        Get the minimum meaningful distance between consecutive points, excluding very small outliers.
        
        Args:
            x: Input coordinates
            outlier_threshold_factor: Factor to identify outliers (distances smaller than this factor 
                                    times the median distance are considered outliers)
        
        Returns:
            Minimum meaningful distance
        """
        distances = np.abs(np.diff(x))
        distances = distances[distances > 0]  # Remove zero distances
        
        if len(distances) == 0:
            return (x.max() - x.min()) / len(x)
        
        # Calculate median distance as baseline
        median_dist = np.median(distances)
        
        # Identify potential outliers (very small distances)
        outlier_threshold = outlier_threshold_factor * median_dist
        meaningful_distances = distances[distances >= outlier_threshold]
        
        if len(meaningful_distances) == 0:
            # Fallback to using all distances if filtering is too aggressive
            meaningful_distances = distances
            
        # Use 5th percentile of meaningful distances
        min_meaningful_dist = np.percentile(meaningful_distances, 5)
        
        # if self.verbose > 1:
        #     print(f"Median distance: {median_dist:.6f}")
        #     print(f"Outlier threshold: {outlier_threshold:.6f}")
        #     print(f"Meaningful distances: {len(meaningful_distances)}/{len(distances)}")
        #     print(f"Min meaningful distance: {min_meaningful_dist:.6f}")
        
        return min_meaningful_dist
    
    def _create_optimized_grid(self, x_warped: np.ndarray, num_shifts: int = 10) -> np.ndarray:
        """
        Create an optimized uniform grid based on data distribution.
        
        Args:
            x_warped: Warped coordinates
            num_shifts: Number of grid shifts to test
            
        Returns:
            Optimized uniform grid
        """
        # Generate step size candidates
        step_min = np.quantile(np.abs(np.diff(x_warped)), 0.8)
        step_max = np.quantile(np.abs(np.diff(x_warped)), 0.99)
        step_candidates = np.linspace(step_min, step_max, 20)
        
        best_grid = None
        best_score = 0
        x_min, x_max = x_warped.min(), x_warped.max()
        
        # Build KD-Tree for efficiency
        tree = cKDTree(x_warped[:, None])
        
        for step in step_candidates:
            shift_range = np.linspace(0, step, num_shifts)
            N_pts = max(1, int((x_max - x_min) / step))
            
            for shift in shift_range:
                # Create grid with current step and shift
                grid = np.linspace(x_min, x_max, N_pts) + shift
                
                # Calculate match score
                distances, _ = tree.query(grid[:, None])
                match_score = np.sum(distances) / len(grid)
                
                if match_score > best_score:
                    best_score = match_score
                    best_grid = grid
        
        if self.verbose > 0:
            print(f"Optimized grid created with {len(best_grid)} points")
            
        return best_grid
    
    def _interpolate_with_bins(self, x: np.ndarray, y: np.ndarray, 
                              x_uniform: np.ndarray) -> np.ndarray:
        """
        Accelerated interpolation using binning with Numba optimization.
        
        Args:
            x: Input coordinates
            y: Input data
            x_uniform: Target uniform coordinates
            
        Returns:
            Interpolated data
        """
        dtype = y.dtype
        is_boolean = (dtype == np.bool_)
        
        # Create bin edges
        df = np.diff(x_uniform).max() if len(x_uniform) > 1 else 1.0
        x_uniform_bins = np.concatenate([
            [x_uniform[0] - df], 
            (x_uniform[1:] + x_uniform[:-1]) / 2, 
            [x_uniform[-1] + df]
        ])
        
        # Digitize the data
        digitized = np.digitize(x, x_uniform_bins)
        num_bins = len(x_uniform) - 1
        
        # Use specialized functions for different data types
        if is_boolean:
            if y.ndim > 1:
                x_avg, y_avg = _fast_boolean_bin_2d(
                    np.asarray(x, dtype=np.float32), 
                    y, digitized, num_bins, y.shape[1]
                )
            else:
                x_avg, y_avg = _fast_boolean_bin_1d(
                    np.asarray(x, dtype=np.float32), 
                    y, digitized, num_bins
                )
            return self._interpolate_boolean_data(x_avg, y_avg, x_uniform, dtype)
        else:
            if y.ndim > 1:
                y_float32 = np.asarray(y, dtype=np.float32)
                x_avg, y_avg = _fast_bin_average_2d(
                    np.asarray(x, dtype=np.float32), 
                    y_float32, digitized, num_bins, y.shape[1]
                )
            else:
                y_float32 = np.asarray(y, dtype=np.float32)
                x_avg, y_avg = _fast_bin_average_1d(
                    np.asarray(x, dtype=np.float32), 
                    y_float32, digitized, num_bins
                )
            return self._interpolate_numeric_data(x_avg, y_avg, x_uniform, dtype)
    
    def _interpolate_boolean_data(self, x_avg: np.ndarray, y_avg: np.ndarray,
                                 x_uniform: np.ndarray, dtype) -> np.ndarray:
        """Handle boolean data interpolation with thresholding."""
        if y_avg.ndim > 1:
            y_interp = np.empty((len(x_uniform), y_avg.shape[1]), dtype=dtype)
            for j in range(y_avg.shape[1]):
                probs = np.interp(x_uniform, x_avg, y_avg[:, j],
                                left=y_avg[0, j], right=y_avg[-1, j])
                y_interp[:, j] = probs >= 0.5
        else:
            probs = np.interp(x_uniform, x_avg, y_avg,
                            left=y_avg[0], right=y_avg[-1])
            y_interp = probs >= 0.5
        return y_interp
    
    def _interpolate_numeric_data(self, x_avg: np.ndarray, y_avg: np.ndarray,
                                 x_uniform: np.ndarray, dtype) -> np.ndarray:
        """Handle numeric data interpolation."""
        # For small datasets, use numpy's fast linear interpolation
        if len(x_avg) < 100 or self.interpolation_kind == 'linear':
            if y_avg.ndim > 1:
                y_interp = np.empty((len(x_uniform), y_avg.shape[1]), dtype=dtype)
                for j in range(y_avg.shape[1]):
                    y_interp[:, j] = np.interp(x_uniform, x_avg, y_avg[:, j],
                                             left=y_avg[0, j], right=y_avg[-1, j])
            else:
                y_interp = np.interp(x_uniform, x_avg, y_avg,
                                   left=y_avg[0], right=y_avg[-1])
        else:
            # Use scipy for higher-order interpolation
            y_interp = self._interpolate_standard(x_avg, y_avg, x_uniform)
        
        return y_interp.astype(dtype)
    
    def _interpolate_standard(self, x: np.ndarray, y: np.ndarray, 
                             x_new: np.ndarray) -> np.ndarray:
        """Standard interpolation using scipy."""
        f = interp1d(x, y, axis=0, kind=self.interpolation_kind,
                    fill_value=self._compute_bounds(y), bounds_error=False)
        return f(x_new)
    
    def _compute_bounds(self, y: np.ndarray) -> Union[float, Tuple[float, float]]:
        """Compute boundary values for extrapolation."""
        if len(y.shape) > 1 and y.shape[1] > 1:
            return (np.median(y[:5, :]), np.median(y[-5:, :]))
        else:
            return (np.median(y[:5]), np.median(y[-5:]))
    
    def _visualize_interpolation(self, x1: np.ndarray, y1: np.ndarray,
                                x2: np.ndarray, y2: np.ndarray,
                                title: str = "Interpolation Results"):
        """Visualize interpolation results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original data
        try:
            ax1.plot(x1, y1[:, 0] if y1.ndim > 1 else y1, 'b.', markersize=1, label='Original data')
        except:
            ax1.plot(x1, y1, 'b.', markersize=1, label='Original data')
        ax1.set_title('Before Interpolation')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        
        # Plot interpolated data
        try:
            ax2.plot(x2, y2[:, 0] if y2.ndim > 1 else y2, 'r.', markersize=1, label='Interpolated data')
        except:
            ax2.plot(x2, y2, 'r.', markersize=1, label='Interpolated data')
        ax2.set_title('After Interpolation')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Numba-optimized functions for fast binning
@numba.njit(parallel=True)
def _fast_bin_average_1d(x_warped, y, digitized, num_bins):
    """Fast binned averaging for 1D data using Numba."""
    bin_counts = np.zeros(num_bins, dtype=np.int32)
    x_sums = np.zeros(num_bins, dtype=np.float32)
    y_sums = np.zeros(num_bins, dtype=np.float32)
    
    for i in range(len(x_warped)):
        bin_idx = digitized[i] - 1
        if 0 <= bin_idx < num_bins:
            bin_counts[bin_idx] += 1
            x_sums[bin_idx] += x_warped[i]
            y_sums[bin_idx] += y[i]
    
    valid_bins = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            valid_bins += 1
    
    x_avg = np.zeros(valid_bins, dtype=np.float32)
    y_avg = np.zeros(valid_bins, dtype=np.float32)
    
    valid_idx = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            x_avg[valid_idx] = x_sums[i] / bin_counts[i]
            y_avg[valid_idx] = y_sums[i] / bin_counts[i]
            valid_idx += 1
    
    return x_avg, y_avg


@numba.njit(parallel=True)
def _fast_bin_average_2d(x_warped, y, digitized, num_bins, num_cols):
    """Fast binned averaging for 2D data using Numba."""
    bin_counts = np.zeros(num_bins, dtype=np.int32)
    x_sums = np.zeros(num_bins, dtype=np.float32)
    y_sums = np.zeros((num_bins, num_cols), dtype=np.float32)
    
    for i in range(len(x_warped)):
        bin_idx = digitized[i] - 1
        if 0 <= bin_idx < num_bins:
            bin_counts[bin_idx] += 1
            x_sums[bin_idx] += x_warped[i]
            for j in range(num_cols):
                y_sums[bin_idx, j] += y[i, j]
    
    valid_bins = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            valid_bins += 1
    
    x_avg = np.zeros(valid_bins, dtype=np.float32)
    y_avg = np.zeros((valid_bins, num_cols), dtype=np.float32)
    
    valid_idx = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            x_avg[valid_idx] = x_sums[i] / bin_counts[i]
            for j in range(num_cols):
                y_avg[valid_idx, j] = y_sums[i, j] / bin_counts[i]
            valid_idx += 1
    
    return x_avg, y_avg


@numba.njit(parallel=True)
def _fast_boolean_bin_1d(x_warped, y, digitized, num_bins):
    """Fast binned processing for 1D boolean data using Numba."""
    bin_counts = np.zeros(num_bins, dtype=np.int32)
    x_sums = np.zeros(num_bins, dtype=np.float32)
    true_counts = np.zeros(num_bins, dtype=np.int32)
    
    for i in range(len(x_warped)):
        bin_idx = digitized[i] - 1
        if 0 <= bin_idx < num_bins:
            bin_counts[bin_idx] += 1
            x_sums[bin_idx] += x_warped[i]
            if y[i]:
                true_counts[bin_idx] += 1
    
    valid_bins = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            valid_bins += 1
    
    x_avg = np.zeros(valid_bins, dtype=np.float32)
    y_avg = np.zeros(valid_bins, dtype=np.float32)
    
    valid_idx = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            x_avg[valid_idx] = x_sums[i] / bin_counts[i]
            y_avg[valid_idx] = true_counts[i] / bin_counts[i]
            valid_idx += 1
    
    return x_avg, y_avg


@numba.njit(parallel=True)
def _fast_boolean_bin_2d(x_warped, y, digitized, num_bins, num_cols):
    """Fast binned processing for 2D boolean data using Numba."""
    bin_counts = np.zeros(num_bins, dtype=np.int32)
    x_sums = np.zeros(num_bins, dtype=np.float32)
    true_counts = np.zeros((num_bins, num_cols), dtype=np.int32)
    
    for i in range(len(x_warped)):
        bin_idx = digitized[i] - 1
        if 0 <= bin_idx < num_bins:
            bin_counts[bin_idx] += 1
            x_sums[bin_idx] += x_warped[i]
            for j in range(num_cols):
                if y[i, j]:
                    true_counts[bin_idx, j] += 1
    
    valid_bins = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            valid_bins += 1
    
    x_avg = np.zeros(valid_bins, dtype=np.float32)
    y_avg = np.zeros((valid_bins, num_cols), dtype=np.float32)
    
    valid_idx = 0
    for i in range(num_bins):
        if bin_counts[i] > 0:
            x_avg[valid_idx] = x_sums[i] / bin_counts[i]
            for j in range(num_cols):
                y_avg[valid_idx, j] = true_counts[i, j] / bin_counts[i]
            valid_idx += 1
    
    return x_avg, y_avg
