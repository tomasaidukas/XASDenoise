"""
XAS Denoising Pipeline

Modular components:
- DenoisingPipeline: Main denoising pipeline
- PipelineConfig: Configuration of the pipeline
- DataPreprocessor: Baseline removal and noise estimation

- DataWarper: Stationarity warping (input warping, interpolation, smoothness estimation)
    - SmoothnessEstimator: Signal smoothness estimation
    - InputWarper: Input coordinate warping (k-space, smoothness)
    - DataInterpolator: Data interpolation onto a uniform grid

- DenoisingProcessor: Perform denoising of the processed data
- DataPostprocessor: Undo preprocessing steps
"""

from .config import PipelineConfig
from .preprocessing import DataPreprocessor
from .stationarity_warping import DataWarper
from .smoothness_estimation import SmoothnessEstimator
from .input_warping import InputWarper
from .interpolation import DataInterpolator
from .denoising import DenoisingProcessor
from .postprocessing import DataPostprocessor
from .pipeline import DenoisingPipeline

__all__ = [
    'PipelineConfig',
    'DataPreprocessor', 
    'DataWarper',
    'SmoothnessEstimator',
    'InputWarper',
    'DataInterpolator',
    'DenoisingProcessor',
    'DataPostprocessor',
    'DenoisingPipeline',
]
