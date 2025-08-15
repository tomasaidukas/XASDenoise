"""
xasdenoise denoiser classes.

This module re-exports all denoiser classes from their individual files
to maintain backward compatibility.
"""

# Import all denoiser classes from their new locations
from xasdenoise.denoising_methods.regular_denoiser import RegularDenoiser
from xasdenoise.denoising_methods.encoder_denoiser import EncoderDenoiser
from xasdenoise.denoising_methods.gaussian_process_denoiser import GPDenoiser

# Re-export all classes to maintain compatibility
__all__ = [
    'RegularDenoiser',
    'EncoderDenoiser',
    'GPDenoiser'
]