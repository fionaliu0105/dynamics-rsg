"""Shared preprocessing: the SAME steps for model and neural data before comparison.

Make it structurally hard for RSA/iDSA to receive unstandardized input (AGENTS.md,
"Identical preprocessing"). Differences in dimensionality, activation scale, or time
base otherwise masquerade as findings.
"""

from src.preprocess.pipeline import Preprocessor, PreprocessConfig

__all__ = ["Preprocessor", "PreprocessConfig"]
