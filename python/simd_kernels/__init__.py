"""Deprecated import name for :mod:`simd_ml_kernels`.

The project was renamed to simd-ml-kernels (PyPI) / ``simd_ml_kernels``
(import). ``import simd_kernels`` keeps working, returns the same module
object, and emits a DeprecationWarning. It will be removed in 0.5.0.
"""

import sys
import warnings

import simd_ml_kernels

warnings.warn(
    "The 'simd_kernels' module has been renamed to 'simd_ml_kernels'. "
    "Use 'import simd_ml_kernels'; the old name will be removed in 0.5.0.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = simd_ml_kernels
