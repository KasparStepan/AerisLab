"""
Validation utilities for physical parameters and state variables.

Provides functions to validate inputs for physics simulations,
ensuring physical consistency and numerical stability.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import warnings


def validate_positive(value: float, name: str, strict: bool = True) -> None:
    """
    Validate that a scalar value is positive.
    
    Parameters
    ----------
    value : float
        Value to validate
    name : str
        Parameter name for error messages
    strict : bool
        If True, raise ValueError. If False, issue warning.
        
    Raises
    ------
    ValueError
        If strict=True and value <= 0
    """
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_quaternion(q: NDArray[np.float64], tol: float = 1e-6) -> None:
    """
    Validate that array is a unit quaternion.
    
    Parameters
    ----------
    q : NDArray[np.float64]
        Quaternion [x, y, z, w]
    tol : float
        Tolerance for unit norm check
        
    Raises
    ------
    ValueError
        If quaternion shape or norm is invalid
    """
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")
    
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > tol:
        warnings.warn(
            f"Quaternion not normalized: |q| = {norm:.6f}. "
            "Consider normalizing before use.",
            RuntimeWarning,
            stacklevel=2
        )


def validate_inertia_tensor(I: NDArray[np.float64]) -> None:
    """
    Validate inertia tensor is 3x3 and positive definite.
    
    Parameters
    ----------
    I : NDArray[np.float64]
        Inertia tensor (3, 3)
        
    Raises
    ------
    ValueError
        If shape is wrong or matrix is not positive definite
    """
    if I.shape != (3, 3):
        raise ValueError(f"Inertia tensor must be 3x3, got shape {I.shape}")
    
    # Check symmetry
    if not np.allclose(I, I.T):
        warnings.warn(
            "Inertia tensor is not symmetric. Using (I + I^T)/2.",
            RuntimeWarning,
            stacklevel=2
        )
    
    # Check positive definiteness
    eigenvalues = np.linalg.eigvals(I)
    if np.any(eigenvalues <= 0):
        raise ValueError(
            f"Inertia tensor must be positive definite. "
            f"Got eigenvalues: {eigenvalues}"
        )


def validate_timestep(dt: float, max_dt: float = 1.0) -> None:
    """
    Validate timestep is positive and reasonable.
    
    Parameters
    ----------
    dt : float
        Time step [s]
    max_dt : float
        Maximum reasonable timestep [s]
        
    Raises
    ------
    ValueError
        If timestep is invalid
    """
    if dt <= 0:
        raise ValueError(f"Timestep must be positive, got {dt}")
    if dt > max_dt:
        warnings.warn(
            f"Large timestep {dt}s may cause instability. "
            f"Consider using dt < {max_dt}s.",
            RuntimeWarning,
            stacklevel=2
        )
