# src/aerislab/utils/validation.py
import numpy as np
from typing import Union, List

def check_positive_float(value: float, name: str) -> None:
    """Ensures a physical quantity (mass, area, time) is positive."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def check_vector_3d(vector: Union[List, np.ndarray], name: str) -> np.ndarray:
    """Ensures input is a valid 3D numpy array."""
    vec = np.array(vector, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector (shape (3,)), got shape {vec.shape}")
    return vec

def check_normalized_quaternion(quat: np.ndarray, tolerance: float = 1e-4) -> None:
    """Ensures a quaternion is valid and normalized."""
    if quat.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {quat.shape}")
    
    norm = np.linalg.norm(quat)
    if not np.isclose(norm, 1.0, atol=tolerance):
        raise ValueError(f"Quaternion is not normalized (norm={norm:.6f}). Simulation results will be invalid.")