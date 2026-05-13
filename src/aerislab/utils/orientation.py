"""
Engineer-friendly orientation utilities.

This module provides simple ways to specify body orientation without
needing to understand quaternions. All functions return quaternions
in the format expected by RigidBody6DOF: [x, y, z, w] (scalar-last).

Common Use Cases
----------------
- Point a body's axis in a direction: use `orientation_from_direction()`
- Specify roll/pitch/yaw angles: use `orientation_from_euler()`
- Rotate around an axis: use `orientation_from_axis_angle()`

Examples
--------
>>> from aerislab.utils.orientation import (
...     orientation_from_euler,
...     orientation_from_direction,
...     IDENTITY
... )

# Tip the body 45° nose-down (pitch)
>>> q = orientation_from_euler(pitch=-45)

# Point the body's Z-axis toward +Y global
>>> q = orientation_from_direction(body_axis='z', toward=[0, 1, 0])

# No rotation (identity quaternion)
>>> q = IDENTITY
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Identity quaternion (no rotation)
# =============================================================================

IDENTITY: NDArray[np.float64] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
"""Identity quaternion [0, 0, 0, 1] representing no rotation."""


# =============================================================================
# Euler Angles (Roll, Pitch, Yaw)
# =============================================================================

def orientation_from_euler(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    degrees: bool = True,
    order: str = "xyz"
) -> NDArray[np.float64]:
    """
    Create orientation quaternion from Euler angles.
    
    This is the most intuitive way for engineers to specify orientation.
    Uses aerospace convention by default (XYZ extrinsic rotations).
    
    Parameters
    ----------
    roll : float
        Rotation about X axis (bank) [degrees or radians]
    pitch : float
        Rotation about Y axis (nose up/down) [degrees or radians]
    yaw : float
        Rotation about Z axis (heading) [degrees or radians]
    degrees : bool
        If True (default), angles are in degrees. If False, radians.
    order : str
        Euler angle sequence. Default "xyz" for aerospace convention.
        Common options: "xyz", "zyx", "zxz"
        
    Returns
    -------
    NDArray[np.float64]
        Quaternion [x, y, z, w] for RigidBody6DOF.
        
    Examples
    --------
    >>> # Rotate 90 degrees about Z (yaw)
    >>> q = orientation_from_euler(yaw=90)
    
    >>> # Pitch down 30 degrees
    >>> q = orientation_from_euler(pitch=-30)
    
    >>> # Combined rotation
    >>> q = orientation_from_euler(roll=10, pitch=-5, yaw=45)
    """
    angles = np.array([roll, pitch, yaw])
    
    if degrees:
        angles = np.deg2rad(angles)
    
    rot = R.from_euler(order, angles, degrees=False)
    return rot.as_quat()  # Returns [x, y, z, w]


# =============================================================================
# Axis-Angle Rotation
# =============================================================================

def orientation_from_axis_angle(
    axis: tuple[float, float, float] | list[float] | NDArray,
    angle: float,
    degrees: bool = True
) -> NDArray[np.float64]:
    """
    Create orientation from axis-angle representation.
    
    Rotates `angle` degrees/radians about the given axis.
    
    Parameters
    ----------
    axis : array-like
        Rotation axis [x, y, z]. Will be normalized.
    angle : float
        Rotation angle [degrees or radians]
    degrees : bool
        If True (default), angle is in degrees.
        
    Returns
    -------
    NDArray[np.float64]
        Quaternion [x, y, z, w]
        
    Examples
    --------
    >>> # Rotate 45 degrees about Z axis
    >>> q = orientation_from_axis_angle([0, 0, 1], 45)
    
    >>> # Rotate 30 degrees about diagonal axis
    >>> q = orientation_from_axis_angle([1, 1, 0], 30)
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)  # Normalize
    
    if degrees:
        angle = np.deg2rad(angle)
    
    rotvec = axis * angle
    rot = R.from_rotvec(rotvec)
    return rot.as_quat()


# =============================================================================
# Direction-based orientation (most intuitive!)
# =============================================================================

def orientation_from_direction(
    body_axis: str = "z",
    toward: tuple[float, float, float] | list[float] | NDArray = (0, 0, 1),
    up_hint: tuple[float, float, float] | list[float] | NDArray = (0, 0, 1),
) -> NDArray[np.float64]:
    """
    Create orientation by pointing a body axis toward a direction.
    
    This is the most intuitive method for engineers. Specify which
    body axis (x, y, or z) should point in which global direction.
    
    Parameters
    ----------
    body_axis : str
        Which body axis to align: 'x', 'y', or 'z' (or '+x', '-x', etc.)
    toward : array-like
        Target direction in global frame [x, y, z]. Will be normalized.
    up_hint : array-like
        Hint for "up" direction to resolve rotation ambiguity.
        Default is global Z-up.
        
    Returns
    -------
    NDArray[np.float64]
        Quaternion [x, y, z, w]
        
    Examples
    --------
    >>> # Point body's +Z axis toward +Y global (tilted 90 deg)
    >>> q = orientation_from_direction('z', toward=[0, 1, 0])
    
    >>> # Point body's +X axis downward (-Z global)
    >>> q = orientation_from_direction('x', toward=[0, 0, -1])
    
    >>> # Default: Z-up (identity)
    >>> q = orientation_from_direction('z', toward=[0, 0, 1])
    """
    # Parse axis sign
    axis = body_axis.lower().strip()
    sign = 1.0
    if axis.startswith('-'):
        sign = -1.0
        axis = axis[1:]
    elif axis.startswith('+'):
        axis = axis[1:]
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError(f"body_axis must be 'x', 'y', or 'z', got '{body_axis}'")
    
    # Normalize target direction
    target = np.asarray(toward, dtype=np.float64)
    target = target / np.linalg.norm(target)
    target *= sign  # Apply sign
    
    # Get the body axis vector
    body_vec = np.zeros(3)
    body_vec[axis_map[axis]] = 1.0
    
    # Find rotation from body_vec to target
    # Using scipy's align_vectors for robustness
    up = np.asarray(up_hint, dtype=np.float64)
    up = up / np.linalg.norm(up)
    
    # Create orthonormal frame
    # Primary: target direction
    # Secondary: derived from up hint
    z_new = target
    x_new = np.cross(up, z_new)
    
    if np.linalg.norm(x_new) < 1e-6:
        # up_hint parallel to target, pick arbitrary perpendicular
        x_new = np.array([1, 0, 0]) if abs(z_new[0]) < 0.9 else np.array([0, 1, 0])
        x_new = x_new - z_new * np.dot(x_new, z_new)
    
    x_new = x_new / np.linalg.norm(x_new)
    y_new = np.cross(z_new, x_new)
    
    # Build rotation matrix
    R_mat = np.column_stack([x_new, y_new, z_new])
    
    # If aligning a different axis than Z, we need to adjust
    if axis == 'x':
        # Cycle: X->Z, Y->X, Z->Y
        R_mat = R_mat[:, [2, 0, 1]]
    elif axis == 'y':
        # Cycle: X->Z, Y->X, Z->Y then swap
        R_mat = R_mat[:, [0, 2, 1]]
        R_mat[:, 2] *= -1  # Fix handedness
    
    rot = R.from_matrix(R_mat)
    return rot.as_quat()


# =============================================================================
# Two-vector alignment (look-at)
# =============================================================================

def orientation_look_at(
    forward: tuple[float, float, float] | list[float] | NDArray,
    up: tuple[float, float, float] | list[float] | NDArray = (0, 0, 1),
    forward_axis: str = "x"
) -> NDArray[np.float64]:
    """
    Create orientation for a "look-at" style camera/sensor.
    
    Points the forward axis of the body toward a direction,
    with the body's up roughly aligned with the given up vector.
    
    Parameters
    ----------
    forward : array-like
        Direction to look toward in global frame.
    up : array-like
        Approximate up direction. Default is Z-up.
    forward_axis : str
        Which body axis is "forward": 'x', 'y', or 'z'.
        
    Returns
    -------
    NDArray[np.float64]
        Quaternion [x, y, z, w]
        
    Examples
    --------
    >>> # Look toward +X global with Z-up
    >>> q = orientation_look_at([1, 0, 0], up=[0, 0, 1])
    
    >>> # Look downward
    >>> q = orientation_look_at([0, 0, -1], up=[1, 0, 0])
    """
    return orientation_from_direction(forward_axis, toward=forward, up_hint=up)


# =============================================================================
# Conversion utilities
# =============================================================================

def quaternion_to_euler(
    q: NDArray[np.float64],
    order: str = "xyz",
    degrees: bool = True
) -> tuple[float, float, float]:
    """
    Convert quaternion to Euler angles for inspection.
    
    Parameters
    ----------
    q : NDArray[np.float64]
        Quaternion [x, y, z, w]
    order : str
        Euler sequence, default "xyz"
    degrees : bool
        If True, return degrees. Otherwise radians.
        
    Returns
    -------
    tuple[float, float, float]
        (roll, pitch, yaw) angles
    """
    rot = R.from_quat(q)
    angles = rot.as_euler(order, degrees=degrees)
    return tuple(angles)


def describe_orientation(q: NDArray[np.float64]) -> str:
    """
    Get human-readable description of an orientation.
    
    Parameters
    ----------
    q : NDArray[np.float64]
        Quaternion [x, y, z, w]
        
    Returns
    -------
    str
        Description string with Euler angles.
        
    Examples
    --------
    >>> describe_orientation([0, 0, 0.707, 0.707])
    "Roll: 0.0°, Pitch: 0.0°, Yaw: 90.0°"
    """
    roll, pitch, yaw = quaternion_to_euler(q, degrees=True)
    return f"Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°"
