"""
fall_simulator.utils
====================
Utility functions for plotting and quaternion math.
"""

from .quaternions import (
    quaternion_multiply,
    quaternion_derivative,
    normalize_quaternion,
)

from .plotting import (
    plot_trajectory,
    plot_position_vs_time,
    plot_energy_vs_time,
    animate_multibody_3d,
)
