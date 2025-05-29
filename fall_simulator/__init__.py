"""
fall_simulator
==============
Main package for simulating 6DOF dynamics of a falling object.
"""

# Expose major modules if needed
from .dynamics import rigid_body
from .integration import euler, implicit
from .utils import quaternions, plotting