"""
AerisLab Component Architecture.

Components wrap RigidBody6DOF with domain-specific behavior using composition.
This pattern separates dynamics from aerospace-specific logic.

Example
-------
>>> from aerislab.components import Payload, Parachute, System
>>> system = System("recovery_system")
>>> system.add_component(Payload(...))
>>> system.add_component(Parachute(...))
"""

from .base import Component
from .parachute import DeploymentState, Parachute
from .payload import Payload
from .system import System

__all__ = [
    "Component",
    "Payload",
    "Parachute",
    "DeploymentState",
    "System",
]
