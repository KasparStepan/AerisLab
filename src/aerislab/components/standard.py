
"""
Standard concrete components for quick simulation setup.

These components pre-configure their own rigid bodies and default parameters,
avoiding the need for users to manually create RigidBody6DOF instances.
"""

from __future__ import annotations

import numpy as np
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.components.base import Component
from aerislab.components.system import System
from aerislab.models.aerodynamics.parachute_models import (
    ParachuteGeometry,
    ParachuteModelType,
    AdvancedParachute,
    create_parachute
)
from aerislab.dynamics.forces import Drag

class StandardComponent(Component):
    """
    Base for components that self-configure their rigid body.
    """
    def __init__(
        self, 
        name: str,
        mass: float,
        position: list[float] | np.ndarray = [0, 0, 0],
        velocity: list[float] | np.ndarray = [0, 0, 0],
        radius: float = 0.1,
        shape_inertia_factor: float = 0.4, # 0.4 for solid sphere
    ):
        p = np.array(position, dtype=float)
        v = np.array(velocity, dtype=float)
        
        # Simple isotropic inertia approximation for standard components
        # I = k * m * r^2
        if mass > 1e-6:
             I_val = shape_inertia_factor * mass * (radius**2)
        else:
             I_val = 1e-6
             
        I_body = np.eye(3) * I_val
        
        body = RigidBody6DOF(
            name=name + "_body",
            mass=mass,
            inertia_tensor_body=I_body,
            position=p,
            orientation=np.array([0, 0, 0, 1.0]), # Identity quaternion
            linear_velocity=v,
            radius=radius
        )
        
        super().__init__(name, body)
        self.initial_position = p
        self.initial_velocity = v

    def update_state(self, t: float, dt: float) -> None:
        pass


class Payload(StandardComponent):
    """
    Standard payload (capsule/cube/sphere).
    """
    def __init__(
        self,
        name: str = "payload",
        mass: float = 10.0,
        radius: float = 0.5,
        Cd: float = 0.5,
        area: float | None = None,
        **kwargs
    ):
        super().__init__(name, mass, radius=radius, **kwargs)
        
        # Add basic drag
        ref_area = area if area else np.pi * radius**2
        self.drag = Drag(
             rho=1.225, # Standard sea level, can be updated by world
             Cd=Cd,
             area=ref_area
        )
        self.add_force(self.drag)
        self._ref_area = ref_area

class Parachute(StandardComponent):
    """
    Standard round parachute using AdvancedParachute model.
    """
    def __init__(
        self,
        name: str = "parachute",
        mass: float = 2.0,
        diameter: float = 8.0,
        model: str = "knacke",
        activation_altitude: float = 0.0,
        **kwargs
    ):
        # Parachutes are light and large, inertia roughly estimated as hollow shell + entrained air approximation?
        # For now, just use standard solid sphere est for stability
        super().__init__(name, mass, radius=diameter/20.0, **kwargs) # Packed radius approx
        
        self.model = create_parachute(
            diameter=diameter,
            model=model,
            activation_altitude=activation_altitude,
            activation_velocity=0.0, # Handled by tailored deployment logic if needed
            Cd=0.8, # Default round chute
        )
        self.add_force(self.model)

    def update_state(self, t: float, dt: float) -> None:
         # Simple state logging
         self._state = {
            "inflated_area": self.model.get_current_area(),
            "is_open": self.model.is_activated()
         }
