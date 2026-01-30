"""
Example: Parachute-Payload System with Continuous Inflation Model.

This example demonstrates how to use the new System/Component architecture
integrated with the advanced continuous inflation parachute model and
the high-fidelity HybridIVPSolver.

Key Features:
- System/Component architecture for defining the vehicle
- AdvancedParachute model (Continuous Inflation) for realistic opening loads
- HybridIVPSolver (Radau) for stiff dynamics integration
- Custom ModernParachute component implementation
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aerislab.core.simulation import World
from aerislab.core.solver import HybridIVPSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.joints import RigidTetherJoint
from aerislab.components.system import System
from aerislab.components.base import Component
from aerislab.components.payload import Payload
from aerislab.models.aerodynamics.parachute_models import create_parachute, ParachuteModelType


class ModernParachute(Component):
    """
    Component wrapper for the AdvancedParachute model.
    
    Bridges the Component system architecture with the advanced 
    aerodynamic models.
    """
    
    def __init__(
        self,
        name: str,
        body: RigidBody6DOF,
        diameter: float,
        model_type: str = "continuous_inflation",
        activation_altitude: float = 300.0,
        activation_velocity: float = 30.0
    ):
        super().__init__(name, body)
        
        # Create the advanced parachute model using the factory
        self.model = create_parachute(
            diameter=diameter,
            model=model_type,
            activation_altitude=activation_altitude,
            activation_velocity=activation_velocity,
            Cd=0.85,  # Typical for round parachute
            porosity=0.05
        )
        
        # Register the model as a force on this component's body
        # The Component base class handles calling apply() during simulation
        self.add_force(self.model)
        
    def update_state(self, t: float, dt: float) -> None:
        """Sync component state with model state for logging."""
        self._state = {
            "component_type": "modern_parachute",
            "is_activated": self.model.is_activated(),
            "inflation_complete": self.model.is_fully_inflated(),
            "current_area": self.model.get_current_area(),
            "peak_ratio_est": self.model.get_peak_to_steady_ratio_estimate()
        }
        
        # Debug print every 1.0 second roughly (or if status changes)
        if int(t * 10) % 10 == 0 and dt > 0:
             print(f"t={t:.2f}s | Alt={self.body.p[2]:.1f}m | Vel={np.linalg.norm(self.body.v):.1f}m/s | "
                   f"Area={self.model.get_current_area():.2f}m2 | Act={self.model.is_activated()}")


def run_simulation():
    # 1. Create the Physics World
    # ---------------------------
    world = World.with_logging(
        name="continuous_inflation_demo",
        ground_z=0.0,
        payload_index=0,
        auto_save_plots=True
    )
    
    # 2. Create the System and Bodies
    # -------------------------------
    # Define physical properties
    payload_mass = 10.0  # kg
    parachute_mass = 2.0  # kg
    tether_length = 5.0   # m
    
    # Payload Body
    body_payload = RigidBody6DOF(
        name="capsule",
        mass=payload_mass,
        inertia_tensor_body=np.eye(3) * 5.0,
        position=np.array([0, 0, 300.0]),
        orientation=np.array([0, 0, 0, 1]),
        radius=0.5,
        linear_velocity=np.array([0, 30, 0])
    )
    
    # Parachute Body (initially packed on top of payload)
    body_chute = RigidBody6DOF(
        name="canopy",
        mass=parachute_mass,
        inertia_tensor_body=np.eye(3) * 1.0,
        position=np.array([0, 0, 300.0 + tether_length]),
        orientation=np.array([0, 0, 0, 1]),
        radius=0.3,  # Packed radius
        linear_velocity=np.array([0, 5, 0])
    )

    # 3. Create System Components
    # ---------------------------
    recovery_system = System(name="recovery_system")
    
    # Component A: Payload using standard library component
    payload_comp = Payload(
        name="payload_comp",
        body=body_payload,
        Cd=0.5,
        area=0.8
    )
    
    # Component B: Parachute using our custom wrapper and Continuous Inflation model
    parachute_comp = ModernParachute(
        name="parachute_comp",
        body=body_chute,
        diameter=3.0,  # 8m diameter
        model_type="continuous_inflation",
        activation_altitude=200.0,  # Deploy at 800m
        activation_velocity=30.0
    )
    
    # Add components to system
    idx_payload = recovery_system.add_component(payload_comp)
    idx_chute = recovery_system.add_component(parachute_comp)
    
    # 4. Add Constraints (Tether)
    # ---------------------------
    # Connect payload and parachute with a rigid tether
    tether_joint = RigidTetherJoint(
        body_i=idx_payload,
        body_j=idx_chute,
        attach_i_local=np.array([0, 0, 0.5]),  # Top of payload
        attach_j_local=np.array([0, 0, -0.3]), # Bottom of packed chute
        length=tether_length
    )
    # Convert joint definition to constraint using system bodies
    bodies = recovery_system.get_bodies()
    constraint = tether_joint.attach(bodies)
    recovery_system.add_constraint(constraint)
    
    # Register system with world
    world.add_system(recovery_system)
    
    # Also add gravity (System doesn't add global forces automatically)
    from aerislab.dynamics.forces import Gravity
    world.add_global_force(Gravity(np.array([0, 0, -9.81])))
    
    # 5. Configure Solver (IVP)
    # -------------------------
    # Use HybridIVPSolver with Radau (implicit) method for stiff inflation dynamics
    solver = HybridIVPSolver(
        method="Radau",
        rtol=1e-6,
        atol=1e-8,
        alpha=10.0,  # Baumgarte stabilization
        beta=2.0
    )
    
    print("="*60)
    print("Running Continuous Inflation System Demo")
    print(f"Model: {parachute_comp.model.model_type.name}")
    print(f"Solver: {solver.method} (IVP)")
    print("="*60)
    
    # 6. Run Simulation
    # -----------------
    # Integrate until touchdown or max time
    world.integrate_to(solver, t_end=60.0)
    
    print("\nSimulation Complete!")
    print(f"Touchdown time: {world.t:.2f} s")
    print(f"Final velocity: {np.linalg.norm(body_payload.v):.2f} m/s")
    print(f"Output saved to: {world.output_path}")

if __name__ == "__main__":
    run_simulation()
