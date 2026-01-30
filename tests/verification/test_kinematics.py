"""
Kinematic Verification Tests.

Tests basic kinematic motion against analytical solutions:
- Free fall under gravity
- Constant velocity (force-free)
- Constant acceleration
"""

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity


class TestFreeFall:
    """
    Verify free-fall motion under gravity.
    
    Analytical solution:
        z(t) = z₀ + v₀t - ½gt²
        v(t) = v₀ - gt
    """
    
    def test_free_fall_position(self):
        """Position matches analytical solution."""
        # Setup
        z0 = 100.0  # m
        v0 = 0.0    # m/s
        g = 9.81    # m/s²
        t_end = 2.0  # s
        dt = 0.0001  # High precision
        
        body = RigidBody6DOF(
            name="ball",
            mass=1.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, z0]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.add_global_force(Gravity(np.array([0, 0, -g])))
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            world.step(solver, dt)
        
        # Analytical solution
        z_analytical = z0 + v0 * t_end - 0.5 * g * t_end**2
        z_simulated = body.p[2]
        
        error = abs(z_simulated - z_analytical)
        assert error < 0.01, f"Position error {error:.6f}m exceeds tolerance"
    
    def test_free_fall_velocity(self):
        """Velocity matches analytical solution."""
        z0 = 100.0
        v0 = 0.0
        g = 9.81
        t_end = 2.0
        dt = 0.0001
        
        body = RigidBody6DOF(
            name="ball",
            mass=1.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, z0]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.add_global_force(Gravity(np.array([0, 0, -g])))
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            world.step(solver, dt)
        
        # Analytical solution
        vz_analytical = v0 - g * t_end
        vz_simulated = body.v[2]
        
        error = abs(vz_simulated - vz_analytical)
        assert error < 0.01, f"Velocity error {error:.6f}m/s exceeds tolerance"
    
    def test_free_fall_with_initial_velocity(self):
        """Free fall with non-zero initial velocity."""
        z0 = 50.0
        v0 = 10.0  # Upward
        g = 9.81
        t_end = 3.0
        dt = 0.0001
        
        body = RigidBody6DOF(
            name="ball",
            mass=1.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, z0]),
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=np.array([0, 0, v0]),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.add_global_force(Gravity(np.array([0, 0, -g])))
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            world.step(solver, dt)
        
        z_analytical = z0 + v0 * t_end - 0.5 * g * t_end**2
        z_simulated = body.p[2]
        
        error = abs(z_simulated - z_analytical)
        assert error < 0.01, f"Position error {error:.6f}m exceeds tolerance"


class TestConstantVelocity:
    """
    Verify force-free motion (constant velocity).
    
    Analytical solution:
        x(t) = x₀ + v·t
    """
    
    def test_constant_velocity_no_forces(self):
        """Body drifts at constant velocity without forces."""
        x0 = np.array([0.0, 0.0, 100.0])
        v0 = np.array([10.0, 5.0, -2.0])
        t_end = 5.0
        dt = 0.001
        
        body = RigidBody6DOF(
            name="drifter",
            mass=1.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=x0.copy(),
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=v0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        # No forces!
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            world.step(solver, dt)
        
        x_analytical = x0 + v0 * t_end
        x_simulated = body.p
        
        error = np.linalg.norm(x_simulated - x_analytical)
        assert error < 0.01, f"Position error {error:.6f}m exceeds tolerance"
        
        # Velocity should remain constant
        v_error = np.linalg.norm(body.v - v0)
        assert v_error < 1e-6, f"Velocity changed by {v_error:.6f}m/s"
    
    def test_velocity_conservation(self):
        """Velocity magnitude conserved in force-free motion."""
        v0 = np.array([10.0, 5.0, -2.0])
        speed0 = np.linalg.norm(v0)
        
        body = RigidBody6DOF(
            name="drifter",
            mass=1.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=v0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        for _ in range(1000):
            world.step(solver, 0.01)
        
        speed_final = np.linalg.norm(body.v)
        error = abs(speed_final - speed0)
        assert error < 1e-6, f"Speed changed by {error:.6f}m/s"


class TestConstantAcceleration:
    """
    Verify motion under constant force (constant acceleration).
    
    Analytical solution for F = ma:
        a = F/m
        v(t) = v₀ + a·t
        x(t) = x₀ + v₀·t + ½·a·t²
    """
    
    def test_horizontal_constant_force(self):
        """Horizontal acceleration from constant force."""
        mass = 2.0  # kg
        force = np.array([10.0, 0.0, 0.0])  # N
        x0 = np.zeros(3)
        v0 = np.zeros(3)
        t_end = 3.0
        dt = 0.0001
        
        body = RigidBody6DOF(
            name="pushed",
            mass=mass,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=x0.copy(),
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=v0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        # Custom constant force to avoid Gravity (acceleration) confusion
        class ConstantForce:
            def __init__(self, f_vec): self.f = f_vec
            def apply(self, body, t): body.apply_force(self.f)
            
        world.add_global_force(ConstantForce(force))
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            world.step(solver, dt)
        
        # Analytical
        a = force / mass
        x_analytical = x0 + v0 * t_end + 0.5 * a * t_end**2
        v_analytical = v0 + a * t_end
        
        pos_error = np.linalg.norm(body.p - x_analytical)
        vel_error = np.linalg.norm(body.v - v_analytical)
        
        assert pos_error < 0.01, f"Position error {pos_error:.6f}m"
        assert vel_error < 0.01, f"Velocity error {vel_error:.6f}m/s"
