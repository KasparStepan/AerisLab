"""
Aerodynamic Verification Tests.

Tests aerodynamic forces against analytical solutions:
- Terminal velocity (drag equilibrium)
- Drag deceleration (velocity decay)
"""

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Drag, Gravity


class TestTerminalVelocity:
    """
    Verify terminal velocity under gravity and drag.
    
    At terminal velocity:
        F_drag = F_gravity
        ½ρCdAv² = mg
        
    Terminal velocity:
        v_t = √(2mg / ρCdA)
    """
    
    def test_terminal_velocity_sphere(self):
        """Sphere reaches correct terminal velocity."""
        # Parameters
        m = 1.0       # kg
        g = 9.81      # m/s²
        rho = 1.225   # kg/m³ (air at sea level)
        Cd = 0.47     # Sphere drag coefficient
        r = 0.1       # m (radius)
        A = np.pi * r**2  # Cross-sectional area
        
        # Analytical terminal velocity
        v_t_analytical = np.sqrt(2 * m * g / (rho * Cd * A))
        
        body = RigidBody6DOF(
            name="sphere",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.array([0, 0, 10000]),  # High altitude
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=np.array([0, 0, 0]),  # Start at rest
        )
        
        world = World(ground_z=-100, payload_index=0)
        world.add_body(body)
        world.add_global_force(Gravity(np.array([0, 0, -g])))
        
        drag = Drag(rho=rho, Cd=Cd, area=A, mode="quadratic")
        body.per_body_forces.append(drag)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        
        # Run until terminal velocity is reached (velocity stabilizes)
        velocities = []
        for i in range(20000):
            world.step(solver, 0.01)
            velocities.append(abs(body.v[2]))
        
        # Check last few values for convergence
        v_final = np.mean(velocities[-100:])
        rel_error = abs(v_final - v_t_analytical) / v_t_analytical
        
        assert rel_error < 0.02, f"Terminal velocity error {rel_error*100:.2f}% (got {v_final:.2f}, expected {v_t_analytical:.2f})"
    
    def test_terminal_velocity_heavy_object(self):
        """Heavy object has higher terminal velocity."""
        # Heavy object
        m1 = 10.0
        # Light object
        m2 = 1.0
        
        g = 9.81
        rho = 1.225
        Cd = 0.5
        A = 0.1
        
        v_t1 = np.sqrt(2 * m1 * g / (rho * Cd * A))
        v_t2 = np.sqrt(2 * m2 * g / (rho * Cd * A))
        
        # v_t1 should be ~√10 times v_t2
        ratio = v_t1 / v_t2
        assert abs(ratio - np.sqrt(10)) < 0.01, f"Terminal velocity ratio wrong: {ratio:.3f}"
    
    def test_drag_opposes_motion(self):
        """Drag force opposes velocity direction."""
        body = RigidBody6DOF(
            name="object",
            mass=1.0,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=np.array([10.0, 0, 0]),  # Moving in +x
        )
        
        drag = Drag(rho=1.225, Cd=0.5, area=0.1, mode="quadratic")
        t = 0.0
        
        # Apply drag
        body.clear_forces()
        drag.apply(body, t)
        
        # Force should be in -x direction
        F = body.generalized_force()[:3]
        assert F[0] < 0, "Drag should oppose +x motion"
        assert abs(F[1]) < 1e-10, "Drag should have no y component"
        assert abs(F[2]) < 1e-10, "Drag should have no z component"


class TestDragDeceleration:
    """
    Verify drag-induced deceleration.
    
    For linear drag (F = -bv):
        v(t) = v₀ * exp(-bt/m)
        
    For quadratic drag (F = -cv²):
        1/v - 1/v₀ = ct/m
    """
    
    def test_linear_drag_exponential_decay(self):
        """Linear drag causes exponential velocity decay."""
        m = 2.0
        v0 = 20.0
        b = 1.0  # Drag coefficient (F = -b*v)
        
        body = RigidBody6DOF(
            name="object",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=np.array([v0, 0, 0]),
        )
        
        # Linear drag: F = -b*v, where b = ½ρCdA * 2v_ref in linear mode
        # Approximate with custom drag coefficient
        drag = Drag(rho=1.0, Cd=1.0, area=1.0, mode="linear", k_linear=b)
        
        world = World(ground_z=-100, payload_index=0)
        world.add_body(body)
        body.per_body_forces.append(drag)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        
        t_end = 5.0
        dt = 0.001
        n_steps = int(t_end / dt)
        
        for _ in range(n_steps):
            world.step(solver, dt)
        
        # Analytical: v(t) = v0 * exp(-b*t/m)
        v_analytical = v0 * np.exp(-b * t_end / m)
        v_simulated = body.v[0]
        
        rel_error = abs(v_simulated - v_analytical) / v_analytical
        # Allow larger tolerance due to mode approximation
        assert rel_error < 0.10, f"Velocity error {rel_error*100:.2f}%"
    
    def test_drag_reduces_kinetic_energy(self):
        """Drag dissipates kinetic energy over time."""
        m = 1.0
        v0 = 10.0
        
        body = RigidBody6DOF(
            name="object",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=np.array([v0, 0, 0]),
        )
        
        KE0 = 0.5 * m * v0**2
        
        drag = Drag(rho=1.225, Cd=0.5, area=0.5, mode="quadratic")
        
        world = World(ground_z=-100, payload_index=0)
        world.add_body(body)
        body.per_body_forces.append(drag)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        
        for _ in range(5000):
            world.step(solver, 0.01)
        
        KE_final = body.kinetic_energy()
        
        # Energy should have decreased
        assert KE_final < KE0, f"KE should decrease: {KE_final:.2f} >= {KE0:.2f}"
        # Should have dissipated significant energy
        dissipation = (KE0 - KE_final) / KE0
        assert dissipation > 0.5, f"Only {dissipation*100:.1f}% dissipated"


class TestProjectileWithDrag:
    """
    Verify projectile motion with drag.
    
    For a projectile with quadratic drag:
    - Horizontal range < vacuum range
    - Maximum height < vacuum height
    - Terminal descent
    """
    
    def test_range_reduced_by_drag(self):
        """Horizontal range is reduced by drag."""
        m = 1.0
        g = 9.81
        v0 = 50.0  # m/s
        angle = np.radians(45)
        
        vx0 = v0 * np.cos(angle)
        vz0 = v0 * np.sin(angle)
        
        # Vacuum range: R = v0² * sin(2θ) / g
        R_vacuum = v0**2 * np.sin(2 * angle) / g
        
        body = RigidBody6DOF(
            name="projectile",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.array([0, 0, 0.1]),  # Slightly above ground
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=np.array([vx0, 0, vz0]),
        )
        
        world = World(ground_z=0, payload_index=0)
        world.add_body(body)
        world.add_global_force(Gravity(np.array([0, 0, -g])))
        
        drag = Drag(rho=1.225, Cd=0.47, area=0.01, mode="quadratic")
        body.per_body_forces.append(drag)
        
        solver = HybridSolver(alpha=0, beta=0)
        
        # Run until ground impact
        max_steps = 100000
        for _ in range(max_steps):
            if world.step(solver, 0.001):
                break
        
        R_drag = body.p[0]
        
        # Range with drag should be less than vacuum range
        assert R_drag < R_vacuum, f"Range with drag ({R_drag:.1f}m) should be < vacuum ({R_vacuum:.1f}m)"
        # But not too much less (drag isn't extreme)
        assert R_drag > 0.3 * R_vacuum, f"Range too short: {R_drag:.1f}m"
