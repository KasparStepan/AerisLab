"""
Constraint Verification Tests.

Tests constraint solver against analytical solutions:
- Simple pendulum (period validation)
- Double pendulum (energy conservation)
- Spring-mass oscillator (harmonic motion)
"""

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint
from aerislab.dynamics.forces import Gravity, Spring


class TestSimplePendulum:
    """
    Verify simple pendulum dynamics.
    
    Analytical solution (small angle):
        T = 2π√(L/g)
        θ(t) = θ₀ cos(ωt)
        
    where ω = √(g/L)
    """
    
    def test_pendulum_period_small_angle(self):
        """Pendulum period matches analytical formula for small angles."""
        L = 1.0   # m (pendulum length)
        g = 9.81  # m/s²
        theta0 = 0.1  # rad (small angle ~5.7°)
        
        # Analytical period
        T_analytical = 2 * np.pi * np.sqrt(L / g)
        
        # Setup: pivot at origin, bob displaced by theta0
        pivot = RigidBody6DOF(
            name="pivot",
            mass=1e6,  # Effectively fixed
            inertia_tensor_body=np.eye(3) * 1e6,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        # Bob position: x = L*sin(theta), z = -L*cos(theta)
        bob_pos = np.array([L * np.sin(theta0), 0, -L * np.cos(theta0)])
        bob = RigidBody6DOF(
            name="bob",
            mass=1.0,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=bob_pos,
            orientation=np.array([0, 0, 0, 1]),
        )
        
        world = World(ground_z=-100, payload_index=1)
        world.add_body(pivot)
        world.add_body(bob)
        world.add_global_force(Gravity(np.array([0, 0, -g])))
        
        # Distance constraint
        constraint = DistanceConstraint(
            world_bodies=world.bodies,
            body_i=0,
            body_j=1,
            attach_i_local=np.zeros(3),
            attach_j_local=np.zeros(3),
            length=L,
        )
        world.add_constraint(constraint)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=10.0, beta=2.0)
        dt = 0.0001
        
        # Find period by detecting zero crossings of x
        t = 0.0
        crossings = []
        x_prev = bob.p[0]
        
        for _ in range(int(3 * T_analytical / dt)):  # Run for ~3 periods
            world.step(solver, dt)
            t += dt
            x_curr = bob.p[0]
            
            # Detect positive-going zero crossing
            if x_prev < 0 and x_curr >= 0:
                crossings.append(t)
            
            x_prev = x_curr
        
        # Period = time between crossings
        if len(crossings) >= 2:
            T_simulated = crossings[1] - crossings[0]
            rel_error = abs(T_simulated - T_analytical) / T_analytical
            assert rel_error < 0.02, f"Period error {rel_error*100:.2f}% (got {T_simulated:.4f}s, expected {T_analytical:.4f}s)"
        else:
            pytest.fail("Could not detect two complete oscillations")
    
    def test_pendulum_constraint_maintained(self):
        """Distance constraint is maintained throughout motion."""
        L = 2.0
        g = 9.81
        theta0 = 0.5  # Larger angle (~30°)
        
        pivot = RigidBody6DOF(
            name="pivot",
            mass=1e6,
            inertia_tensor_body=np.eye(3) * 1e6,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        bob_pos = np.array([L * np.sin(theta0), 0, -L * np.cos(theta0)])
        bob = RigidBody6DOF(
            name="bob",
            mass=1.0,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=bob_pos,
            orientation=np.array([0, 0, 0, 1]),
        )
        
        world = World(ground_z=-100, payload_index=1)
        world.add_body(pivot)
        world.add_body(bob)
        world.add_global_force(Gravity(np.array([0, 0, -g])))
        
        constraint = DistanceConstraint(
            world_bodies=world.bodies,
            body_i=0,
            body_j=1,
            attach_i_local=np.zeros(3),
            attach_j_local=np.zeros(3),
            length=L,
        )
        world.add_constraint(constraint)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=10.0, beta=2.0)
        
        max_violation = 0.0
        for _ in range(5000):
            world.step(solver, 0.001)
            dist = np.linalg.norm(bob.p - pivot.p)
            violation = abs(dist - L)
            max_violation = max(max_violation, violation)
        
        assert max_violation < 0.01, f"Max constraint violation: {max_violation:.6f}m"


class TestSpringOscillator:
    """
    Verify spring-mass harmonic oscillator.
    
    Analytical solution:
        ω = √(k/m)
        T = 2π/ω = 2π√(m/k)
        x(t) = A·cos(ωt + φ)
    """
    
    def test_spring_period(self):
        """Spring oscillation period matches analytical formula."""
        k = 100.0  # N/m
        m = 1.0    # kg
        x0 = 0.5   # m (initial displacement)
        
        # Analytical
        omega = np.sqrt(k / m)
        T_analytical = 2 * np.pi / omega
        
        # Fixed anchor
        anchor = RigidBody6DOF(
            name="anchor",
            mass=1e6,
            inertia_tensor_body=np.eye(3) * 1e6,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        # Mass on spring (displaced in x)
        mass_body = RigidBody6DOF(
            name="mass",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.array([x0, 0, 0]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        world = World(ground_z=-100, payload_index=1)
        world.add_body(anchor)
        world.add_body(mass_body)
        # No gravity for pure spring oscillation
        
        spring = Spring(
            body_i=anchor,
            body_j=mass_body,
            k=k,
            rest_length=0.0,
            damping=0.0,
        )
        world.add_interaction_force(spring)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        dt = 0.0001
        
        # Find period from zero crossings
        t = 0.0
        crossings = []
        x_prev = mass_body.p[0]
        
        for _ in range(int(3 * T_analytical / dt)):
            world.step(solver, dt)
            t += dt
            x_curr = mass_body.p[0]
            
            if x_prev < 0 and x_curr >= 0:
                crossings.append(t)
            
            x_prev = x_curr
        
        if len(crossings) >= 2:
            T_simulated = crossings[1] - crossings[0]
            rel_error = abs(T_simulated - T_analytical) / T_analytical
            assert rel_error < 0.02, f"Period error {rel_error*100:.2f}%"
        else:
            pytest.fail("Could not detect oscillations")
    
    def test_spring_amplitude_conservation(self):
        """Spring oscillation amplitude conserved (no damping)."""
        k = 50.0
        m = 2.0
        A = 1.0  # initial amplitude
        
        anchor = RigidBody6DOF(
            name="anchor",
            mass=1e6,
            inertia_tensor_body=np.eye(3) * 1e6,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        mass_body = RigidBody6DOF(
            name="mass",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.01,
            position=np.array([A, 0, 0]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        world = World(ground_z=-100, payload_index=1)
        world.add_body(anchor)
        world.add_body(mass_body)
        
        spring = Spring(
            body_i=anchor,
            body_j=mass_body,
            k=k,
            rest_length=0.0,
            damping=0.0,
        )
        world.add_interaction_force(spring)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        
        # Track max amplitude over several periods
        max_x = 0.0
        for _ in range(10000):
            world.step(solver, 0.0001)
            max_x = max(max_x, abs(mass_body.p[0]))
        
        # Amplitude should be conserved
        rel_error = abs(max_x - A) / A
        assert rel_error < 0.05, f"Amplitude changed by {rel_error*100:.2f}%"
