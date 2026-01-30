"""
Energy Conservation Verification Tests.

Tests energy conservation in various scenarios:
- Free fall (potential → kinetic)
- Pendulum (energy exchange)
- Spring-mass (energy exchange)
"""

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import DistanceConstraint
from aerislab.dynamics.forces import Gravity, Spring


class TestEnergyConservation:
    """
    Verify total mechanical energy conservation.
    
    For conservative systems:
        E = KE + PE = constant
    """
    
    def test_free_fall_energy_conservation(self):
        """Potential energy converts to kinetic in free fall."""
        z0 = 100.0  # m
        g = 9.81    # m/s²
        m = 2.0     # kg
        
        body = RigidBody6DOF(
            name="ball",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, z0]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.add_global_force(Gravity(np.array([0, 0, -g])))
        world.set_termination_callback(lambda w: False)
        
        # Initial energy: all potential
        PE0 = m * g * z0
        KE0 = 0.0
        E0 = PE0 + KE0
        
        solver = HybridSolver(alpha=0, beta=0)
        
        energies = []
        for _ in range(5000):
            world.step(solver, 0.001)
            
            PE = m * g * body.p[2]
            KE = 0.5 * m * np.dot(body.v, body.v)
            E = PE + KE
            energies.append(E)
        
        # Energy should be constant
        E_array = np.array(energies)
        rel_variation = (E_array.max() - E_array.min()) / E0
        
        assert rel_variation < 0.01, f"Energy varied by {rel_variation*100:.2f}%"
    
    def test_pendulum_energy_conservation(self):
        """Pendulum trades potential and kinetic energy."""
        L = 2.0
        g = 9.81
        m = 1.0
        theta0 = 0.5  # ~30°
        
        # Pivot (fixed)
        pivot = RigidBody6DOF(
            name="pivot",
            mass=1e6,
            inertia_tensor_body=np.eye(3) * 1e6,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        # Bob
        bob_pos = np.array([L * np.sin(theta0), 0, -L * np.cos(theta0)])
        bob = RigidBody6DOF(
            name="bob",
            mass=m,
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
        
        # Initial energy (bob only, pivot is fixed)
        PE0 = m * g * bob.p[2]
        KE0 = 0.0
        E0 = PE0 + KE0
        
        solver = HybridSolver(alpha=10.0, beta=2.0)
        
        energies = []
        for _ in range(5000):
            world.step(solver, 0.001)
            
            PE = m * g * bob.p[2]
            KE = 0.5 * m * np.dot(bob.v, bob.v)
            energies.append(PE + KE)
        
        E_array = np.array(energies)
        rel_variation = (E_array.max() - E_array.min()) / abs(E0)
        
        # Constraint can dissipate some energy via Baumgarte
        assert rel_variation < 0.05, f"Energy varied by {rel_variation*100:.2f}%"
    
    def test_spring_energy_conservation(self):
        """Spring-mass system conserves energy."""
        k = 100.0  # N/m
        m = 1.0    # kg
        x0 = 0.5   # m
        
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
            position=np.array([x0, 0, 0]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        world = World(ground_z=-100, payload_index=1)
        world.add_body(anchor)
        world.add_body(mass_body)
        
        spring = Spring(
            body_a=anchor,
            body_b=mass_body,
            attach_a_local=np.zeros(3),
            attach_b_local=np.zeros(3),
            k=k,
            rest_length=0.0,
            c=0.0,
        )
        world.add_interaction_force(spring)
        world.set_termination_callback(lambda w: False)
        
        # Initial energy: all spring potential
        PE_spring0 = 0.5 * k * x0**2
        KE0 = 0.0
        E0 = PE_spring0 + KE0
        
        solver = HybridSolver(alpha=0, beta=0)
        
        energies = []
        for _ in range(10000):
            world.step(solver, 0.0001)
            
            x = np.linalg.norm(mass_body.p - anchor.p)
            PE_spring = 0.5 * k * x**2
            KE = 0.5 * m * np.dot(mass_body.v, mass_body.v)
            energies.append(PE_spring + KE)
        
        E_array = np.array(energies)
        rel_variation = (E_array.max() - E_array.min()) / E0
        
        assert rel_variation < 0.02, f"Energy varied by {rel_variation*100:.2f}%"


class TestKineticEnergyPartitioning:
    """Test proper partitioning of kinetic energy."""
    
    def test_translation_kinetic_energy(self):
        """Translational KE = ½mv²."""
        m = 3.0
        v = np.array([4.0, 3.0, 0.0])  # speed = 5 m/s
        
        body = RigidBody6DOF(
            name="mover",
            mass=m,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=v,
        )
        
        KE_analytical = 0.5 * m * np.dot(v, v)  # = 0.5 * 3 * 25 = 37.5 J
        KE_simulated = body.kinetic_energy()
        
        # Should match exactly (no rotation)
        error = abs(KE_simulated - KE_analytical)
        assert error < 1e-10, f"KE error: {error:.6e} J"
    
    def test_rotation_kinetic_energy(self):
        """Rotational KE = ½ω·I·ω."""
        I = np.diag([1.0, 2.0, 3.0])
        w = np.array([2.0, 1.0, 0.5])
        
        body = RigidBody6DOF(
            name="spinner",
            mass=1.0,
            inertia_tensor_body=I,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=w,
        )
        
        # KE = ½ * (I1*w1² + I2*w2² + I3*w3²)
        KE_analytical = 0.5 * (I[0,0]*w[0]**2 + I[1,1]*w[1]**2 + I[2,2]*w[2]**2)
        KE_simulated = body.kinetic_energy()
        
        error = abs(KE_simulated - KE_analytical)
        assert error < 1e-10, f"Rotational KE error: {error:.6e} J"


def test_energy_conservation():
    """Test energy tracking for unconstrained system."""
    world = World(ground_z=-1000.0)  # Ground far away

    I = np.eye(3) * 0.1
    body = RigidBody6DOF(
        "test", 1.0, I,
        np.array([0, 0, 100]),
        np.array([0, 0, 0, 1]),
        linear_velocity=np.array([0, 0, -10])
    )
    world.add_body(body)
    world.add_global_force(Gravity(np.array([0, 0, -9.81])))

    E0 = world.get_energy()

    solver = HybridSolver()
    world.run(solver, duration=2.0, dt=0.01)

    E1 = world.get_energy()

    # Energy should be approximately conserved
    # (some drift expected due to numerical integration)
    assert abs(E1['total'] - E0['total']) / abs(E0['total']) < 0.05
