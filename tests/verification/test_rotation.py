"""
Rotational Dynamics Verification Tests.

Tests rotational physics against analytical solutions:
- Torque-free spinning (angular momentum conservation)
- Dzhanibekov effect (intermediate axis instability)
"""

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver, HybridIVPSolver
from aerislab.dynamics.body import RigidBody6DOF


class TestTorqueFreeSpin:
    """
    Verify torque-free rotation (angular momentum conservation).
    
    For a torque-free rigid body:
        L = I·ω = constant (in inertial frame)
        E = ½ω·I·ω = constant
    """
    
    def test_angular_momentum_conservation_symmetric(self):
        """Angular momentum conserved for symmetric body."""
        # Symmetric body: I1 = I2 = I3
        I = 2.0
        inertia = I * np.eye(3)
        w0 = np.array([1.0, 0.5, 0.3])
        
        body = RigidBody6DOF(
            name="spinner",
            mass=1.0,
            inertia_tensor_body=inertia,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=w0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.set_termination_callback(lambda w: False)
        
        # Initial angular momentum (body frame = world frame initially)
        L0 = inertia @ w0
        L0_mag = np.linalg.norm(L0)
        
        solver = HybridIVPSolver(method="RK45", rtol=1e-9, atol=1e-9)
        world.integrate_to(solver, t_end=5.0)
        
        # Final angular momentum in world frame
        R = body.rotation_world()
        I_world = R @ inertia @ R.T
        L_final = I_world @ body.w
        L_final_mag = np.linalg.norm(L_final)
        
        rel_error = abs(L_final_mag - L0_mag) / L0_mag
        assert rel_error < 0.01, f"Angular momentum changed by {rel_error*100:.2f}%"
    
    def test_rotational_kinetic_energy_conservation(self):
        """Rotational kinetic energy conserved."""
        inertia = np.diag([1.0, 2.0, 3.0])
        w0 = np.array([2.0, 1.0, 0.5])
        
        body = RigidBody6DOF(
            name="spinner",
            mass=1.0,
            inertia_tensor_body=inertia,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=w0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.set_termination_callback(lambda w: False)
        
        # Initial kinetic energy
        KE0 = 0.5 * w0 @ inertia @ w0
        
        solver = HybridIVPSolver(method="RK45", rtol=1e-9, atol=1e-9)
        world.integrate_to(solver, t_end=5.0)
        
        # Final kinetic energy
        KE_final = body.kinetic_energy()  # This includes translational too
        # For pure rotation: KE = 0.5 * w · I · w
        R = body.rotation_world()
        I_world = R @ inertia @ R.T
        KE_rot_final = 0.5 * body.w @ I_world @ body.w
        
        rel_error = abs(KE_rot_final - KE0) / KE0
        assert rel_error < 0.05, f"Rotational KE changed by {rel_error*100:.2f}%"
    
    def test_spin_about_principal_axis(self):
        """Stable spin about principal axis (no wobble)."""
        # Spin purely about z-axis (principal axis)
        inertia = np.diag([1.0, 1.0, 2.0])
        w0 = np.array([0.0, 0.0, 5.0])
        
        body = RigidBody6DOF(
            name="spinner",
            mass=1.0,
            inertia_tensor_body=inertia,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=w0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        for _ in range(2000):
            world.step(solver, 0.001)
        
        # Angular velocity should remain along z-axis
        w_xy = np.sqrt(body.w[0]**2 + body.w[1]**2)
        assert w_xy < 0.01, f"Wobble detected: w_xy = {w_xy:.4f} rad/s"


class TestDzhanibekovEffect:
    """
    Verify the Dzhanibekov (Tennis Racket) effect.
    
    For a body with I1 < I2 < I3:
    - Rotation about I1 (smallest): STABLE
    - Rotation about I2 (intermediate): UNSTABLE (flips!)
    - Rotation about I3 (largest): STABLE
    
    This is a key test of 3D rotational dynamics.
    """
    
    def test_stable_rotation_min_inertia_axis(self):
        """Rotation about minimum inertia axis is stable."""
        # I1 = 1 (min), I2 = 2, I3 = 3 (max)
        inertia = np.diag([1.0, 2.0, 3.0])
        
        # Spin about x-axis (minimum inertia) with small perturbation
        w0 = np.array([5.0, 0.01, 0.01])
        
        body = RigidBody6DOF(
            name="racket",
            mass=1.0,
            inertia_tensor_body=inertia,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=w0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        
        # Track wx over time
        wx_values = []
        for _ in range(5000):
            world.step(solver, 0.001)
            wx_values.append(body.w[0])
        
        # For stable rotation, wx should stay close to initial
        wx_array = np.array(wx_values)
        wx_variation = np.std(wx_array)
        assert wx_variation < 0.5, f"Rotation unstable: std(wx) = {wx_variation:.3f}"
    
    def test_stable_rotation_max_inertia_axis(self):
        """Rotation about maximum inertia axis is stable."""
        inertia = np.diag([1.0, 2.0, 3.0])
        
        # Spin about z-axis (maximum inertia) with perturbation
        w0 = np.array([0.01, 0.01, 5.0])
        
        body = RigidBody6DOF(
            name="racket",
            mass=1.0,
            inertia_tensor_body=inertia,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=w0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=0, beta=0)
        
        wz_values = []
        for _ in range(5000):
            world.step(solver, 0.001)
            wz_values.append(body.w[2])
        
        wz_array = np.array(wz_values)
        wz_variation = np.std(wz_array)
        assert wz_variation < 0.5, f"Rotation unstable: std(wz) = {wz_variation:.3f}"
    
    def test_unstable_rotation_intermediate_axis(self):
        """Rotation about intermediate inertia axis shows instability."""
        inertia = np.diag([1.0, 2.0, 3.0])
        
        # Spin about y-axis (intermediate inertia) with small perturbation
        # Increased perturbation to ensure instability triggers within simulation time
        w0 = np.array([0.2, 5.0, 0.2])
        
        body = RigidBody6DOF(
            name="racket",
            mass=1.0,
            inertia_tensor_body=inertia,
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
            angular_velocity=w0.copy(),
        )
        
        world = World(ground_z=-1000, payload_index=0)
        world.add_body(body)
        world.set_termination_callback(lambda w: False)
        
        solver = HybridIVPSolver(method="RK45", rtol=1e-6, atol=1e-8)
        
        # We need to capture wy history. integrating_to only gives final state unless we use dense_output or manual steps.
        # HybridIVPSolver.integrate updates world to final state but logs if logger enabled.
        # To get array of values without logging, we can just loop integrate_to with small chunks or use internal solver.
        
        # Let's run a loop of small integrals
        wy_values = []
        t_current = 0.0
        dt_check = 0.01
        for _ in range(1000):
            world.integrate_to(solver, t_current + dt_check)
            wy_values.append(body.w[1])
            t_current += dt_check
        
        # For Dzhanibekov effect, wy should show large oscillations/flips
        wy_array = np.array(wy_values)
        wy_range = np.max(wy_array) - np.min(wy_array)
        
        # Intermediate axis should show instability (large range)
        assert wy_range > 2.0, f"Expected instability not observed: range = {wy_range:.3f}"
