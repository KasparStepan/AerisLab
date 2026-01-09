"""
Tests for World simulation orchestrator.
"""
import pytest
import numpy as np
from pathlib import Path
import shutil

from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver, HybridIVPSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity, Drag
from aerislab.dynamics.constraints import DistanceConstraint


@pytest.fixture
def simple_world():
    """Create a simple world with one body in free fall."""
    world = World(ground_z=0.0, payload_index=0)
    
    I = np.eye(3) * 0.1
    body = RigidBody6DOF(
        "payload",
        mass=1.0,
        inertia_tensor_body=I,
        position=np.array([0.0, 0.0, 100.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    world.add_body(body)
    world.add_global_force(Gravity(np.array([0.0, 0.0, -9.81])))
    
    return world


@pytest.fixture
def cleanup_output():
    """Clean up output directory after tests."""
    yield
    output_dir = Path("output")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_world_creation():
    """Test basic World instantiation."""
    world = World(ground_z=0.0, payload_index=0)
    assert world.ground_z == 0.0
    assert world.payload_index == 0
    assert world.t == 0.0
    assert len(world.bodies) == 0
    assert world.logger is None


def test_world_with_logging_factory(cleanup_output):
    """Test World.with_logging() factory method."""
    world = World.with_logging("test_sim", ground_z=5.0)
    
    assert world.output_path is not None
    assert world.output_path.exists()
    assert (world.output_path / "logs").exists()
    assert (world.output_path / "plots").exists()
    assert world.logger is not None


def test_enable_logging(cleanup_output):
    """Test explicit logging enablement."""
    world = World(ground_z=0.0)
    
    # Should fail without name
    with pytest.raises(ValueError, match="Simulation name required"):
        world.enable_logging()
    
    # Should succeed with name
    output_path = world.enable_logging("my_test")
    assert output_path.exists()
    assert world.logger is not None


def test_add_body(simple_world):
    """Test adding bodies to world."""
    assert len(simple_world.bodies) == 1
    
    I = np.eye(3)
    body2 = RigidBody6DOF(
        "body2", 2.0, I, 
        np.array([0, 0, 50]),
        np.array([0, 0, 0, 1])
    )
    idx = simple_world.add_body(body2)
    
    assert idx == 1
    assert len(simple_world.bodies) == 2


def test_fixed_step_integration(simple_world):
    """Test fixed-step simulation runs without error."""
    solver = HybridSolver(alpha=5.0, beta=1.0)
    simple_world.run(solver, duration=1.0, dt=0.01)
    
    # Check body has fallen
    assert simple_world.bodies[0].p[2] < 100.0
    assert simple_world.t == pytest.approx(1.0, abs=0.02)


def test_ground_termination(simple_world):
    """Test simulation stops at ground contact."""
    solver = HybridSolver(alpha=5.0, beta=1.0)
    simple_world.run(solver, duration=100.0, dt=0.01)
    
    # Should stop before 100s (body hits ground)
    assert simple_world.t < 100.0
    assert simple_world.t_touchdown is not None
    
    # Payload should be at or below ground
    assert simple_world.bodies[0].p[2] <= simple_world.ground_z + 0.1


def test_custom_termination():
    """Test custom termination callback."""
    world = World(ground_z=0.0)
    I = np.eye(3) * 0.1
    body = RigidBody6DOF(
        "test", 1.0, I,
        np.array([0, 0, 100]),
        np.array([0, 0, 0, 1])
    )
    world.add_body(body)
    world.add_global_force(Gravity(np.array([0, 0, -9.81])))
    
    # Stop at 50m altitude
    world.set_termination_callback(lambda w: w.bodies[0].p[2] < 50.0)
    
    solver = HybridSolver()
    world.run(solver, duration=100.0, dt=0.01)
    
    # Should stop around 50m
    assert 49.0 < world.bodies[0].p[2] < 51.0


def test_logging_output(simple_world, cleanup_output):
    """Test that logging creates expected files."""
    simple_world.enable_logging("log_test")
    
    solver = HybridSolver()
    simple_world.run(solver, duration=0.5, dt=0.01)
    
    # Check files exist
    csv_file = simple_world.output_path / "logs" / "simulation.csv"
    assert csv_file.exists()
    
    # Check CSV has data
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 2  # Header + at least one data row


def test_auto_save_plots(simple_world, cleanup_output):
    """Test automatic plot generation."""
    world = World.with_logging(
        "plot_test",
        ground_z=0.0,
        auto_save_plots=True
    )
    
    # Add body
    I = np.eye(3) * 0.1
    body = RigidBody6DOF(
        "test", 1.0, I,
        np.array([0, 0, 50]),
        np.array([0, 0, 0, 1])
    )
    world.add_body(body)
    world.add_global_force(Gravity(np.array([0, 0, -9.81])))
    
    solver = HybridSolver()
    world.run(solver, duration=1.0, dt=0.01)
    
    # Check plots created
    plots_dir = world.output_path / "plots"
    assert (plots_dir / "test_trajectory_3d.png").exists()
    assert (plots_dir / "test_velocity_acceleration.png").exists()
    assert (plots_dir / "test_forces.png").exists()


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


def test_ivp_integration(simple_world):
    """Test variable-step IVP solver."""
    solver = HybridIVPSolver(method="Radau", rtol=1e-6, atol=1e-8)
    sol = simple_world.integrate_to(solver, t_end=5.0)
    
    assert sol.success or sol.status == 1  # 1 = event terminated
    assert len(sol.t) > 0
    assert simple_world.t > 0


def test_disable_logging(cleanup_output):
    """Test that logging can be disabled."""
    world = World.with_logging("disable_test")
    assert world.logger is not None
    
    world.disable_logging()
    assert world.logger is None


def test_multiple_bodies_constraint():
    """Test simulation with multiple bodies and constraints."""
    world = World(ground_z=0.0, payload_index=0)
    
    I = np.eye(3) * 0.1
    body1 = RigidBody6DOF(
        "body1", 1.0, I,
        np.array([0, 0, 100]),
        np.array([0, 0, 0, 1])
    )
    body2 = RigidBody6DOF(
        "body2", 0.5, I,
        np.array([0, 0, 105]),
        np.array([0, 0, 0, 1])
    )
    
    idx1 = world.add_body(body1)
    idx2 = world.add_body(body2)
    
    # Add distance constraint
    constraint = DistanceConstraint(
        world.bodies, idx1, idx2,
        np.zeros(3), np.zeros(3),
        length=5.0
    )
    world.add_constraint(constraint)
    
    world.add_global_force(Gravity(np.array([0, 0, -9.81])))
    
    solver = HybridSolver(alpha=5.0, beta=1.0)
    world.run(solver, duration=1.0, dt=0.01)
    
    # Check constraint maintained (approximately)
    dist = np.linalg.norm(body2.p - body1.p)
    assert abs(dist - 5.0) < 0.1  # Within 10cm tolerance
