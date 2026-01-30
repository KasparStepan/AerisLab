"""
Component Architecture Verification Tests.

Tests the component system functionality:
- Component state updates
- Parachute deployment
- System management
"""

import numpy as np
import pytest

from aerislab.components import Component, DeploymentState, Parachute, Payload, System
from aerislab.core.simulation import World
from aerislab.core.solver import HybridSolver
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Gravity


class TestPayloadComponent:
    """Test Payload component functionality."""
    
    def test_payload_creation(self):
        """Payload component initializes correctly."""
        body = RigidBody6DOF(
            name="test_payload",
            mass=10.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, 100]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        payload = Payload(name="payload", body=body, Cd=0.47, area=0.1)
        
        assert payload.name == "payload"
        assert payload.body is body
        assert payload.Cd == 0.47
        assert payload.area == 0.1
        assert len(payload.forces) == 1  # Drag force
    
    def test_payload_properties_delegate_to_body(self):
        """Payload properties delegate to wrapped body."""
        pos = np.array([1, 2, 3])
        vel = np.array([4, 5, 6])
        
        body = RigidBody6DOF(
            name="test",
            mass=5.0,
            inertia_tensor_body=np.eye(3),
            position=pos,
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=vel,
        )
        
        payload = Payload(name="p", body=body)
        
        np.testing.assert_array_equal(payload.position, pos)
        np.testing.assert_array_equal(payload.velocity, vel)
        assert payload.mass == 5.0


class TestParachuteComponent:
    """Test Parachute component and deployment state machine."""
    
    def test_parachute_starts_stowed(self):
        """Parachute initializes in STOWED state."""
        body = RigidBody6DOF(
            name="canopy",
            mass=2.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, 1000]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        chute = Parachute(
            name="main",
            body=body,
            Cd=1.5,
            area=15.0,
            activation_altitude=800.0,
        )
        
        assert chute.deployment_state == DeploymentState.STOWED
        assert chute.deployment_time is None
    
    def test_parachute_deploy_manual(self):
        """Parachute can be manually deployed."""
        body = RigidBody6DOF(
            name="canopy",
            mass=2.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, 1000]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        chute = Parachute(name="main", body=body, Cd=1.5, area=15.0)
        
        chute.deploy(t=5.0)
        
        assert chute.deployment_state == DeploymentState.DEPLOYING
        assert chute.deployment_time == 5.0
    
    def test_parachute_altitude_trigger(self):
        """Parachute deploys when altitude drops below trigger."""
        body = RigidBody6DOF(
            name="canopy",
            mass=2.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, 850]),  # Above trigger
            orientation=np.array([0, 0, 0, 1]),
        )
        
        chute = Parachute(
            name="main",
            body=body,
            Cd=1.5,
            area=15.0,
            activation_altitude=800.0,
        )
        
        # Above trigger - should stay stowed
        chute.update_state(t=0.0, dt=0.01)
        assert chute.deployment_state == DeploymentState.STOWED
        
        # Drop below trigger
        body.p[2] = 750.0
        chute.update_state(t=1.0, dt=0.01)
        assert chute.deployment_state == DeploymentState.DEPLOYING
    
    def test_parachute_velocity_trigger(self):
        """Parachute deploys when velocity exceeds trigger."""
        body = RigidBody6DOF(
            name="canopy",
            mass=2.0,
            inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, 2000]),  # High altitude
            orientation=np.array([0, 0, 0, 1]),
            linear_velocity=np.array([0, 0, -30]),  # Slow
        )
        
        chute = Parachute(
            name="main",
            body=body,
            Cd=1.5,
            area=15.0,
            activation_velocity=50.0,
        )
        
        # Below velocity trigger
        chute.update_state(t=0.0, dt=0.01)
        assert chute.deployment_state == DeploymentState.STOWED
        
        # Exceed velocity
        body.v[2] = -60.0
        chute.update_state(t=1.0, dt=0.01)
        assert chute.deployment_state == DeploymentState.DEPLOYING


class TestSystem:
    """Test System class for multi-component management."""
    
    def test_system_creation(self):
        """System initializes empty."""
        system = System(name="test_system")
        
        assert system.name == "test_system"
        assert len(system.components) == 0
        assert len(system.constraints) == 0
    
    def test_add_component(self):
        """Components can be added to system."""
        system = System(name="test")
        
        body = RigidBody6DOF(
            name="b1",
            mass=1.0,
            inertia_tensor_body=np.eye(3),
            position=np.zeros(3),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        payload = Payload(name="p1", body=body)
        idx = system.add_component(payload)
        
        assert idx == 0
        assert len(system.components) == 1
        assert system.get_component("p1") is payload
    
    def test_get_bodies(self):
        """System returns all rigid bodies."""
        system = System(name="test")
        
        body1 = RigidBody6DOF(
            name="b1", mass=1.0, inertia_tensor_body=np.eye(3),
            position=np.zeros(3), orientation=np.array([0, 0, 0, 1]),
        )
        body2 = RigidBody6DOF(
            name="b2", mass=2.0, inertia_tensor_body=np.eye(3),
            position=np.zeros(3), orientation=np.array([0, 0, 0, 1]),
        )
        
        system.add_component(Payload(name="p1", body=body1))
        system.add_component(Payload(name="p2", body=body2))
        
        bodies = system.get_bodies()
        assert len(bodies) == 2
        assert bodies[0] is body1
        assert bodies[1] is body2
    
    def test_update_all_states(self):
        """System updates all component states."""
        system = System(name="test")
        
        body = RigidBody6DOF(
            name="canopy", mass=2.0, inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, 700]),  # Below trigger
            orientation=np.array([0, 0, 0, 1]),
        )
        
        chute = Parachute(
            name="main", body=body, Cd=1.5, area=15.0,
            activation_altitude=800.0,
        )
        system.add_component(chute)
        
        assert chute.deployment_state == DeploymentState.STOWED
        
        system.update_all_states(t=1.0, dt=0.01)
        
        assert chute.deployment_state == DeploymentState.DEPLOYING


class TestWorldSystemIntegration:
    """Test World + System integration."""
    
    def test_add_system_registers_bodies(self):
        """World.add_system registers all bodies."""
        world = World(ground_z=0, payload_index=0)
        system = System(name="test")
        
        body1 = RigidBody6DOF(
            name="b1", mass=1.0, inertia_tensor_body=np.eye(3),
            position=np.zeros(3), orientation=np.array([0, 0, 0, 1]),
        )
        body2 = RigidBody6DOF(
            name="b2", mass=2.0, inertia_tensor_body=np.eye(3),
            position=np.zeros(3), orientation=np.array([0, 0, 0, 1]),
        )
        
        system.add_component(Payload(name="p1", body=body1))
        system.add_component(Payload(name="p2", body=body2))
        
        world.add_system(system)
        
        assert len(world.bodies) == 2
        assert len(world.systems) == 1
    
    def test_simulation_updates_component_states(self):
        """Simulation properly calls component state updates."""
        world = World(ground_z=0, payload_index=0)
        system = System(name="test")
        
        body = RigidBody6DOF(
            name="canopy", mass=2.0, inertia_tensor_body=np.eye(3) * 0.1,
            position=np.array([0, 0, 1000]),
            orientation=np.array([0, 0, 0, 1]),
        )
        
        chute = Parachute(
            name="main", body=body, Cd=1.5, area=15.0,
            activation_altitude=800.0,
        )
        system.add_component(chute)
        
        world.add_system(system)
        world.add_global_force(Gravity(np.array([0, 0, -9.81])))
        world.set_termination_callback(lambda w: False)
        
        solver = HybridSolver(alpha=5.0, beta=1.0)
        
        # Run until deployment should trigger
        for _ in range(10000):
            world.step(solver, 0.01)
            if chute.deployment_state != DeploymentState.STOWED:
                break
        
        # Should have deployed
        assert chute.deployment_state in [DeploymentState.DEPLOYING, DeploymentState.DEPLOYED]
        assert chute.deployment_time is not None
