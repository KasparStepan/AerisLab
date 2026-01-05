import numpy as np
import pytest
from aerislab import World, RigidBody6DOF, Gravity, HybridSolver

def test_fixed_step_termination_on_ground():
    I = np.eye(3)
    payload = RigidBody6DOF("payload", 1.0, I, np.array([0,0,10.0]), np.array([1,0,0,0]))
    w = World(ground_z=0.0, payload_index=0)
    w.add_body(payload)
    w.add_global_force(Gravity(np.array([0,0,-9.81])))

    solver = HybridSolver()
    dt = 1e-2
    w.run(solver, duration=5.0, dt=dt)

    # Free-fall exact touchdown time from 10 m: t = sqrt(2h/g)
    t_exact = (2*10.0/9.81)**0.5
    assert w.t_touchdown is not None
    assert abs(w.t_touchdown - t_exact) < 0.05  # linear interpolation helps

def test_ivp_event_halts_at_ground():
    scipy = pytest.importorskip("scipy")

    from aerislab import HybridIVPSolver

    I = np.eye(3)
    payload = RigidBody6DOF("payload", 1.0, I, np.array([0,0,5.0]), np.array([1,0,0,0]))
    w = World(ground_z=0.0, payload_index=0)
    w.add_body(payload)
    w.add_global_force(Gravity(np.array([0,0,-9.81])))

    ivp = HybridIVPSolver(method="Radau", rtol=1e-9, atol=1e-11, max_step=np.inf)
    sol = w.integrate_to(ivp, t_end=5.0)
    t_exact = (2*5.0/9.81)**0.5
    assert w.t_touchdown is not None
    assert abs(w.t_touchdown - t_exact) < 1e-6
