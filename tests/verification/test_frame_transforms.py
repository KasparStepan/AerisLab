"""
Frame-Transform Verification Tests.

RigidBody6DOF.to_body / to_world are the single canonical place to convert
vectors between world and body frames (for orientation-dependent forces such as
aerodynamics or rotor thrust). These tests pin down the convention so future
force models can rely on it.
"""

import numpy as np
from scipy.spatial.transform import Rotation as ScR

from aerislab.dynamics.body import RigidBody6DOF


def _body(quat):
    return RigidBody6DOF(
        name="b", mass=1.0, inertia_tensor_body=np.eye(3),
        position=np.zeros(3), orientation=np.asarray(quat, dtype=float),
    )


def test_round_trip_identity():
    # to_world(to_body(v)) == v for an arbitrary orientation and vector.
    q = ScR.from_euler("xyz", [30, -45, 60], degrees=True).as_quat()
    b = _body(q)
    v = np.array([1.3, -2.1, 0.7])
    assert np.allclose(b.to_world(b.to_body(v)), v)
    assert np.allclose(b.to_body(b.to_world(v)), v)


def test_identity_orientation_is_passthrough():
    b = _body([0, 0, 0, 1])
    v = np.array([2.0, -3.0, 5.0])
    assert np.allclose(b.to_body(v), v)
    assert np.allclose(b.to_world(v), v)


def test_known_rotation_convention():
    # 90° about world +z: body +x axis points along world +y.
    q = ScR.from_euler("z", 90, degrees=True).as_quat()
    b = _body(q)
    # to_world maps body-frame x-hat -> world +y.
    assert np.allclose(b.to_world(np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-12)
    # to_body maps world +y -> body +x.
    assert np.allclose(b.to_body(np.array([0.0, 1.0, 0.0])), [1.0, 0.0, 0.0], atol=1e-12)


def test_matches_rotation_matrix():
    q = ScR.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()
    b = _body(q)
    R = b.rotation_world()
    v = np.array([0.4, 0.5, -0.6])
    assert np.allclose(b.to_world(v), R @ v)
    assert np.allclose(b.to_body(v), R.T @ v)
