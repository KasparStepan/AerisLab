import numpy as np
from .math_utils import normalize_quaternion, quaternion_multiply, quaternion_to_rotation_matrix

class RigidBody6DOF:
    def __init__(self,name, mass, inertia_tensor, position, orientation, linear_velocity, angular_velocity, radius=0.1):

        """
        A rigid body with 6 DOF state.
        orientation is quaternion [x,y,z,w].
        """

        self.name = name
        self.mass = float(mass)
        self.inv_mass = 0.0 if self.mass == 0 else 1.0 / self.mass
        self.inertia_tensor_body = np.array(inertia_tensor, dtype=np.float64)
        self.inv_inertia_tensor_body = np.linalg.inv(self.inertia_tensor_body) if mass != 0 else np.zeros((3, 3), dtype=np.float64)
        self.position = np.array(position, dtype=np.float64)
        self.orientation = normalize_quaternion(np.array(orientation, dtype=np.float64))
        self.linear_velocity = np.array(linear_velocity, dtype=np.float64)
        self.angular_velocity = np.array(angular_velocity, dtype=np.float64)
        self.radius = float(radius)
        self.force = np.zeros(3, dtype=np.float64)
        self.torque = np.zeros(3, dtype=np.float64)
        self.forces = [] # per-body force objects

        # Accels for hybrid solver integration
        self._a_lin = np.zeros(3)
        self._a_ang = np.zeros(3)


    def __str__(self):
        return (f"RigidBody6DOF(name={self.name})")

    def apply_force(self, force, point=None):
        
        """
        Apply a world-space force. If 'point' (world) is given,
        also apply corresponding torque τ += (point - position) X force.
        """

        f = np.array(force, dtype=np.float64)
        self.force += f

        if point is not None:
            r = np.array(point, dtype=np.float64) - self.position
            self.torque += np.cross(r,f)

    def apply_torque(self, torque):
        self.torque += np.array(torque, dtype=np.float64)             

    def clear_forces(self):
        """Clear the accumulated forces and torques."""
        self.force.fill(0)
        self.torque.fill(0)

    def get_inertia_world(self):
        """Compute the inertia tensor in world coordinates."""
        R = quaternion_to_rotation_matrix(self.orientation)
        return R @ self.inertia_tensor_body @ R.T
    
    def mass_matrix_world(self):
        """6x6 generalized mass matrix diag(m I3, I_world)."""
        Iw = self.get_inertia_world()
        M = np.zeros((6, 6))
        M[0:3, 0:3] = np.eye(3) * self.mass
        M[3:6, 3:6] = Iw
        return M

    def generalized_force(self):
        """Stacked [Fx,Fy,Fz, τx,τy,τz]."""
        return np.hstack([self.force, self.torque])