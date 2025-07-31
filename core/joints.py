import numpy as np
from .math_utils import quaternion_to_rotation_matrix

class Joint:
    def apply_constraint_forces(self): pass

class FixedJoint(Joint):
    def __init__(self, body_a, body_b, pA_local, pB_local, k=1e5, d=1e3):
        self.body_a, self.body_b = body_a, body_b
        self.pA, self.pB = np.array(pA_local), np.array(pB_local)
        self.k, self.d = k, d
    def apply_constraint_forces(self):
        Ra = quaternion_to_rotation_matrix(self.body_a.orientation)
        Rb = quaternion_to_rotation_matrix(self.body_b.orientation)
        pa = self.body_a.position + Ra @ self.pA
        pb = self.body_b.position + Rb @ self.pB
        δ = pb - pa
        v_rel = self.body_b.linear_velocity - self.body_a.linear_velocity
        F = self.k * δ - self.d * v_rel
        self.body_a.apply_force(F, pa)
        self.body_b.apply_force(-F, pb)
        print(f"[JOINT] Fixed between {self.body_a.name}-{self.body_b.name}: F={F}")

class RevoluteJoint(FixedJoint):
    def __init__(self, body_a, body_b, anchorA, anchorB,
                 axisA, axisB, k=1e5, d=1e3, friction_coef=0.1):
        super().__init__(body_a,body_b,anchorA,anchorB,k,d)
        self.axisA = np.array(axisA)/np.linalg.norm(axisA)
        self.axisB = np.array(axisB)/np.linalg.norm(axisB)
        self.friction_coef = friction_coef
    def apply_constraint_forces(self):
        super().apply_constraint_forces()
        Ra = quaternion_to_rotation_matrix(self.body_a.orientation)
        Rb = quaternion_to_rotation_matrix(self.body_b.orientation)
        axisA_w = Ra @ self.axisA
        axisB_w = Rb @ self.axisB
        torque_align = self.k * np.cross(axisA_w, axisB_w)
        self.body_a.torque += torque_align
        self.body_b.torque -= torque_align
        τ_norm = np.dot(self.body_a.angular_velocity, self.axisA)
        τ_fric = -self.friction_coef * τ_norm * self.axisA
        self.body_a.torque += τ_fric
        self.body_b.torque -= τ_fric
        print(f"[JOINT] Revolute: align τ={torque_align}, fric τ_fric={τ_fric}")
