from rigid_body import RigidBody6DOF
from forces import GravityForce, DragForce
from joints import Joint
from world import World

# Example simulation
world = World(dt=0.01, integrator='semi')

payload = RigidBody6DOF(
    name='payload',
    mass=10,
    inertia_tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    position=[0, 0, 10],
    orientation=[0, 0, 0, 1],
    linear_velocity=[0, 0, 0],
    angular_velocity=[0, 0, 0],
    radius=0.5
)

parachute = RigidBody6DOF(
    name='parachute',
    mass=1,
    inertia_tensor=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
    position=[0, 0, 11],
    orientation=[0, 0, 0, 1],
    linear_velocity=[0, 0, 0],
    angular_velocity=[0, 0, 0],
    radius=0.3
)

gravity = GravityForce([0, 0, -9.81])
world.add_global_force(gravity)

drag_parachute = DragForce(rho=1.2, Cd=1.5, area=10)
parachute.forces.append(drag_parachute)

joint = Joint(
    body1=payload,
    body2=parachute,
    local_point1=[0, 0, 0],
    local_point2=[0, 0, 0],
    joint_type='ball_socket'
)
world.add_joint(joint)

world.add_body(payload)
world.add_body(parachute)

world.run(duration=10)