from core import World
from core import RigidBody6DOF
from core import GravityForce, DragForce, SpringForce
import matplotlib.pyplot as plt

# Example simulation with a payload and parachute
world = World(dt=0.01, integrator='semi')

# Payload
payload = RigidBody6DOF(
    name='payload',
    mass=1,
    inertia_tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    position=[0, 0, 20],
    orientation=[0, 0, 0, 1],
    linear_velocity=[0, 0, 10],
    angular_velocity=[0, 0, 0]
)

# Parachute
# parachute = RigidBody6DOF(
#     name='parachute',
#     mass=1,
#     inertia_tensor=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
#     position=[0, 0, 11],
#     orientation=[0, 0, 0, 1],
#     linear_velocity=[0, 0, 0],
#     angular_velocity=[0, 0, 0]
# )

# Forces
gravity = GravityForce([0, 0, -9.81])
world.add_global_force(gravity)

payload_drag = DragForce(rho=1.2, Cd=0.47, area=0.1)
payload.forces.append(payload_drag)

# drag_parachute = DragForce(rho=1.2, Cd=1.5, area=10)
# parachute.forces.append(drag_parachute)

# spring = SpringForce(
#     body1=payload,
#     body2=parachute,
#     local_point1=[0, 0, 0],
#     local_point2=[0, 0, 0],
#     rest_length=1,
#     stiffness=1000,
#     damping=10
# )
# world.add_interaction_force(spring)

world.add_body(payload)
# world.add_body(parachute)

world.run(duration=5.0)

# world.logger.print_all_logs()
#world.logger.save_to_csv('simulation_log.csv')
world.logger.plot_property('Pos_Z', 'payload', title='Payload Z-Position vs Time', ylabel='Z-Position (m)')

# Run and log
# for i in range(20):
#     world.step()
#     if i % 100 == 0:
#         print(f"Time: {world.time:.2f}, Payload pos: {payload.position}")#, Parachute pos: {parachute.position}")