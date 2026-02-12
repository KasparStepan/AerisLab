
"""
Example 01: Simple Drop Test using the Scenario API.

This script demonstrates how to define a single falling object with 
minimal code using the new high-level API.
"""
import sys
from pathlib import Path

# Setup path for local development (not needed if installed via pip)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aerislab.api.scenario import Scenario
from aerislab.components.standard import Payload

def run_example():
    # 1. Define a payload (e.g., a simple capsule)
    # The 'Payload' component automatically creates a rigid body with 
    # appropriate inertia and basic drag properties.
    capsule = Payload(
        name="capsule", 
        mass=10.0,      # kg
        radius=0.5,     # m
        Cd=0.5,         # Drag coefficient
        position=[0, 0, 1000], # Start at 1km altitude
        velocity=[0, 0, 0]     # Start from rest
    )

    # 2. Create and run the scenario
    # The Scenario manager handles the world, solver, and logging automatically.
    scenario = Scenario(name="01_simple_drop") \
        .add_system([capsule]) \
        .enable_plotting(show=True) \
        .run(duration=30.0)

    print(f"Simulation complete. Results saved to {scenario.world.output_path}")

if __name__ == "__main__":
    run_example()
