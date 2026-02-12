
"""
Example 02: Parachute System Definition.

Demonstrates how to link multiple components (Payload + Parachute)
using the Scenario API.
"""
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aerislab.api.scenario import Scenario
from aerislab.components.standard import Payload, Parachute

def run_example():
    # 1. Define physical components
    # Payload: 50kg instrument package
    payload = Payload(
        name="instrument_package",
        mass=50.0,
        radius=0.4,
        position=[0, 0, 2000] # 2km altitude
    )

    # Parachute: 12m diameter, Knacke inflation model
    # Starts packed just above the payload
    parachute = Parachute(
        name="main_chute",
        mass=5.0,
        diameter=12.0,
        model="knacke",
        activation_altitude=1500, # Opens at 1.5km
        position=[0, 0, 2000.5]   # 0.5m above payload
    )

    # 2. Build the scenario
    scenario = Scenario(name="02_parachute_system")
    
    # Add both components as a single system
    scenario.add_system([payload, parachute], name="recovery_system")
    
    # 3. connect components with a tether
    # Connect payload to parachute with a 10m line
    scenario.connect(payload, parachute, type="tether", length=10.0)

    # 4. Run simulation
    scenario.enable_plotting()
    scenario.run(duration=60.0)

    print(f"Simulation complete. Results saved to {scenario.world.output_path}")

if __name__ == "__main__":
    run_example()
