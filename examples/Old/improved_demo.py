
"""
Verification script for the new Scenario API.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aerislab.components.standard import Parachute, Payload
from aerislab.api.scenario import Scenario
import numpy as np

# 1. Define Components (Physical defaults handled internally)
# We can override defaults cleanly
payload = Payload(
    name="capsule_verif", 
    mass=10.0, 
    radius=0.5,
    position=[0, 0, 400.0],
    velocity=[0, 0, 0]
)

chute = Parachute(
    name="chute_verif", 
    mass=2.0, 
    diameter=8.0, 
    model="knacke",
    activation_altitude=200.0,
    position=[0, 0, 405.0], # 5m above payload
    velocity=[0, 0, 0]
)

# 2. Build & Run
# Fluent interface
scenario = Scenario(name="improved_demo_verif")
scenario.add_system([payload, chute], name="recovery_verif")

# Connect them
scenario.connect(payload, chute, type="tether", length=5.0)

# Configure and Run
scenario.enable_plotting()
scenario.run(duration=10.0) # Short run for verification

print("Verification complete. Check output folder.")
