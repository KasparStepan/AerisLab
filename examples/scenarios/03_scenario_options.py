"""
Example 03: Scenario Options
Demonstrates the new ease-of-use features for setting up a scenario.
"""

from aerislab import Scenario
from aerislab.components import Payload, Parachute

def run_demo():
    print("\n--- Running Scenario with Custom Options ---")
    
    # 1. Define Components (Standard defaults)
    capsule = Payload("capsule", mass=100.0)
    chute = Parachute("chute", mass=5.0, diameter=15.0)
    
    # 2. Build Scenario with new options
    scenario = (
        Scenario(name="03_scenario_options")
        .add_system([capsule, chute])
        .connect(capsule, chute, length=10.0)
        
        # New Feature: Set global initial state
        # Shifts everything to 2000m and sets initial velocity
        .set_initial_state(altitude=2000.0, velocity=[20.0, 0, 0])
        
        # New Feature: Configure solver presets
        # 'fast' uses RK45 with looser tolerances for quick checks
        .configure_solver(preset="fast", atol=1e-5)
        
        # Enable plotting but don't block
        .enable_plotting(show=False)
    )
    
    # 3. Run
    scenario.run(duration=10.0, log_interval=1.0)
    print(f"Results saved to: {scenario.world.output_path}")

if __name__ == "__main__":
    run_demo()
