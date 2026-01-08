# src/aerislab/utils/io.py
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def save_simulation_history(history: List[Dict[str, Any]], filepath: str) -> None:
    """
    Saves a list of state dictionaries to a CSV file.
    
    Args:
        history: List of dicts, e.g., [{'time': 0.1, 'x': 1.0}, ...]
        filepath: Destination path (e.g., 'results/run1.csv')
    """
    if not history:
        raise ValueError("Simulation history is empty. Nothing to save.")

    path = Path(filepath)
    # Ensure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history)
    df.to_csv(path, index=False)
    print(f"Simulation results saved to {path.absolute()}")

def load_simulation_config(filepath: str) -> Dict[str, Any]:
    """Placeholder for loading simulation settings from JSON/YAML in the future."""
    # This promotes separating code from configuration
    raise NotImplementedError("Config loading not yet implemented.")