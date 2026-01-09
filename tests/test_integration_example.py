"""
Integration tests that run actual example simulations.

These are smoke tests to ensure examples don't crash.
"""
import pytest
import subprocess
import sys
from pathlib import Path
import shutil


@pytest.fixture(scope="module")
def examples_dir():
    """Get examples directory."""
    return Path(__file__).parent.parent / "examples"


@pytest.fixture
def cleanup_output():
    """Clean up output after tests."""
    yield
    output_dir = Path("output")
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.mark.slow
def test_simple_drop_runs(examples_dir, cleanup_output):
    """Test simple_drop.py runs without error."""
    script = examples_dir / "simple_drop.py"
    
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Simulation terminated" in result.stdout or "completed" in result.stdout


@pytest.mark.slow
def test_parachute_fixed_runs(examples_dir, cleanup_output):
    """Test parachute_fixed.py runs without error."""
    script = examples_dir / "parachute_fixed.py"
    
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Touchdown" in result.stdout or "completed" in result.stdout


@pytest.mark.slow
def test_parachute_ivp_runs(examples_dir, cleanup_output):
    """Test parachute_ivp.py runs without error."""
    script = examples_dir / "parachute_ivp.py"
    
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Success" in result.stdout or "event" in result.stdout.lower()


@pytest.mark.slow
def test_examples_create_output_files(examples_dir, cleanup_output):
    """Test that examples create expected output files."""
    script = examples_dir / "simple_drop.py"
    
    subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        timeout=30
    )
    
    # Check output directory structure
    output_dir = Path("output")
    assert output_dir.exists()
    
    # Find the simulation folder (has timestamp)
    sim_folders = list(output_dir.glob("simple_drop_*"))
    assert len(sim_folders) > 0
    
    sim_folder = sim_folders[0]
    assert (sim_folder / "logs").exists()
    assert (sim_folder / "plots").exists()
    assert (sim_folder / "logs" / "simulation.csv").exists()
