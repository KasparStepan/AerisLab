"""
Integration tests that run actual example simulations.

These are smoke tests to ensure examples don't crash.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


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
    """Test 01_simple_drop.py runs without error."""
    script = examples_dir / "scenarios" / "01_simple_drop.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=30
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"


@pytest.mark.slow
def test_parachute_system_runs(examples_dir, cleanup_output):
    """Test 02_parachute_system.py runs without error."""
    script = examples_dir / "scenarios" / "02_parachute_system.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"


@pytest.mark.slow
def test_scenario_options_runs(examples_dir, cleanup_output):
    """Test 03_scenario_options.py runs without error."""
    script = examples_dir / "scenarios" / "03_scenario_options.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"


@pytest.mark.slow
def test_examples_create_output_files(examples_dir, cleanup_output):
    """Test that examples create expected output files."""
    script = examples_dir / "scenarios" / "01_simple_drop.py"

    subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        timeout=30
    )

    # Check output directory structure
    output_dir = Path("output")
    assert output_dir.exists()

    # Find the simulation folder (has timestamp)
    sim_folders = list(output_dir.glob("*simple*"))
    assert len(sim_folders) > 0, f"No output folders found in {output_dir}"

    sim_folder = sim_folders[0]
    assert (sim_folder / "logs").exists()
    assert (sim_folder / "plots").exists()
    assert (sim_folder / "logs" / "simulation.csv").exists()
