"""
Tests for CSV logger with new Path support.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aerislab.core.simulation import World
from aerislab.dynamics.body import RigidBody6DOF
from aerislab.logger import CSVLogger


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


def test_logger_path_support(temp_dir):
    """Test logger accepts Path objects."""
    csv_path = temp_dir / "test.csv"
    logger = CSVLogger(csv_path)

    assert logger.filepath == csv_path
    assert isinstance(logger.filepath, Path)


def test_logger_invalid_fields():
    """Test logger rejects invalid field names."""
    with pytest.raises(ValueError, match="Invalid fields"):
        CSVLogger("test.csv", fields=["p", "invalid_field"])


def test_logger_context_manager(temp_dir):
    """Test logger works as context manager."""
    csv_path = temp_dir / "context.csv"

    world = World()
    body = RigidBody6DOF(
        "test", 1.0, np.eye(3),
        np.zeros(3), np.array([0, 0, 0, 1])
    )
    world.add_body(body)

    with CSVLogger(csv_path) as logger:
        for _ in range(10):
            logger.log(world)
            world.t += 0.1

    # File should exist and have data
    assert csv_path.exists()
    with open(csv_path) as f:
        lines = f.readlines()
        assert len(lines) == 11  # Header + 10 data rows


def test_logger_manual_close(temp_dir):
    """Test manual logger management."""
    csv_path = temp_dir / "manual.csv"

    world = World()
    body = RigidBody6DOF(
        "test", 1.0, np.eye(3),
        np.zeros(3), np.array([0, 0, 0, 1])
    )
    world.add_body(body)

    logger = CSVLogger(csv_path, buffer_size=5)

    for _ in range(10):
        logger.log(world)
        world.t += 0.1

    logger.close()

    assert csv_path.exists()


def test_logger_buffer_flush(temp_dir):
    """Test buffer flushes at correct size."""
    csv_path = temp_dir / "buffer.csv"

    world = World()
    body = RigidBody6DOF(
        "test", 1.0, np.eye(3),
        np.zeros(3), np.array([0, 0, 0, 1])
    )
    world.add_body(body)

    logger = CSVLogger(csv_path, buffer_size=3)

    # Log 2 rows (buffer not full)
    logger.log(world)
    world.t += 0.1
    logger.log(world)

    # Check file has only header (buffer not flushed)
    with open(csv_path) as f:
        lines = f.readlines()
        assert len(lines) == 1  # Just header

    # Log 3rd row (triggers flush)
    world.t += 0.1
    logger.log(world)

    # Now should have header + 3 rows
    with open(csv_path) as f:
        lines = f.readlines()
        assert len(lines) == 4

    logger.close()


def test_logger_custom_fields(temp_dir):
    """Test logging with custom field selection."""
    csv_path = temp_dir / "custom.csv"

    world = World()
    body = RigidBody6DOF(
        "test", 1.0, np.eye(3),
        np.zeros(3), np.array([0, 0, 0, 1])
    )
    world.add_body(body)

    # Log only position and velocity
    logger = CSVLogger(csv_path, fields=["p", "v"])
    logger.log(world)
    logger.close()

    # Check header has only p and v
    with open(csv_path) as f:
        header = f.readline().strip()
        assert "test.p_x" in header
        assert "test.v_x" in header
        assert "test.q_x" not in header  # Quaternion not logged
