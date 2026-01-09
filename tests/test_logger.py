import csv
import numpy as np
import pytest
from aerislab.logger import CSVLogger
from aerislab.core import World, HybridSolver
from aerislab.dynamics.body import RigidBody6DOF

# --- Mock Objects for Isolation ---
class MockBody:
    def __init__(self, name="b1"):
        self.name = name
        self.p = np.array([1.0, 2.0, 3.0])
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.v = np.array([0.1, 0.2, 0.3])
        self.w = np.array([0.0, 0.0, 0.1])
        self.f = np.array([0.0, 0.0, -9.81])
        self.tau = np.array([0.0, 0.0, 0.0])

class MockWorld:
    def __init__(self):
        self.t = 0.0
        self.bodies = [MockBody("body1"), MockBody("body2")]


# --- Tests ---

def test_logger_basic_io(tmp_path):
    """Test that logger creates file and writes header + data correctly."""
    log_path = tmp_path / "test_basic.csv"
    
    # Use context manager to ensure close/flush
    with CSVLogger(str(log_path), buffer_size=1) as logger:
        world = MockWorld()
        logger.log(world)
        
    assert log_path.exists()
    
    with open(log_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    # Header + 1 data row
    assert len(rows) == 2
    
    # Check header columns (default fields)
    header = rows[0]
    # t + 19 fields * 2 bodies = 1 + 38 = 39 columns
    assert header[0] == "t"
    assert "body1.p_x" in header
    assert "body2.tau_z" in header
    
    # Check data t
    assert float(rows[1][0]) == 0.0

def test_logger_buffering(tmp_path):
    """Test that data is buffered and only written when buffer fills or flush is called."""
    log_path = tmp_path / "test_buffer.csv"
    buffer_size = 5
    
    logger = CSVLogger(str(log_path), buffer_size=buffer_size)
    world = MockWorld()
    
    # 1. Log fewer items than buffer size
    for i in range(buffer_size - 1):
        world.t = float(i)
        logger.log(world)
        
    # File should exist (created in init/first log) but contain only header 
    with open(log_path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1  # Header only
    
    # 2. Log one more to trigger flush
    world.t = float(buffer_size)
    logger.log(world)
    
    with open(log_path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1 + buffer_size  # Header + buffered rows
    
    logger.close()

def test_logger_custom_fields(tmp_path):
    """Test logging with a restricted set of fields."""
    log_path = tmp_path / "test_custom.csv"
    fields = ["p", "v"] # Only position and velocity
    
    with CSVLogger(str(log_path), fields=fields) as logger:
        world = MockWorld()
        logger.log(world)
        
    with open(log_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    header = rows[0]
    # t + (p(3) + v(3)) * 2 bodies = 1 + 6*2 = 13 columns
    assert len(header) == 13
    assert header[1] == "body1.p_x"
    assert "body1.q_x" not in header

def test_simulation_integration(tmp_path):
    """Test that World automatically handles logging during run()."""
    log_path = tmp_path / "sim_auto.csv"
    
    # Create world with logging enabled
    w = World(log_enabled=True, log_file=str(log_path))
    w.add_body(RigidBody6DOF("test_body", 1.0, np.eye(3), np.zeros(3), np.array([0,0,0,1])))
    
    w.run(HybridSolver(), duration=0.1, dt=0.05)
    
    assert log_path.exists()
    with open(log_path, "r") as f:
        assert len(f.readlines()) >= 3 # Header + at least 2 steps