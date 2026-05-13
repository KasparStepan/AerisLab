"""
Tests for the visualization module.

Covers:
- Both Matplotlib and Plotly backends
- All plot types (trajectory, kinematics, forces, comparison)
- Customization parameters
- Error handling
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure package import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aerislab.visualization import plotting
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dummy_csv(tmp_path):
    """Create a valid simulation CSV for testing."""
    fn = tmp_path / "simulation.csv"
    t = np.linspace(0, 10, 100)
    data = {
        "t": t,
        "payload.p_x": np.sin(t),
        "payload.p_y": np.cos(t),
        "payload.p_z": t * 0.5,
        "payload.v_x": np.cos(t),
        "payload.v_y": -np.sin(t),
        "payload.v_z": np.ones_like(t) * 0.5,
        "payload.f_x": np.sin(t) * 10,
        "payload.f_y": np.cos(t) * 10,
        "payload.f_z": np.ones_like(t) * -98.1,
        "payload.tau_x": np.zeros_like(t),
        "payload.tau_y": np.zeros_like(t),
        "payload.tau_z": np.sin(t) * 0.1,
        # Second body for comparison tests
        "canopy.p_x": np.sin(t) * 1.1,
        "canopy.p_y": np.cos(t) * 1.1,
        "canopy.p_z": t * 0.5 + 2,
        "canopy.v_x": np.cos(t) * 1.1,
        "canopy.v_y": -np.sin(t) * 1.1,
        "canopy.v_z": np.ones_like(t) * 0.5,
        "canopy.f_x": np.zeros_like(t),
        "canopy.f_y": np.zeros_like(t),
        "canopy.f_z": np.ones_like(t) * -19.6,
        "canopy.tau_x": np.zeros_like(t),
        "canopy.tau_y": np.zeros_like(t),
        "canopy.tau_z": np.zeros_like(t),
    }
    pd.DataFrame(data).to_csv(fn, index=False)
    return str(fn)


@pytest.fixture
def minimal_csv(tmp_path):
    """Create a minimal CSV with just required columns."""
    fn = tmp_path / "minimal.csv"
    data = {
        "t": [0, 1, 2],
        "body.p_x": [0, 1, 2],
        "body.p_y": [0, 0, 0],
        "body.p_z": [0, 1, 4],
    }
    pd.DataFrame(data).to_csv(fn, index=False)
    return str(fn)


# =============================================================================
# Trajectory Tests
# =============================================================================

class TestTrajectory3D:
    """Tests for plot_trajectory_3d function."""
    
    def test_matplotlib_basic(self, dummy_csv):
        """Test basic Matplotlib trajectory plot."""
        fig = plotting.plot_trajectory_3d(dummy_csv, "payload", engine="matplotlib")
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plotly_basic(self, dummy_csv):
        """Test basic Plotly trajectory plot."""
        fig = plotting.plot_trajectory_3d(dummy_csv, "payload", engine="plotly")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # trajectory + start + end markers
    
    def test_customization(self, dummy_csv):
        """Test all customization parameters."""
        fig = plotting.plot_trajectory_3d(
            dummy_csv, "payload",
            engine="plotly",
            title="Custom Title",
            xlabel="Custom X",
            ylabel="Custom Y", 
            zlabel="Custom Z",
            font_size=16,
            line_color="red",
            line_width=3.0,
            marker_size=10,
            show_grid=False,
            show_legend=False,
            figsize=(12, 10)
        )
        assert fig is not None
        assert fig.layout.title.text == "Custom Title"
    
    def test_save_matplotlib(self, dummy_csv, tmp_path):
        """Test saving with Matplotlib."""
        save_path = tmp_path / "trajectory.png"
        plotting.plot_trajectory_3d(
            dummy_csv, "payload",
            engine="matplotlib",
            save_path=str(save_path),
            dpi=100
        )
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    
    def test_invalid_body_raises(self, dummy_csv):
        """Test that invalid body name raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            plotting.plot_trajectory_3d(dummy_csv, "nonexistent")
    
    def test_minimal_data(self, minimal_csv):
        """Test with minimal valid data."""
        fig = plotting.plot_trajectory_3d(minimal_csv, "body", engine="plotly")
        assert fig is not None


# =============================================================================
# Kinematics Tests
# =============================================================================

class TestKinematics:
    """Tests for plot_velocity_and_acceleration function."""
    
    def test_matplotlib_magnitude(self, dummy_csv):
        """Test Matplotlib with magnitude mode."""
        fig = plotting.plot_velocity_and_acceleration(
            dummy_csv, "payload",
            engine="matplotlib",
            magnitude=True
        )
        assert fig is not None
        plt.close('all')
    
    def test_plotly_components(self, dummy_csv):
        """Test Plotly with component mode."""
        fig = plotting.plot_velocity_and_acceleration(
            dummy_csv, "payload",
            engine="plotly",
            magnitude=False
        )
        assert isinstance(fig, go.Figure)
        # Should have 6 traces: vx, vy, vz, ax, ay, az
        assert len(fig.data) == 6
    
    def test_customization(self, dummy_csv):
        """Test customization options."""
        fig = plotting.plot_velocity_and_acceleration(
            dummy_csv, "payload",
            engine="plotly",
            title="Velocity Test",
            xlabel="t [s]",
            font_size=18,
            line_width=3.0,
            show_grid=False,
            show_legend=False
        )
        assert fig.layout.title.text == "Velocity Test"


# =============================================================================
# Forces Tests
# =============================================================================

class TestForces:
    """Tests for plot_forces function."""
    
    def test_matplotlib_magnitude(self, dummy_csv):
        """Test Matplotlib force plot."""
        fig = plotting.plot_forces(
            dummy_csv, "payload",
            engine="matplotlib",
            magnitude=True
        )
        assert fig is not None
        plt.close('all')
    
    def test_plotly_components(self, dummy_csv):
        """Test Plotly with component mode."""
        fig = plotting.plot_forces(
            dummy_csv, "payload",
            engine="plotly",
            magnitude=False
        )
        assert isinstance(fig, go.Figure)
        # Should have 6 traces: Fx, Fy, Fz, τx, τy, τz
        assert len(fig.data) == 6
    
    def test_missing_torque_columns(self, tmp_path):
        """Test error when torque columns are missing."""
        fn = tmp_path / "no_torque.csv"
        data = {
            "t": [0, 1],
            "body.f_x": [0, 1],
            "body.f_y": [0, 1],
            "body.f_z": [0, 1],
            # Missing tau columns
        }
        pd.DataFrame(data).to_csv(fn, index=False)
        
        with pytest.raises(KeyError):
            plotting.plot_forces(str(fn), "body")


# =============================================================================
# Comparison Tests
# =============================================================================

class TestCompareTrajectories:
    """Tests for compare_trajectories function."""
    
    def test_compare_two(self, dummy_csv):
        """Test comparing same file twice (tests multi-trace logic)."""
        fig = plotting.compare_trajectories(
            [dummy_csv, dummy_csv],
            "payload",
            labels=["Run 1", "Run 2"],
            engine="plotly"
        )
        assert isinstance(fig, go.Figure)
        # 2 trajectories + 2 start markers
        assert len(fig.data) >= 4
    
    def test_matplotlib_comparison(self, dummy_csv):
        """Test Matplotlib comparison."""
        fig = plotting.compare_trajectories(
            [dummy_csv],
            "payload",
            engine="matplotlib"
        )
        assert fig is not None
        plt.close(fig)
    
    def test_label_mismatch_raises(self, dummy_csv):
        """Test that mismatched labels raise error."""
        with pytest.raises(ValueError, match="Labels"):
            plotting.compare_trajectories(
                [dummy_csv, dummy_csv],
                "payload",
                labels=["Only One"]
            )


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""
    
    def test_missing_time_column(self, tmp_path):
        """Test error when 't' column is missing."""
        fn = tmp_path / "no_time.csv"
        data = {
            "body.p_x": [0, 1],
            "body.p_y": [0, 1],
            "body.p_z": [0, 1],
        }
        pd.DataFrame(data).to_csv(fn, index=False)
        
        # Trajectory doesn't need 't', but kinematics does
        with pytest.raises(KeyError):
            plotting.plot_velocity_and_acceleration(str(fn), "body")
    
    def test_single_point_trajectory(self, tmp_path):
        """Test trajectory with single data point."""
        fn = tmp_path / "single.csv"
        data = {
            "body.p_x": [0],
            "body.p_y": [0],
            "body.p_z": [0],
        }
        pd.DataFrame(data).to_csv(fn, index=False)
        
        fig = plotting.plot_trajectory_3d(str(fn), "body", engine="plotly")
        assert fig is not None
    
    def test_empty_csv(self, tmp_path):
        """Test with empty data."""
        fn = tmp_path / "empty.csv"
        pd.DataFrame({"body.p_x": [], "body.p_y": [], "body.p_z": []}).to_csv(fn, index=False)
        
        with pytest.raises(ValueError, match="No trajectory data"):
            plotting.plot_trajectory_3d(str(fn), "body")
