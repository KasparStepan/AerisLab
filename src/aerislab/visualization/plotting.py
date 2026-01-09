"""
Visualization utilities for simulation results.

Provides functions to create standard plots from CSV log files:
- 3D trajectory plots
- Velocity and acceleration time series
- Force and torque time series
- Comparison plots for multiple simulations
"""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
from typing import Dict, List, Iterable, Optional


def _load_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load CSV and return dictionary of column name -> numpy array.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping column names to data arrays
    """
    df = pd.read_csv(csv_path)
    return {col: df[col].values for col in df.columns}


def _get_components(cols: Dict[str, np.ndarray], names: Iterable[str]) -> List[np.ndarray]:
    """
    Extract named columns from dictionary.
    
    Parameters
    ----------
    cols : Dict[str, np.ndarray]
        Column dictionary
    names : Iterable[str]
        Column names to extract
        
    Returns
    -------
    List[np.ndarray]
        List of data arrays
        
    Raises
    ------
    KeyError
        If a requested column is not found
    """
    out = []
    for name in names:
        if name not in cols:
            raise KeyError(f"Column '{name}' not found in CSV.")
        out.append(cols[name])
    return out


def plot_trajectory_3d(
    csv_path: str,
    body_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot 3D trajectory of a body.
    
    Creates a 3D plot showing the spatial path of the body with start and
    end points marked.
    
    Parameters
    ----------
    csv_path : str
        Path to simulation CSV file
    body_name : str
        Name of the body to plot
    save_path : str | None
        If provided, save figure to this path
    show : bool
        If True, display the plot interactively
    figsize : tuple
        Figure size (width, height) in inches
        
    Examples
    --------
    >>> plot_trajectory_3d(
    ...     "output/my_sim/logs/simulation.csv",
    ...     "payload",
    ...     save_path="trajectory.png"
    ... )
    """
    cols = _load_csv(csv_path)
    
    # Get position components
    p = {k: f"{body_name}.{k}" for k in ["p_x", "p_y", "p_z"]}
    px, py, pz = _get_components(cols, [p["p_x"], p["p_y"], p["p_z"]])
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(px, py, pz, 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')
    
    # Mark start and end
    ax.scatter([px[0]], [py[0]], [pz[0]], c='green', s=100, marker='o', 
               label='Start', zorder=5)
    ax.scatter([px[-1]], [py[-1]], [pz[-1]], c='red', s=100, marker='x', 
               label='End', zorder=5)
    
    # Labels and formatting
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'3D Trajectory: {body_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio for better visualization
    max_range = np.array([px.max()-px.min(), py.max()-py.min(), 
                          pz.max()-pz.min()]).max() / 2.0
    mid_x = (px.max()+px.min()) * 0.5
    mid_y = (py.max()+py.min()) * 0.5
    mid_z = (pz.max()+pz.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()


def plot_velocity_and_acceleration(
    csv_path: str,
    body_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    magnitude: bool = True,
    figsize: tuple = (12, 8),
) -> None:
    """
    Plot velocity and acceleration time series.
    
    Creates two subplots showing velocity and acceleration over time.
    Can plot either magnitude or individual components.
    
    Parameters
    ----------
    csv_path : str
        Path to simulation CSV file
    body_name : str
        Name of the body to plot
    save_path : str | None
        If provided, save figure to this path
    show : bool
        If True, display the plot interactively
    magnitude : bool
        If True, plot magnitudes. If False, plot x/y/z components.
    figsize : tuple
        Figure size (width, height) in inches
        
    Examples
    --------
    >>> plot_velocity_and_acceleration(
    ...     "output/my_sim/logs/simulation.csv",
    ...     "payload",
    ...     magnitude=False  # Show components
    ... )
    """
    cols = _load_csv(csv_path)
    t = cols["t"]
    
    # Get velocity and acceleration components
    v = {k: f"{body_name}.{k}" for k in ["v_x", "v_y", "v_z"]}
    vx, vy, vz = _get_components(cols, [v["v_x"], v["v_y"], v["v_z"]])
    
    # Calculate acceleration from velocity (numerical derivative)
    dt = np.diff(t)
    ax_arr = np.gradient(vx, t)
    ay_arr = np.gradient(vy, t)
    az_arr = np.gradient(vz, t)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    if magnitude:
        # Plot magnitudes
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        a_mag = np.sqrt(ax_arr**2 + ay_arr**2 + az_arr**2)
        
        ax1.plot(t, v_mag, 'b-', linewidth=2, label='|v|')
        ax1.set_ylabel('Velocity Magnitude [m/s]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t, a_mag, 'r-', linewidth=2, label='|a|')
        ax2.set_ylabel('Acceleration Magnitude [m/s²]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Plot components
        ax1.plot(t, vx, 'r-', linewidth=1.5, label='vx', alpha=0.8)
        ax1.plot(t, vy, 'g-', linewidth=1.5, label='vy', alpha=0.8)
        ax1.plot(t, vz, 'b-', linewidth=1.5, label='vz', alpha=0.8)
        ax1.set_ylabel('Velocity [m/s]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t, ax_arr, 'r-', linewidth=1.5, label='ax', alpha=0.8)
        ax2.plot(t, ay_arr, 'g-', linewidth=1.5, label='ay', alpha=0.8)
        ax2.plot(t, az_arr, 'b-', linewidth=1.5, label='az', alpha=0.8)
        ax2.set_ylabel('Acceleration [m/s²]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    ax1.set_title(f'Kinematics: {body_name}')
    ax2.set_xlabel('Time [s]')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()


def plot_forces(
    csv_path: str,
    body_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    magnitude: bool = True,
    figsize: tuple = (12, 8),
) -> None:
    """
    Plot forces and torques acting on a body.
    
    Creates two subplots showing forces and torques over time.
    Can plot either magnitude or individual components.
    
    Parameters
    ----------
    csv_path : str
        Path to simulation CSV file
    body_name : str
        Name of the body to plot
    save_path : str | None
        If provided, save figure to this path
    show : bool
        If True, display the plot interactively
    magnitude : bool
        If True, plot magnitudes. If False, plot x/y/z components.
    figsize : tuple
        Figure size (width, height) in inches
        
    Examples
    --------
    >>> plot_forces(
    ...     "output/my_sim/logs/simulation.csv",
    ...     "payload",
    ...     magnitude=False  # Show components
    ... )
    """
    cols = _load_csv(csv_path)
    t = cols["t"]
    
    # Get force components - USE LOWERCASE f_x, f_y, f_z
    f = {k: f"{body_name}.{k}" for k in ["f_x", "f_y", "f_z"]}
    Fx, Fy, Fz = _get_components(cols, [f["f_x"], f["f_y"], f["f_z"]])
    
    # Get torque components
    tau = {k: f"{body_name}.{k}" for k in ["tau_x", "tau_y", "tau_z"]}
    Tx, Ty, Tz = _get_components(cols, [tau["tau_x"], tau["tau_y"], tau["tau_z"]])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    if magnitude:
        # Plot magnitudes
        F_mag = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        T_mag = np.sqrt(Tx**2 + Ty**2 + Tz**2)
        
        ax1.plot(t, F_mag, 'b-', linewidth=2, label='|F|')
        ax1.set_ylabel('Force Magnitude [N]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t, T_mag, 'r-', linewidth=2, label='|τ|')
        ax2.set_ylabel('Torque Magnitude [N·m]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Plot components
        ax1.plot(t, Fx, 'r-', linewidth=1.5, label='Fx', alpha=0.8)
        ax1.plot(t, Fy, 'g-', linewidth=1.5, label='Fy', alpha=0.8)
        ax1.plot(t, Fz, 'b-', linewidth=1.5, label='Fz', alpha=0.8)
        ax1.set_ylabel('Force [N]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t, Tx, 'r-', linewidth=1.5, label='τx', alpha=0.8)
        ax2.plot(t, Ty, 'g-', linewidth=1.5, label='τy', alpha=0.8)
        ax2.plot(t, Tz, 'b-', linewidth=1.5, label='τz', alpha=0.8)
        ax2.set_ylabel('Torque [N·m]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    ax1.set_title(f'Forces and Torques: {body_name}')
    ax2.set_xlabel('Time [s]')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()


def compare_trajectories(
    csv_paths: List[str],
    body_name: str,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: tuple = (10, 8),
) -> None:
    """
    Compare 3D trajectories from multiple simulations.
    
    Plots multiple trajectories on the same 3D axes for comparison.
    Useful for parameter studies or comparing solver methods.
    
    Parameters
    ----------
    csv_paths : List[str]
        List of paths to CSV files to compare
    body_name : str
        Name of the body to plot (must exist in all CSVs)
    labels : List[str] | None
        Labels for each trajectory. If None, uses filenames.
    save_path : str | None
        If provided, save figure to this path
    show : bool
        If True, display the plot interactively
    figsize : tuple
        Figure size (width, height) in inches
        
    Examples
    --------
    >>> compare_trajectories(
    ...     ["sim1/logs/simulation.csv", "sim2/logs/simulation.csv"],
    ...     "payload",
    ...     labels=["Case 1", "Case 2"]
    ... )
    """
    if labels is None:
        labels = [Path(p).parent.parent.name for p in csv_paths]
    
    if len(labels) != len(csv_paths):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of CSV files ({len(csv_paths)})")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(csv_paths)))
    
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        cols = _load_csv(csv_path)
        
        # Get position components
        p = {k: f"{body_name}.{k}" for k in ["p_x", "p_y", "p_z"]}
        px, py, pz = _get_components(cols, [p["p_x"], p["p_y"], p["p_z"]])
        
        # Plot trajectory
        ax.plot(px, py, pz, linewidth=2, alpha=0.7, label=label, color=colors[i])
        
        # Mark start
        ax.scatter([px[0]], [py[0]], [pz[0]], c=[colors[i]], s=100, marker='o', zorder=5)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Trajectory Comparison: {body_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    if show:
        plt.show()
    else:
        plt.close()
