from __future__ import annotations
import os
import csv
from typing import Dict, Tuple, List, Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


def _load_csv(filepath: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Load a CSV produced by CSVLogger.

    Returns
    -------
    t : (N,) array
        Time vector.
    cols : dict[str, np.ndarray]
        Mapping column_name -> (N,) array.
    headers : list[str]
        Column headers in order (first one should be 't').

    Notes
    -----
    We avoid pandas; this is robust enough for the logger's well-formed CSV.
    """
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
    data = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=float)
    if data.ndim == 1:  # single row edge case
        data = data[None, :]
    cols: Dict[str, np.ndarray] = {}
    for j, name in enumerate(headers):
        cols[name] = data[:, j]
    if headers[0] != "t":
        raise ValueError("First column must be time 't'.")
    t = cols["t"]
    return t, cols, headers


def _body_fields(prefix: str) -> Dict[str, str]:
    """
    For body 'payload', build expected header names used by CSVLogger.
    """
    base = prefix
    return dict(
        p_x=f"{base}.p_x", p_y=f"{base}.p_y", p_z=f"{base}.p_z",
        q_w=f"{base}.q_w", q_x=f"{base}.q_x", q_y=f"{base}.q_y", q_z=f"{base}.q_z",
        v_x=f"{base}.v_x", v_y=f"{base}.v_y", v_z=f"{base}.v_z",
        w_x=f"{base}.w_x", w_y=f"{base}.w_y", w_z=f"{base}.w_z",
        F_x=f"{base}.F_x", F_y=f"{base}.F_y", F_z=f"{base}.F_z",
        T_x=f"{base}.T_x", T_y=f"{base}.T_y", T_z=f"{base}.T_z",
    )


def _get_components(cols: Dict[str, np.ndarray], names: Iterable[str]) -> List[np.ndarray]:
    out = []
    for name in names:
        if name not in cols:
            raise KeyError(f"Column '{name}' not found in CSV.")
        out.append(cols[name])
    return out


def _finite_diff(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Central-difference derivative with uneven spacing support via np.gradient.

    Parameters
    ----------
    t : (N,)
    y : (N,) or (N,3)

    Returns
    -------
    dy_dt : same shape as y
    """
    if y.ndim == 1:
        return np.gradient(y, t)
    elif y.ndim == 2 and y.shape[1] == 3:
        return np.column_stack([np.gradient(y[:, k], t) for k in range(3)])
    else:
        raise ValueError("y must be shape (N,) or (N,3).")


def plot_trajectory_3d(
    csv_path: str,
    body_name: str,
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """
    Plot 3D trajectory (x,y,z) of a given body and z(t) subplot.

    Parameters
    ----------
    csv_path : str
        Path to logger CSV.
    body_name : str
        The 'name' used when creating the body (e.g., 'payload').
    save_path : str | None
        If given, save the figure to this path (png/svg).
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : Figure
    """
    t, cols, _ = _load_csv(csv_path)
    f = _body_fields(body_name)
    px, py, pz = _get_components(cols, [f["p_x"], f["p_y"], f["p_z"]])

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    axz = fig.add_subplot(gs[1, :])

    # 3D path
    ax3d.plot(px, py, pz, lw=2.0, color="#1a73e8")
    ax3d.scatter(px[0], py[0], pz[0], color="#34a853", s=40, label="start")
    ax3d.scatter(px[-1], py[-1], pz[-1], color="#ea4335", s=40, label="end")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title(f"3D trajectory — {body_name}")
    ax3d.legend(loc="best")

    # z(t)
    axz.plot(t, pz, color="#1a73e8", lw=2)
    axz.set_xlabel("t [s]"); axz.set_ylabel("z [m]")
    axz.grid(True, alpha=0.3)
    axz.set_title("Altitude vs time")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_velocity_and_acceleration(
    csv_path: str,
    body_name: str,
    save_path: str | None = None,
    show: bool = True,
    magnitude: bool = True,
) -> Figure:
    """
    Plot velocity components & magnitude, and acceleration components & magnitude
    (acceleration via finite difference of velocity).

    Parameters
    ----------
    csv_path : str
    body_name : str
    save_path : str | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    t, cols, _ = _load_csv(csv_path)
    f = _body_fields(body_name)
    vx, vy, vz = _get_components(cols, [f["v_x"], f["v_y"], f["v_z"]])
    V = np.column_stack([vx, vy, vz])
    speed = np.linalg.norm(V, axis=1)

    A = _finite_diff(t, V)  # (N,3)
    ax, ay, az = A[:, 0], A[:, 1], A[:, 2]
    amag = np.linalg.norm(A, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Velocity
    axes[0].plot(t, vx, label="v_x", color="#1a73e8")
    axes[0].plot(t, vy, label="v_y", color="#34a853")
    axes[0].plot(t, vz, label="v_z", color="#fbbc05")
    if magnitude:
        axes[0].plot(t, speed, label="|v|", color="#ea4335", lw=2.0, alpha=0.8)
    axes[0].set_ylabel("velocity [m/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[0].set_title(f"Velocity — {body_name}")

    # Acceleration
    axes[1].plot(t, ax, label="a_x", color="#1a73e8")
    axes[1].plot(t, ay, label="a_y", color="#34a853")
    axes[1].plot(t, az, label="a_z", color="#fbbc05")
    if magnitude:
        axes[1].plot(t, amag, label="|a|", color="#ea4335", lw=2.0, alpha=0.8)
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("accel [m/s²]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].set_title("Acceleration (finite diff of v)")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_forces(
    csv_path: str,
    body_name: str,
    save_path: str | None = None,
    show: bool = True,
    magnitude: bool = True
) -> Figure:
    """
    Plot resultant force components (from logger) and magnitude.

    Parameters
    ----------
    csv_path : str
    body_name : str
    save_path : str | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    t, cols, _ = _load_csv(csv_path)
    f = _body_fields(body_name)
    Fx, Fy, Fz = _get_components(cols, [f["F_x"], f["F_y"], f["F_z"]])
    F = np.column_stack([Fx, Fy, Fz])
    Fmag = np.linalg.norm(F, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, Fx, label="F_x", color="#1a73e8")
    ax.plot(t, Fy, label="F_y", color="#34a853")
    ax.plot(t, Fz, label="F_z", color="#fbbc05")
    if magnitude:
        ax.plot(t, Fmag, label="|F|", color="#ea4335", lw=2.0, alpha=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("force [N]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(f"Resultant force — {body_name}")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig
