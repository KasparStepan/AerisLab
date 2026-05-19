from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).parent / "Permeability_MIL-C-7020-III.csv"
IMAGE_FILE = Path(__file__).parent / "permeability_fit.png"

FABRIC_THICKNESS = 6*10**-5 # m, thickness of fabric used on Proto Parachute
AIR_DYNAMIC_VISCOSITY = 1.81*10**-5 # kg/(m·s), dynamic viscosity of air at room temperature
AIR_DENSITY = 1.225 # kg/m³, density of air at sea level

def fit_quadratic(x: np.ndarray, y: np.ndarray, through_origin: bool = True) -> np.ndarray:

    if through_origin:
        A = np.column_stack([x**2, x])
        a,b = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.array([a, b, 0])
    return np.polyfit(x, y, 2)


def fit_linear(x: np.ndarray, y: np.ndarray, through_origin: bool = True) -> np.ndarray:

    if through_origin:
        A = x.reshape(-1, 1)
        b = np.linalg.lstsq(A, y, rcond=None)[0][0]
        return np.array([b, 0])
    return np.polyfit(x, y, 1)


def fit_cubic(x: np.ndarray, y: np.ndarray, through_origin: bool = True) -> np.ndarray:

    if through_origin:
        A = np.column_stack([x**3, x**2, x])
        a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.array([a, b, c, 0])
    return np.polyfit(x, y, 3)


def fit_forchheimer(v: np.ndarray, dP: np.ndarray) -> tuple[float, float]:
    """Fit ΔP = α·v + β·v² (through origin, physical Darcy–Forchheimer form).

    Parameters
    ----------
    v  : superficial velocity samples [m/s]
    dP : differential pressure samples [Pa]

    Returns
    -------
    alpha : [Pa·s/m]   viscous coefficient   (= μ·t / K)
    beta  : [Pa·s²/m²] inertial coefficient  (= ρ·t·C_F / √K)
    """
    A = np.column_stack([v, v**2])
    alpha, beta = np.linalg.lstsq(A, dP, rcond=None)[0]
    return alpha, beta



def root_mean_square_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0-ss_res/ss_tot

def plot_comparison(
    x: np.ndarray,
    y: np.ndarray,
    a: float, b: float, c: float,
    m: float, n: float,
    p: float, q: float, r: float, s: float,
    alpha: float, beta: float,
) -> None:
    plt.scatter(x, y, label="Data", color="blue")
    x_fit = np.linspace(min(x), max(x), 100)
    y_quad = a * x_fit**2 + b * x_fit + c
    y_lin = m * x_fit + n
    y_cub = p * x_fit**3 + q * x_fit**2 + r * x_fit + s
    # Invert ΔP = α·v + β·v²  →  v = (-α + √(α² + 4β·ΔP)) / (2β)
    y_df = (-alpha + np.sqrt(alpha**2 + 4 * beta * x_fit)) / (2 * beta)
    plt.plot(x_fit, y_quad, label="Quadratic Fit", color="red")
    plt.plot(x_fit, y_lin, label="Linear Fit", color="green", linestyle="--")
    plt.plot(x_fit, y_cub, label="Cubic Fit", color="purple", linestyle=":")
    plt.plot(x_fit, y_df, label="Darcy–Forchheimer Fit", color="black", linewidth=2)
    plt.xlabel("Differential Pressure (Pa)")
    plt.ylabel("Air Permeability (m/s)")
    plt.title("Air Permeability vs. Differential Pressure")
    plt.yscale("log")
    plt.legend()
    plt.grid(which="both")
    plt.savefig(IMAGE_FILE)
    
    


def main() -> None:
    df = pd.read_csv(DATA_FILE, sep=";", decimal=",")
    x = df["Differential Pressure Pa"].to_numpy()
    y = df["Air Permeability m/s"].to_numpy()

    a, b, c = fit_quadratic(x, y, through_origin=True)
    y_pred_quad = a * x**2 + b * x + c

    m, n = fit_linear(x, y, through_origin=True)
    y_pred_lin = m * x + n

    p, q, r, s = fit_cubic(x, y, through_origin=True)
    y_pred_cub = p * x**3 + q * x**2 + r * x + s

    # Physical Darcy–Forchheimer fit: ΔP = α·v + β·v²  (LS-DYNA SI units)
    alpha, beta = fit_forchheimer(y, x)
    dP_pred_df = alpha * y + beta * y**2

    permeability = AIR_DYNAMIC_VISCOSITY * FABRIC_THICKNESS / alpha       # K [m²]
    forchheimer_cf = beta * np.sqrt(permeability) / (AIR_DENSITY * FABRIC_THICKNESS)  # C_F [-]

    print(f"Quadratic:  v(ΔP) = {a:.6e}·ΔP² + {b:.6e}·ΔP + {c:.6e}")
    print(f"  R²              : {root_mean_square_error(y, y_pred_quad):.6f}")
    print(f"Linear   :  v(ΔP) = {m:.6e}·ΔP + {n:.6e}")
    print(f"  R²              : {root_mean_square_error(y, y_pred_lin):.6f}")
    print(f"Cubic    :  v(ΔP) = {p:.6e}·ΔP³ + {q:.6e}·ΔP² + {r:.6e}·ΔP + {s:.6e}")
    print(f"  R²              : {root_mean_square_error(y, y_pred_cub):.6f}")
    print("Darcy–Forchheimer: ΔP = α·v + β·v²")
    print(f"  α (viscous)     : {alpha:.6e} Pa·s/m")
    print(f"  β (inertial)    : {beta:.6e} Pa·s²/m²")
    print(f"  R²              : {root_mean_square_error(x, dP_pred_df):.6f}")
    print("LS-DYNA ICFD parameters (SI):")
    print(f"  Permeability K  : {permeability:.6e} m²")
    print(f"  Forchheimer C_F : {forchheimer_cf:.6e} (dimensionless)")
    plot_comparison(x, y, a, b, c, m, n, p, q, r, s, alpha, beta)


if __name__ == "__main__":
    main()

