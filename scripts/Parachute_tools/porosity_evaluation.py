from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).parent / "Permeability_MIL-C-7020-III_v3.csv"
IMAGE_FILE = Path(__file__).parent / "permeability_fit.png"

FABRIC_THICKNESS = 6*10**-5 # m, thickness of fabric used on Proto Parachute
AIR_DYNAMIC_VISCOSITY = 1.81*10**-5 # kg/(m·s), dynamic viscosity of air at room temperature
E_POROSITY = 0.8 # Porosity of the fabric, estimated from literature
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



def root_mean_square_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0-ss_res/ss_tot

def plot_comparison(
    x: np.ndarray,
    y: np.ndarray,
    a: float, b: float, c: float,
    m: float, n: float,
) -> None:
    plt.scatter(x, y, label="Data", color="blue")
    x_fit = np.linspace(min(x), max(x), 100)
    y_quad = a * x_fit**2 + b * x_fit + c
    y_lin = m * x_fit + n
    plt.plot(x_fit, y_quad, label="Quadratic Fit", color="red")
    plt.plot(x_fit, y_lin, label="Linear Fit", color="green", linestyle="--")
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

    permeability = FABRIC_THICKNESS * AIR_DYNAMIC_VISCOSITY / b
    forchheimer_factor = (a * np.sqrt(permeability)) / (AIR_DYNAMIC_VISCOSITY * AIR_DENSITY * E_POROSITY)
    permeability_linear = FABRIC_THICKNESS * AIR_DYNAMIC_VISCOSITY / m

    print(f"Quadratic:  v(ΔP) = {a:.6e}·ΔP² + {b:.6e}·ΔP + {c:.6e}")
    print(f"  R²              : {root_mean_square_error(y, y_pred_quad):.6f}")
    print(f"  Permeability    : {permeability:.6e} m²")
    print(f"  Forchheimer Fac.: {forchheimer_factor:.6e} m⁻¹")
    print(f"Linear   :  v(ΔP) = {m:.6e}·ΔP + {n:.6e}")
    print(f"  R²              : {root_mean_square_error(y, y_pred_lin):.6f}")
    print(f"  Permeability    : {permeability_linear:.6e} m²")
    plot_comparison(x, y, a, b, c, m, n)


if __name__ == "__main__":
    main()

