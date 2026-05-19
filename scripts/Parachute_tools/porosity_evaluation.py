from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).parent / "Permeability_MIL-C-7020-III.csv"
IMAGE_FILE = Path(__file__).parent / "permeability_fit.png"

def fit_quadratic(x: np.ndarray, y: np.ndarray, through_origin: bool = True) -> np.ndarray:

    if through_origin:
        A = np.column_stack([x**2, x])
        a,b = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.array([a, b, 0])
    return np.polyfit(x, y, 2)

def root_mean_square_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0-ss_res/ss_tot

def plot_comparison(x: np.ndarray, y: np.ndarray, a: float, b: float, c: float) -> None:
    plt.scatter(x, y, label="Data", color="blue")
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = a * x_fit**2 + b * x_fit + c
    plt.plot(x_fit, y_fit, label="Quadratic Fit", color="red")
    plt.xlabel("Differential Pressure (Pa)")
    plt.ylabel("Air Permeability (m/s)")
    plt.title("Air Permeability vs. Differential Pressure")
    plt.legend()
    plt.grid()
    plt.savefig(IMAGE_FILE)
    
    


def main() -> None:
    df = pd.read_csv(DATA_FILE, sep=";", decimal=",")
    x = df["Differential Pressure Pa"].to_numpy()
    y = df["Air Permeability m/s"].to_numpy()

    a, b, c = fit_quadratic(x, y, through_origin=True)
    y_pred = a * x**2 + b * x + c

    print(f"Model:  v(ΔP) = {a:.6e}·ΔP² + {b:.6e}·ΔP + {c:.6e}")
    print(f"R²    : {root_mean_square_error(y, y_pred):.6f}")
    plot_comparison(x, y, a, b, c)


if __name__ == "__main__":
    main()

