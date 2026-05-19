from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).parent / "Permeability_MIL-C-7020-III.csv"
IMAGE_FILE = Path(__file__).parent / "permeability_fit.png"

FABRIC_THICKNESS = 6*10**-5 # m, thickness of fabric used on Proto Parachute
AIR_DYNAMIC_VISCOSITY = 1.81*10**-5 # kg/(m·s), dynamic viscosity of air at room temperature
AIR_DENSITY = 1.225 # kg/m³, density of air at sea level


def fit_linear(v: np.ndarray, dP: np.ndarray, through_origin: bool = True) -> tuple[float, float]:
    """Fit ΔP = α·v + c (pure Darcy if through_origin)."""
    if through_origin:
        A = v.reshape(-1, 1)
        alpha = np.linalg.lstsq(A, dP, rcond=None)[0][0]
        return alpha, 0.0
    alpha, c = np.polyfit(v, dP, 1)
    return alpha, c


def fit_quadratic(v: np.ndarray, dP: np.ndarray, through_origin: bool = True) -> tuple[float, float, float]:
    """Fit ΔP = α·v + β·v² + c (Darcy–Forchheimer if through_origin)."""
    if through_origin:
        A = np.column_stack([v, v**2])
        alpha, beta = np.linalg.lstsq(A, dP, rcond=None)[0]
        return alpha, beta, 0.0
    beta, alpha, c = np.polyfit(v, dP, 2)
    return alpha, beta, c


def fit_cubic(v: np.ndarray, dP: np.ndarray, through_origin: bool = True) -> tuple[float, float, float, float]:
    """Fit ΔP = α·v + β·v² + γ·v³ + c."""
    if through_origin:
        A = np.column_stack([v, v**2, v**3])
        alpha, beta, gamma = np.linalg.lstsq(A, dP, rcond=None)[0]
        return alpha, beta, gamma, 0.0
    gamma, beta, alpha, c = np.polyfit(v, dP, 3)
    return alpha, beta, gamma, c


def r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def plot_comparison(
    dP: np.ndarray,
    v: np.ndarray,
    lin: tuple[float, float],
    quad: tuple[float, float, float],
    cub: tuple[float, float, float, float],
) -> None:
    plt.scatter(dP, v, label="Data", color="blue")

    # Parametric sweep: pick a velocity grid, compute ΔP from each model.
    v_param = np.linspace(0, max(v) * 1.02, 200)

    alpha_l, c_l = lin
    alpha_q, beta_q, c_q = quad
    alpha_c, beta_c, gamma_c, c_c = cub

    dP_lin = alpha_l * v_param + c_l
    dP_quad = alpha_q * v_param + beta_q * v_param**2 + c_q
    dP_cub = alpha_c * v_param + beta_c * v_param**2 + gamma_c * v_param**3 + c_c

    plt.plot(dP_lin, v_param, label="Linear (Darcy)", color="green", linestyle="--")
    plt.plot(dP_quad, v_param, label="Quadratic (Darcy–Forchheimer)", color="black", linewidth=2)
    plt.plot(dP_cub, v_param, label="Cubic", color="purple", linestyle=":")
    plt.xlim(left=min(dP) * 0.9, right=max(dP) * 1.05)
    plt.xlabel("Differential Pressure ΔP (Pa)")
    plt.ylabel("Superficial Velocity v (m/s)")
    plt.title("Air Permeability vs. Differential Pressure")
    plt.yscale("log")
    plt.legend()
    plt.grid(which="both")
    plt.savefig(IMAGE_FILE)


def main() -> None:
    df = pd.read_csv(DATA_FILE, sep=";", decimal=",")
    dP = df["Differential Pressure Pa"].to_numpy()    # [Pa]
    v  = df["Air Permeability m/s"].to_numpy()        # [m/s]

    # All fits in the physical direction: ΔP = f(v)
    alpha_l, c_l = fit_linear(v, dP, through_origin=True)
    dP_pred_lin = alpha_l * v + c_l

    alpha_q, beta_q, c_q = fit_quadratic(v, dP, through_origin=True)
    dP_pred_quad = alpha_q * v + beta_q * v**2 + c_q

    alpha_c, beta_c, gamma_c, c_c = fit_cubic(v, dP, through_origin=True)
    dP_pred_cub = alpha_c * v + beta_c * v**2 + gamma_c * v**3 + c_c

    # LS-DYNA ICFD parameters (from the quadratic = Darcy–Forchheimer fit)
    permeability   = AIR_DYNAMIC_VISCOSITY * FABRIC_THICKNESS / alpha_q                  # K [m²]
    forchheimer_cf = beta_q * np.sqrt(permeability) / (AIR_DENSITY * FABRIC_THICKNESS)   # C_F [-]

    print("Linear (Darcy):  ΔP = α·v")
    print(f"  α               : {alpha_l:.6e} Pa·s/m")
    print(f"  R²              : {r_squared(dP, dP_pred_lin):.6f}")
    print()
    print("Quadratic (Darcy–Forchheimer):  ΔP = α·v + β·v²")
    print(f"  α (viscous)     : {alpha_q:.6e} Pa·s/m")
    print(f"  β (inertial)    : {beta_q:.6e} Pa·s²/m²")
    print(f"  R²              : {r_squared(dP, dP_pred_quad):.6f}")
    print()
    print("Cubic:  ΔP = α·v + β·v² + γ·v³")
    print(f"  α               : {alpha_c:.6e} Pa·s/m")
    print(f"  β               : {beta_c:.6e} Pa·s²/m²")
    print(f"  γ               : {gamma_c:.6e} Pa·s³/m³")
    print(f"  R²              : {r_squared(dP, dP_pred_cub):.6f}")
    print()
    print("LS-DYNA ICFD parameters (SI, from quadratic/Forchheimer fit):")
    print(f"  Permeability K  : {permeability:.6e} m²")
    print(f"  Forchheimer C_F : {forchheimer_cf:.6e} (dimensionless)")

    plot_comparison(
        dP, v,
        lin=(alpha_l, c_l),
        quad=(alpha_q, beta_q, c_q),
        cub=(alpha_c, beta_c, gamma_c, c_c),
    )


if __name__ == "__main__":
    main()
