import pandas as pd
import numpy as np





def main():
    # Import of data
    df = pd.read_csv("Permeability_MIL-C-7020-III.csv", sep=";", decimal=",")

    x = df["Differential Pressure Pa"].to_numpy()
    y = df['Air Permeability m/s'].to_numpy()

    coeffs = np.polyfit(x, y, 2)
    print(f'Coefficients: {coeffs}')

if __name__ == "__main__":
    main()

