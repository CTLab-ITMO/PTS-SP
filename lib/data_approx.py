import os
import json
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from .utils import load_processed_data, smape, log4pl

with open("processed_result.json", "r") as file:
    processed_result = json.load(file)

# Create directory to save plots if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

data_dict_values = load_processed_data()

for dataset, x_y in data_dict_values.items():
    for i, x_y_true_raw in enumerate(x_y):
        if x_y_true_raw == ():
            continue
        x = np.array(x_y_true_raw[0])
        y_true = [x_y_true_raw[1][item][0] for item in x_y_true_raw[1].keys()]

        params, _ = curve_fit(log4pl, x, y_true, maxfev=5000)
        A, B, C, D = params
        processed_result[dataset].extend([A, B, C, D])
        x_min = min(x)
        x_max = max(x)
        step = 0.01
        x_new = np.arange(x_min, x_max, step)
        yfit1_new = ((A - D) / (1.0 + ((x_new / C) ** B))) + D
        y_pred = log4pl(x, A, B, C, D)
        smape_val = smape(y_true, y_pred)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y_true, "r+", label="Actual Data")
        ax.plot(x_new, yfit1_new, label="Fitted Curve")
        ax.set_title(f'{dataset.split("/")[-1]} #{i}')
        ax.text(
            0.65,
            0.05,
            f"A={A:.4f}, B={B:.4f},\nC={C:.4f}, D={D:.4f}\nSMAPE={smape_val:.2f}%",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5},
        )
        ax.set_xlabel("Dataset size")
        ax.set_ylabel("mAP")
        ax.set_xscale("log")
        ax.grid(True)
        ax.legend(loc="best", fancybox=True, shadow=True)

        plot_filename = f"plots/{dataset.split('/')[-1]}_{i}.png"
        plt.savefig(plot_filename)
        plt.close()

with open("approx_result.json", "w") as file:
    json.dump(processed_result, file, indent=4)
