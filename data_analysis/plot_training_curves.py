import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import misc.plot_settings

def epsilon_from_PL(PL, T):
    PL = np.asarray(PL)
    return np.where(PL >= 0.5, 0.5, 0.5 * (1 - (1 - 2 * PL) ** (1 / T)))

def plot_training_curves(df, steps, runs, mwpm_vals, T=50):
    for run_id, label in runs:
        acc = df[f"train_final_t_d15_t50_dt2_alpha{run_id} - val_log_acc"]
        plt.plot(steps, epsilon_from_PL(1 - acc, T), label=label)
    for acc_val, label, color in mwpm_vals:
        plt.axhline(y=epsilon_from_PL(1 - acc_val, T), ls='--', c=color, label=label)

def main():
    df = pd.read_csv("./data_analysis/training_curves/wandb_export_2025-08-14T11_40_36.869+02_00.csv")
    steps = df["Step"]

    runs = [
        ("0.0_250813_131850", r"$\alpha=0$"),
        ("0.1653_250813_133742", r"$\alpha=0.1653$"),
        ("0.3307_250813_145046", r"$\alpha=0.3307$"),
    ]
    mwpm_vals = [
        (0.98393, r"MWPM ($p_{ij}$) $\alpha=0$", "tab:blue"),
        (0.98777, r"MWPM ($p_{ij}$) $\alpha=0.1653$", "tab:orange"),
        (0.98318, r"MWPM ($p_{ij}$) $\alpha=0.3307$", "tab:green"),
    ]

    plt.figure(figsize=(8, 5))
    plot_training_curves(df, steps, runs, mwpm_vals)
    plt.yscale("log")
    plt.xlabel("Epok")
    plt.ylabel("Logiskt fel per cykel")
    plt.title(r"$\epsilon$ för d=15, t=50, shots=1e6, Binär-data från IBM Pittsburg")
    plt.legend()
    plt.grid(True, which='both', ls='dotted')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()