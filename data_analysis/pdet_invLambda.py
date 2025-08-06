import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from args import Args
from mwpm_decoder_ibm import MWPMDecoder
from data_analysis.data_characteristics import analyze_pdet_time
from scipy.optimize import curve_fit

# Function to invert PL to epsilon
def epsilon_from_PL(PL, T):
    if PL >= 0.5:
        return 0.5
    return 0.5 * (1 - (1 - 2 * PL)**(1 / T))

# Derivative for uncertainty propagation
def epsilon_err_from_PL(PL, dPL, T):
    if PL >= 0.5:
        return 0.0
    deriv = (1.0 / T) * (1 - 2 * PL)**(1.0 / T - 1)
    return abs(deriv) * dPL * 0.5 * 2

# Parametrar
distances = list(range(3, 51, 2))
rounds = 50
arr = np.linspace(0, np.pi/2, 20, endpoint=True)
alphas = [round(x, 4) for x in arr]
orders = [(d // 2) + 1 for d in distances]

data_csv = os.path.join(os.path.dirname(__file__), 'data_pittsburgh.csv')

# Läs in befintlig data om filen finns
if os.path.exists(data_csv):
    df = pd.read_csv(data_csv)
    print(f"Data laddad från {data_csv}, {len(df)} rader totalt.")
    file_exists = True
else:
    df = pd.DataFrame(columns=['distance','alpha','order','pdet','eps','eps_err'])
    print(f"Ingen befintlig fil hittad. Ny DataFrame skapad.")
    file_exists = False

# Hitta vilka kombinationer som redan finns
existing = set(zip(df['distance'], df['alpha']))

# Samla in nya mätvärden och spara kontinuerligt\ nfor d in distances:
for d in distances:    
    for alpha in alphas:
        if (d, alpha) in existing:
            continue

        # Sätt upp decoder och kör
        args = Args(t=[rounds], distance=d, noise_angle=alpha, 
                    simulator_backend=False, load_distance=49, sub_dir="/noise_angle_pittsburgh", val_fraction=1)
        decoder = MWPMDecoder(args, weight_scheme='p_ij')

        logical_accuracy, logical_accuracy_err = decoder.decode()
        PL = 1 - logical_accuracy
        dPL = logical_accuracy_err
        eps = epsilon_from_PL(PL, rounds)
        eps_err = epsilon_err_from_PL(PL, dPL, rounds)
        pdet_mean = analyze_pdet_time(decoder.detections)

        row = {
            'distance': d,
            'alpha': alpha,
            'order': (d // 2) + 1,
            'pdet': pdet_mean,
            'eps': eps,
            'eps_err': eps_err
        }
        # Skriv rad direkt till CSV
        one_row_df = pd.DataFrame([row])
        one_row_df.to_csv(data_csv, mode='a', header=not file_exists, index=False)
        file_exists = True  # header is now written
        existing.add((d, alpha))
        df = pd.concat([df, one_row_df], ignore_index=True)

        print(f"Ny data: d={d}, alpha={alpha:.4f}, pdet={pdet_mean:.2e}, eps={eps:.2e}")
        print(f"Sparat rad i {data_csv}")

# --- Resten av analys/plotting som tidigare ---

# Förbered data för plotting
groups = df.groupby('distance')
pdet_data = {d: groups.get_group(d)['pdet'].values for d in distances}
eps_data = {d: groups.get_group(d)['eps'].values for d in distances}
eps_err_data = {d: groups.get_group(d)['eps_err'].values for d in distances}

# 1) Plot p_det vs epsilon med regression
plt.figure()
cmap = plt.get_cmap('tab20', len(distances))
for idx, d in enumerate(distances):
    x = pdet_data[d]
    y = eps_data[d]
    y_err = eps_err_data[d]
    eb = plt.errorbar(x, y, yerr=y_err, fmt='o', label=f"d={d}", color=cmap(idx))
    [bar.set_alpha(0.3) for bar in eb[2]]
    mask = (x > 0) & (y > 0)
    if np.sum(mask) > 1:
        m, b = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
        fit_y = np.exp(b) * x**m
        plt.plot(x, fit_y, linestyle='--', color=cmap(idx), alpha=0.8)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Medelvärde $p_{det}$")
plt.ylabel(r"Logisk fel per cykel $\epsilon$")
plt.legend(title="Kodavstånd", bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True, which='both', linestyle='dotted')
plt.tight_layout()
plt.savefig("error_vs_pdet.png", dpi=300)
plt.show()

# 2) Plot 1/Lambda vs genomsnittligt p_det per alpha
groups_alpha = df.groupby('alpha')
pdet_avg_list = []
invLambda_list = []
for alpha, group in groups_alpha:
    epsilons = group.sort_values('order')['eps'].values
    orders_arr = group.sort_values('order')['order'].values
    popt, _ = curve_fit(lambda x, m, b: m * x + b, orders_arr, np.log(epsilons))
    m = popt[0]
    Lambda = np.exp(-m)
    pdet_avg = group['pdet'].mean()
    pdet_avg_list.append(pdet_avg)
    invLambda_list.append(1 / Lambda)
    print(f"alpha={alpha:.4f}: avg pdet={pdet_avg:.2e}, 1/Lambda={1/Lambda:.2f}")

p = np.polyfit(pdet_avg_list, invLambda_list, 1)
fit_line = np.poly1d(p)
plt.figure()
plt.scatter(pdet_avg_list, invLambda_list, marker='x')
plt.plot(pdet_avg_list, fit_line(pdet_avg_list), linestyle='solid', alpha=0.3)
plt.xlabel(r"Genomsnittligt medelvärde $p_{det}$")
plt.ylabel(r"$1/\Lambda$")
plt.ylim(0,1.2)
plt.grid(True, which='both', linestyle='dotted')
plt.tight_layout()
plt.savefig("invLambda_vs_pdet.png", dpi=300)
plt.show()
