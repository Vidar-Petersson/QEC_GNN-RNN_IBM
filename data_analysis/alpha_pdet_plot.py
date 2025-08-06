import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from args import Args
from mwpm_decoder_ibm import MWPMDecoder
import misc.plot_settings
from data_analysis.data_characteristics import analyze_pdet_time

# Define the range of alphas and distances
distances = list(range(3, 51, 4))  # exempel: [39, 43, 47, 51] (ändra vid behov)
arr = np.linspace(0, 4 * np.pi / 50, 11)
alphas = [round(x, 4) for x in arr]

# Initialize a container for pdet_means per distance
data = {d: [] for d in distances}

# Loop over angles and distances, decode and record pdet_mean
for alpha in alphas:
    for d in distances:
        args = Args(t=[50], distance=d, noise_angle=alpha, simulator_backend=False, load_distance=49)
        

        start = time.perf_counter()
        # Kör dekodern (decode) om du behöver logical_accuracy
        decoder = MWPMDecoder(args, weight_scheme='p_ij')
        logical_accuracy, logical_accuracy_err = decoder.decode()
        duration = time.perf_counter() - start

        pdet_mean = analyze_pdet_time(decoder.detections, verbose=False)
        data[d].append(pdet_mean)

        print(f"d={d}, alpha={alpha:.4f}, pdet_mean={pdet_mean:.2e}, time={duration:.2f}s")

# Skapa figur
plt.figure()
# Välj colormap med tillräckligt många distinkta färger (t.ex. 'tab20' eller 'viridis')
cmap = plt.get_cmap('tab20', len(distances))

# Plot pdet_mean vs alpha för varje kodavstånd med unika färger
for idx, d in enumerate(distances):
    plt.plot(
        alphas,
        data[d],
        marker='.',
        linestyle='-',
        label=f"d={d}",
        color=cmap(idx)
    )

plt.xlabel(r"Brusvinkel $\alpha$")
plt.ylabel(r"Medelvärde $p_{det}$")
plt.yscale('log')
plt.legend(title="Kodavstånd", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='dotted')
plt.tight_layout()
plt.savefig("pdet_vs_alpha.png", dpi=300)
plt.show()
