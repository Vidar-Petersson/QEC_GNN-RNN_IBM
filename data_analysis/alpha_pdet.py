# plot_pdet_vs_alpha.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import misc.plot_settings

# Läs in CSV-filen
df = pd.read_csv("data_analysis/data.csv")

# Sortera efter alpha
df = df.sort_values(by="alpha")

# Hämta varannan unik distance
distances = sorted(df["distance"].unique())
every_other_distance = distances[::2]

# Skapa figur
plt.figure(figsize=(8, 6))

# Färgkarta
colors = plt.cm.viridis_r(np.linspace(0, 1, len(every_other_distance)))

# Plot för varje varannan distance
for i, dist in enumerate(every_other_distance):
    sub_df = df[df["distance"] == dist]
    plt.plot(
        sub_df["alpha"],
        sub_df["pdet"],
        marker=".",
        linestyle="-",
        label=f"d = {dist}",
        color=colors[i]
    )

plt.xlabel(r"Brusvinkel $\alpha$")
plt.ylabel(r"Medelvärde $p_{det}$")
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='dotted')
plt.tight_layout()
plt.savefig("pdet_vs_alpha.png", dpi=300)
plt.show()
