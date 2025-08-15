from __future__ import annotations
import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Tuple, Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from args import Args
from decoder_binary.mwpm_decoder_ibm import MWPMDecoder
from data_analysis.data_characteristics import analyze_pdet_time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def epsilon_from_PL(PL: float, T: int) -> float:
    """
    Invertera PL (logisk felprob per T cykler) till epsilon per cykel.
    Clippar vid 0.5 för ogiltiga/stora PL.
    """
    if PL >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * PL) ** (1.0 / T))

def epsilon_err_from_PL(PL: float, dPL: float, T: int) -> float:
    """
    Propagera osäkerheten dPL till epsilon.
    Derivatan av epsilon(PL) är:
      dε/dPL = (1/T) * (1 - 2PL)^(1/T - 1)
    Så: deps = |deps/dPL| * dPL
    """
    if PL >= 0.5:
        return 0.0
    deriv = (1.0 / T) * (1.0 - 2.0 * PL) ** (1.0 / T - 1.0)
    return abs(deriv) * dPL

# --- IO ---
def get_default_data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "data_pittsburgh_4.csv"


def load_existing_dataframe(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        logging.info(f"Data laddad från {path} ({len(df)} rader).")
        return df
    else:
        cols = ["distance", "alpha", "order", "pdet", "eps", "eps_err"]
        logging.info("Ingen befintlig fil hittad. Skapar tom DataFrame.")
        return pd.DataFrame(columns=cols)


def append_row_to_csv(path: Path, row: Dict[str, Any], write_header: bool) -> None:
    one_row_df = pd.DataFrame([row])
    one_row_df.to_csv(path, mode="a", header=write_header, index=False)


# --- datainsamling ---
def measure_single_run(distance: int, alpha: float, rounds: int) -> Tuple[float, float]:
    """
    Skapa decoder, kör decode och returnera (pdet_mean, eps, eps_err).
    Returnerar endast eps/fel och pdet; anropande kod bygger rad-dict. (Så att den kan fortsätta köra efter ev. krasch)
    """
    args = Args(
        t=rounds,
        distance=distance,
        noise_angle=alpha,
        simulator_backend=False,
        load_distance=None,
        sub_dir="/noise_angle_pittsburgh_2",
        val_fraction=1,
    )
    decoder = MWPMDecoder(args, weight_scheme="p_ij")
    logical_accuracy, logical_accuracy_err = decoder.decode()
    PL = 1.0 - logical_accuracy
    dPL = logical_accuracy_err

    eps = epsilon_from_PL(PL, rounds)
    eps_err = epsilon_err_from_PL(PL, dPL, rounds)
    pdet_mean = analyze_pdet_time(decoder.val_detections)
    print(f"pdet: {pdet_mean}")
    test = decoder.val_detections.mean()
    print(f"test: {test}")

    return pdet_mean, eps, eps_err

def run_data_collection(
    distances: List[int],
    alphas: List[float],
    rounds: int,
    data_csv: Path,
) -> pd.DataFrame:
    """
    Loopar över distances x alphas och sparar nya rader i CSV löpande.
    Returnerar komplett DataFrame (existerande + nya).
    """
    df = load_existing_dataframe(data_csv)
    file_exists = data_csv.exists()
    existing = set(zip(df["distance"].astype(float), df["alpha"].astype(float)))

    # säkerställ att data-mappen finns
    data_csv.parent.mkdir(parents=True, exist_ok=True)

    progress = tqdm([(d, a) for d in distances for a in alphas], desc="Mäter", unit="run")

    for d, alpha in progress:
        if (float(d), float(alpha)) in existing:
            continue

        try:
            pdet_mean, eps, eps_err = measure_single_run(d, alpha, rounds)
        except Exception as e:
            logging.exception(f"Fel vid mätning d={d}, alpha={alpha}: {e}")
            continue

        row = {
            "distance": int(d),
            "alpha": float(alpha),
            "order": int((d // 2) + 1),
            "pdet": float(pdet_mean),
            "eps": float(eps),
            "eps_err": float(eps_err),
        }

        append_row_to_csv(data_csv, row, write_header=not file_exists)
        file_exists = True
        existing.add((float(d), float(alpha)))
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        logging.info(
            f"Ny data: d={d}, alpha={alpha:.4f}, pdet={pdet_mean:.2e}, eps={eps:.2e} (sparad)."
        )

    return df

# --- Plotting & analys ---
def prepare_grouped_data(df: pd.DataFrame, distances: List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    # Filtrera ut oönskade eps (maxvärde 0.5)
    df_clean = df[df["eps"] != 0.5].copy()
    groups = df_clean.groupby("distance")
    pdet_data = {}
    eps_data = {}
    eps_err_data = {}
    for d in distances:
        if d in groups.groups:
            g = groups.get_group(d)
            pdet_data[d] = g["pdet"].values
            eps_data[d] = g["eps"].values
            eps_err_data[d] = g["eps_err"].values
        else:
            pdet_data[d] = np.array([])
            eps_data[d] = np.array([])
            eps_err_data[d] = np.array([])
    return pdet_data, eps_data, eps_err_data


def plot_error_vs_pdet(
    pdet_data: Dict[int, np.ndarray],
    eps_data: Dict[int, np.ndarray],
    eps_err_data: Dict[int, np.ndarray],
    distances: List[int],
) -> Tuple[List[float], List[float], List[int]]:
    """Plottar epsilon vs p_det på log-log och returnerar uppmätta och teoretiska slopes."""
    m_measured: List[float] = []
    m_theoretical: List[float] = []
    d_list: List[int] = []

    plt.figure()
    n_colors = len(distances)
    cmap = plt.get_cmap('tab20', len(distances))
    # cmap = plt.cm.get_cmap("turbo", n_colors)
    colors = [cmap(i) for i in range(n_colors)]

    for idx, d in enumerate(distances):
        x = pdet_data.get(d, np.array([]))
        y = eps_data.get(d, np.array([]))
        y_err = eps_err_data.get(d, np.array([]))
        if x.size == 0:
            continue

        eb = plt.errorbar(x, y, yerr=y_err, fmt="o", label=f"d={d}", color=colors[idx])
        # sänk transparensen på errorbars
        [bar.set_alpha(0.3) for bar in eb[2]]

        # mask för giltiga punkter för fit
        mask = (x > 0) & (y > 0) & (x < 0.3)
        if np.sum(mask) > 1:
            # log-log fit: ln y = m ln x + b
            with np.errstate(all="ignore"):
                m, b = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
            m_measured.append(float(m))
            m_theoretical.append(float((d + 1) / 2.0))
            d_list.append(d)

            fit_y = np.exp(b) * x ** m
            plt.plot(x[mask], fit_y[mask], linestyle="--", color=colors[idx], alpha=0.8)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Medelvärde $p_{det}$")
    plt.ylabel(r"Logisk fel per cykel $\epsilon$")
    plt.legend(title="Kodavstånd", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="dotted")
    plt.tight_layout()
    plt.show()
    return m_measured, m_theoretical, d_list


def fit_slope_vs_distance(d_list: List[int], m_measured: List[float], out_path: Optional[Path] = None) -> Tuple[float, float]:

    d_arr = np.array(d_list)
    m_arr = np.array(m_measured)
    a_fit, b_fit = np.polyfit(d_arr, m_arr, 1)

    d_fit = np.linspace(min(d_arr), max(d_arr), 100)
    m_fit = a_fit * d_fit + b_fit

    plt.figure()
    plt.scatter(d_arr, m_arr, color="k", marker="x")
    plt.plot(d_fit, m_fit, "r--", label=fr"Fit: $m = {a_fit:.4f}d + {b_fit:.4f}$")
    plt.xlabel(r"Kodavstånd $d$")
    plt.ylabel(r"Uppmätt lutning")
    plt.grid(True, which="both", linestyle="dotted")
    plt.legend()
    plt.show()
    logging.info(f"Fitresultat: a = {a_fit:.6f}, b = {b_fit:.6f}")
    return float(a_fit), float(b_fit)


def compare_measured_theoretical(m_measured: List[float], m_theoretical: List[float], out_path: Optional[Path] = None) -> None:
    plt.figure()
    plt.scatter(m_theoretical, m_measured, marker="o")
    lim = [min(m_theoretical), max(m_theoretical)]
    plt.plot(lim, lim, linestyle="--")  # y=x-linje
    plt.xlabel("Teoretisk lutning (d+1)/2")
    plt.ylabel("Uppmätt lutning m")
    plt.grid(True, which="both", linestyle="dotted")
    plt.tight_layout()
    plt.show()


def plot_invLambda_vs_pdet(df: pd.DataFrame, out_path: Optional[Path] = None) -> None:
    groups_alpha = df.groupby("alpha")
    pdet_avg_list: List[float] = []
    invLambda_list: List[float] = []

    for alpha, group in groups_alpha:
        # sortera per order
        ordered = group.sort_values("order")
        epsilons = ordered["eps"].values
        orders_arr = ordered["order"].values

        mask = (epsilons > 0) & np.isfinite(epsilons)
        eps_valid = epsilons[mask]
        orders_valid = orders_arr[mask]

        if len(eps_valid) > 1:

            popt, _ = curve_fit(lambda x, m, b: m * x + b, orders_valid, np.log(eps_valid), maxfev=10000)
            m = popt[0]
            Lambda = np.exp(-m)
            pdet_avg = group["pdet"].mean()
            pdet_avg_list.append(float(pdet_avg))
            invLambda_list.append(float(1.0 / Lambda))
        else:
            logging.debug(f"alpha={alpha:.4f}: för få giltiga datapunkter för curve_fit - hoppade över.")

    if len(pdet_avg_list) == 0:
        logging.warning("Inga punkter för invLambda vs pdet-plot.")
        return

    p = np.polyfit(pdet_avg_list, invLambda_list, 1)
    fit_line = np.poly1d(p)

    plt.figure()
    plt.scatter(pdet_avg_list, invLambda_list, marker="x")

    xs = np.linspace(min(pdet_avg_list), max(pdet_avg_list), 200)
    plt.plot(xs, fit_line(xs), linestyle="solid", alpha=0.3)
    plt.xlabel(r"Genomsnittligt medelvärde $p_{det}$")
    plt.ylabel(r"$1/\Lambda$")
    plt.ylim(0, 1.2)
    plt.grid(True, which="both", linestyle="dotted")
    plt.tight_layout()
    plt.show()


# --- Main-run ---
def main():
    # parametrar för data
    distances = list(range(3, 51, 2))
    rounds = 50
    arr = np.linspace(0, np.pi / 2.0, 20, endpoint=True)
    alphas = [round(x, 4) for x in arr]

    data_csv = get_default_data_path()

    # samla in nya mätvärden och spara kontinuerligt
    df = run_data_collection(distances, alphas, rounds, data_csv)

        # --- Ny plot: pdet_mean vs alpha ---
    plt.figure()
    cmap = plt.get_cmap('tab20', len(distances))

    for idx, d in enumerate(distances):
        df_d = df[df["distance"] == d].sort_values("alpha")
        plt.plot(
            df_d["alpha"],
            df_d["pdet"],
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

    # --- Analys & plottning ---
    pdet_data, eps_data, eps_err_data = prepare_grouped_data(df, distances)
    m_measured, m_theoretical, d_list = plot_error_vs_pdet(pdet_data, eps_data, eps_err_data, distances)

    fit_slope_vs_distance(d_list, m_measured)

    compare_measured_theoretical(m_measured, m_theoretical)
    plot_invLambda_vs_pdet(df)

if __name__ == "__main__":
    main()