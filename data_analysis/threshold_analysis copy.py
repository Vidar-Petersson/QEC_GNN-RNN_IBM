import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binom
from args import Args
from mwpm_decoder_ibm import MWPMDecoder
import misc.plot_settings
from data_analysis.data_characteristics import *


def epsilon_from_PL(PL, T):
    """Invert formula PL = (1 - (1 - 2ε)^T)/2 to extract ε."""
    if PL >= 0.5:
        return 0.5
    return 0.5 * (1 - (1 - 2 * PL)**(1 / T))


def propagate_epsilon_error(PL, PL_std, T):
    """Error propagation for epsilon from PL uncertainty."""
    if PL >= 0.5:
        return 0.0
    factor = (1 - 2 * PL)**(1 / T - 1)
    d_eps_d_PL = (1 / T) * factor
    return abs(d_eps_d_PL * PL_std)


def compute_threshold(distances, rounds, weight_scheme='p_ij'):
    """
    Compute logical error per round and Lambda-factor for a list of code distances.

    Parameters
    ----------
    distances : list of int
        Odd distances to evaluate.
    rounds : int
        Number of stabilizer measurement rounds T.
    weight_scheme : str
        'uniform' or 'p_ij'.
    shots : int
        Number of repeated runs to estimate logical error.

    Returns
    -------
    epsilons : np.ndarray
        Logical error per round.
    epsilon_errs : np.ndarray
        Uncertainty in each epsilon.
    Lambda : float
        Error suppression factor.
    Lambda_err : float
        Uncertainty in Lambda.
    """
    epsilons = []
    epsilon_errs = []
    orders = []
    arr = np.linspace(0, 4*np.pi/50, 11)
    alphas = [round(x, 4) for x in arr]

    for alpha in alphas:
        for d in distances:
            args = Args(t=[rounds], distance=d, noise_angle=alpha, simulator_backend=False)
            decoder = MWPMDecoder(args, weight_scheme=weight_scheme)

            start = time.perf_counter()
            logical_accuracy, logical_accuracy_err = decoder.decode()  # Fraction correct
            duration = time.perf_counter() - start
            PL = 1 - logical_accuracy
            PL_std = logical_accuracy_err

            eps = epsilon_from_PL(PL, rounds)
            eps_std = propagate_epsilon_error(PL, PL_std, rounds)

            epsilons.append(eps)
            epsilon_errs.append(eps_std)
            orders.append((d // 2) + 1)

            print(f"d={d}, PL={PL:.2e} ± {PL_std:.1e}, ε={eps:.2e} ± {eps_std:.1e}, time={duration:.2f}s")
            pdet_mean = analyze_pdet_time(decoder.detections)

    # Linear regression in log-log space
    orders = np.array(orders)
    log_eps = np.log(epsilons)
    log_eps_std = np.array(epsilon_errs) / epsilons  # Δ(log(x)) ≈ Δx / x

    def linear_model(x, m, b):
        return m * x + b

    popt, pcov = curve_fit(linear_model, orders, log_eps, sigma=log_eps_std, absolute_sigma=True)
    m, b = popt
    m_err = np.sqrt(np.diag(pcov))[0]

    Lambda = np.exp(-m)
    Lambda_err = Lambda * m_err

    fit_log_eps = m * orders + b
    fit_curve = np.exp(fit_log_eps)

    print(f"Fitted Lambda = {Lambda:.3f} ± {Lambda_err:.3f}")

    return np.array(epsilons), np.array(epsilon_errs), Lambda, Lambda_err, fit_curve


if __name__ == '__main__':
    distances = list(range(49, 51, 2))
    round_list = [50]

    for rounds in round_list:
        eps, eps_err, Lambda, Lambda_err, fit_curve = compute_threshold(distances, rounds, weight_scheme="p_ij")

        eb = plt.errorbar(
            distances,
            eps,
            yerr=eps_err,
            marker='o',
            linestyle='',
            capsize=3,
            label=fr"t={rounds}, $\Lambda$={Lambda:.2f}$\pm${Lambda_err:.2f}"
        )

        # Extract color from the first artist (the marker line)
        color = eb[0].get_color()

        # Plot the fit curve using the same color
        plt.plot(distances, fit_curve, linestyle='dashed', alpha=0.7, color=color)

    plt.xlabel(r'Code distance $d$')
    plt.ylabel(r'Logical error per round $\epsilon$')
    plt.yscale('log')
    #plt.title('MWPM Decoder Threshold Estimation')
    plt.legend()
    plt.grid(True, which='both', linestyle='dotted')
    plt.savefig("threshold.pdf")
    plt.show()