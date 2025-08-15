import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

def trivial(detections):
    trivial_syndrome_mask = np.any(detections, axis=1)
    pct_trivial = np.mean(~trivial_syndrome_mask) * 100
    n = detections.shape[0]
    stderr = np.std(~trivial_syndrome_mask, ddof=1) / np.sqrt(n) * 100
    print(f"Andel triviala: {pct_trivial:.1f}% ± {stderr:.1f}%")


def analyze_class_balance(train_flips, val_flips=None):
    train_flips = train_flips[:, -1]
    train_ones = np.sum(train_flips)
    train_zeros = train_flips.size - train_ones
    frac_train_ones = train_ones / train_flips.size
    stderr_train = np.sqrt(frac_train_ones * (1 - frac_train_ones) / train_flips.size)

    print("Klassfördelning icke-triviala:")
    print(f"  [Träning]     1: {train_ones}   0: {train_zeros}   Andel 1: {frac_train_ones:.3f} ± {stderr_train:.3f}")

    if val_flips is not None:
        val_flips = val_flips[:, -1]
        val_ones = np.sum(val_flips)
        val_zeros = val_flips.size - val_ones
        frac_val_ones = val_ones / val_flips.size
        stderr_val = np.sqrt(frac_val_ones * (1 - frac_val_ones) / val_flips.size)
        print(f"  [Validering]  1: {val_ones}   0: {val_zeros}   Andel 1: {frac_val_ones:.3f} ± {stderr_val:.3f}")


def analyze_pdet_time(detections, verbose=True):
    pdet_mean = np.mean(detections)
    return pdet_mean