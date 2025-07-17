import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import matplotlib.pyplot as plt
import numpy as np

from dataloader_ibm import IBMSampler


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


def analyze_pdet_time(detections):
    detector1 = detections.mean(axis=0)[1::2]
    detector2 = detections.mean(axis=0)[::2]
    stderr1 = detections[:, 1::2].std(axis=0, ddof=1) / np.sqrt(detections.shape[0])
    stderr2 = detections[:, ::2].std(axis=0, ddof=1) / np.sqrt(detections.shape[0])
    mean1 = np.mean(detector1)
    mean2 = np.mean(detector2)
    stderr_mean1 = np.sqrt(np.mean(stderr1 ** 2))
    stderr_mean2 = np.sqrt(np.mean(stderr2 ** 2))
    print(f"Genomsnittlig detektionssannolikhet: Detektor 1: {mean1:.4f} ± {stderr_mean1:.4f}, "
          f"Detektor 2: {mean2:.4f} ± {stderr_mean2:.4f}")


if __name__ == "__main__":
    sampler = IBMSampler(distance=3, t=50, simulator=False)
    detection_events, observable_flips = sampler.load_jobdata()
    trivial(detection_events)
    analyze_class_balance(observable_flips)
    analyze_pdet_time(detection_events)
