import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def compute_ece(probs, labels, n_bins=10):
    """Beräkna Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        # Index för prediktioner i bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].float().mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def plot_reliability_diagram(probs, labels, n_bins=10):
    """Rita kalibreringsdiagram (reliability diagram)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    accuracies = []
    confidences = []
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if in_bin.sum() > 0:
            accuracies.append(labels[in_bin].float().mean().item())
            confidences.append(probs[in_bin].mean().item())
        else:
            accuracies.append(0)
            confidences.append(0)

    plt.figure(figsize=(6,6))
    plt.plot(bin_centers, accuracies, marker='o', label='Accuracy')
    plt.plot(bin_centers, confidences, marker='x', label='Confidence')
    plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
    plt.xlabel("Predicted confidence")
    plt.ylabel("Actual accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_model_confidence(final_preds: torch.Tensor, last_labels: torch.Tensor):
    """
    Plottar histogram över modellens förtroende (confidence) på prediktionerna.
    Confidence definieras som sannolikheten för den klass som modellen valde.
    
    Args:
        final_preds (torch.Tensor): Modellens sannolikheter för klassen 1, shape (N,)
        last_labels (torch.Tensor): Hårda sanna etiketter, 0 eller 1, shape (N,)
    """
    pred_labels = (final_preds >= 0.5).int()
    correct_mask = (pred_labels == last_labels)

    # Confidence = sannolikhet för den predikterade klassen
    confidence = torch.where(correct_mask, final_preds, 1 - final_preds)
    confidence_np = confidence.cpu().numpy()
    correct_mask_np = correct_mask.cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(confidence_np[correct_mask_np], bins=30, alpha=0.7, label="Correct predictions")
    plt.hist(confidence_np[~correct_mask_np], bins=30, alpha=0.7, label="Incorrect predictions")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Model confidence distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_model_confidence_with_failure_rate(final_preds: torch.Tensor, last_labels: torch.Tensor, num_bins=30):
    """
    Plottar histogram över modellens förtroende (confidence) på prediktionerna
    och visar den genomsnittliga logiska felaktighetsfrekvensen (failure rate) per bin.

    Args:
        final_preds (torch.Tensor): Modellens sannolikheter för klassen 1, shape (N,)
        last_labels (torch.Tensor): Hårda sanna etiketter, 0 eller 1, shape (N,)
        num_bins (int): Antal bins i histogrammet
    """
    pred_labels = (final_preds >= 0.5).int()
    correct_mask = (pred_labels == last_labels)

    # Confidence = sannolikhet för den predikterade klassen
    confidence = torch.where(correct_mask, final_preds, 1 - final_preds)
    
    confidence_np = confidence.cpu().numpy()
    correct_mask_np = correct_mask.cpu().numpy()
    last_labels_np = last_labels.cpu().numpy()
    pred_labels_np = pred_labels.cpu().numpy()

    # Histogram för korrekt och inkorrekt prediktioner
    plt.figure(figsize=(10, 6))

    plt.hist(confidence_np[correct_mask_np], bins=num_bins, alpha=0.6, label="Correct predictions")
    plt.hist(confidence_np[~correct_mask_np], bins=num_bins, alpha=0.6, label="Incorrect predictions")

    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Model confidence distribution with logical failure rate")

    # Beräkna failure rate per bin
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidence_np, bins) - 1  # index från 0 till num_bins-1

    failure_rates = []
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for i in range(num_bins):
        indices_in_bin = np.where(bin_indices == i)[0]
        if len(indices_in_bin) == 0:
            failure_rates.append(np.nan)  # tom bin
            continue
        # Failure rate = andel felaktiga prediktioner i bin
        failure_rate = np.sum(~correct_mask_np[indices_in_bin]) / len(indices_in_bin)
        failure_rates.append(failure_rate)

    failure_rates = np.array(failure_rates)

    # Rita failure rate som linjeplot på sekundär y-axel
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, failure_rates, color='red', marker='o', label='Average logical failure rate')
    ax2.set_ylabel("Logical failure rate")
    ax2.set_ylim(0, 1)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True)

    plt.show()
