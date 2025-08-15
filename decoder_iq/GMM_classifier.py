import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class GMMClassifier:
    """
    SoftCalibrator trains a separate Gaussian Mixture Model (GMM) per detector
    on complex IQ calibration data and computes both P_soft (soft misclassification probability)
    and hard binary labels (0 or 1) for arbitrary IQ measurements.

    Attributes:
        n_detectors: Number of detectors
        models: List of dicts {'gmm': GaussianMixture, 'label_map': np.ndarray}
    """

    def __init__(self, calibration_data: np.ndarray):
        """
        Initialize the calibrator with calibration IQ data.

        Args:
            calibration_data: np.ndarray of shape (R, S, D), where
                R = number of calibration rounds,
                S = number of shots per round,
                D = number of detectors.
                Each entry is a complex IQ value.
        """
        if calibration_data.ndim != 3:
            raise ValueError("calibration_data must be a 3D array (rounds, shots, detectors)")

        self.R, self.S, self.n_detectors = calibration_data.shape
        self.models = []

        # Train a GMM for each detector
        for det in range(self.n_detectors):
            # Flatten all IQ measurements for this detector
            data_det = calibration_data[:, :, det].ravel()
            feats = np.column_stack([data_det.real, data_det.imag])

            # Fit a two-component Gaussian Mixture Model
            gmm = GaussianMixture(
                n_components=2,
                covariance_type='full',
                random_state=42
            )
            gmm.fit(feats)

            # Determine which component corresponds to logical 0 and 1
            centers = gmm.means_  # shape (2, 2)
            label_map = np.argsort(centers[:, 0])  # lower I -> 0, higher I -> 1

            self.models.append({'gmm': gmm, 'label_map': label_map})

    def compute_p_state(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Beräkna sannolikheten att tillståndet är 1 för varje IQ-sample.

        Args:
            iq_data: np.ndarray med form (R, S, D) med komplexa IQ-värden.

        Returns:
            p_state1: np.ndarray med form (R, S, D), med sannolikheten att tillståndet = 1.
        """
        if iq_data.ndim != 3:
            raise ValueError("iq_data must be a 3D array (rounds, shots, detectors)")

        R, S, D = iq_data.shape
        flat = iq_data.reshape(-1, D)
        p_state1_flat = np.zeros_like(flat, dtype=float)

        for det in range(D):
            feats = np.column_stack([flat[:, det].real, flat[:, det].imag])
            probs = self.models[det]['gmm'].predict_proba(feats)

            # Hämta komponent-index för state = 1
            comp_for_state1 = self.models[det]['label_map'][1]

            p_state1_flat[:, det] = probs[:, comp_for_state1]

        return p_state1_flat.reshape(R, S, D)

    def compute_p_soft(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Compute the soft misclassification probability P_soft for each IQ sample.

        Args:
            iq_data: np.ndarray of shape (R, S, D) containing complex IQ values.

        Returns:
            p_soft: np.ndarray of shape (R, S, D) giving the soft probability
                     1 - max posterior probability across components per detector.
        """
        if iq_data.ndim != 3:
            raise ValueError("iq_data must be a 3D array (rounds, shots, detectors)")

        R, S, D = iq_data.shape
        flat = iq_data.reshape(-1, D)
        p_soft_flat = np.zeros_like(flat, dtype=float)

        # Compute soft probabilities per detector
        for det in range(D):
            feats = np.column_stack([flat[:, det].real, flat[:, det].imag])
            probs = self.models[det]['gmm'].predict_proba(feats)
            # P_soft = 1 - max posterior across the sorted label_map
            p_soft_flat[:, det] = 1 - np.max(
                probs[:, self.models[det]['label_map']], axis=1
            )

        return p_soft_flat.reshape(R, S, D)

    def infer_hard(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Infer hard binary labels (0 or 1) from IQ data.

        Args:
            iq_data: np.ndarray of shape (R, S, D) containing complex IQ values.

        Returns:
            hard_labels: np.ndarray of shape (R, S, D) with entries 0 or 1.
        """
        if iq_data.ndim != 3:
            raise ValueError("iq_data must be a 3D array (rounds, shots, detectors)")

        R, S, D = iq_data.shape
        flat = iq_data.reshape(-1, D)
        hard_flat = np.zeros_like(flat, dtype=int)

        # Compute hard assignments per detector
        for det in range(D):
            feats = np.column_stack([flat[:, det].real, flat[:, det].imag])
            comps = self.models[det]['gmm'].predict(feats)

            # Map GMM component indices to {0,1}
            inv_map = np.zeros(2, dtype=int)
            inv_map[self.models[det]['label_map'][0]] = 0
            inv_map[self.models[det]['label_map'][1]] = 1

            hard_flat[:, det] = inv_map[comps]

        return hard_flat.reshape(R, S, D)

    def visualize_iq_with_psoft(self, iq_data: np.ndarray, p_soft: np.ndarray, detector_index=0, max_points=1000):
        """
        Visualize IQ data colored by p_soft for a given detector.

        Args:
            iq_data: np.ndarray of shape (R, S, D) with complex IQ values.
            p_soft: np.ndarray of shape (R, S, D) with soft misclassification probabilities.
            detector_index: int, which detector to visualize.
            max_points: int, max number of points to plot.
        """
        if iq_data.ndim != 3 or p_soft.ndim != 3:
            raise ValueError("iq_data and p_soft must be 3D arrays (R, S, D)")

        iq_flat = iq_data[:, :, detector_index].ravel()
        p_soft_flat = p_soft[:, :, detector_index].ravel()

        # Limit number of points for visual clarity
        if len(iq_flat) > max_points:
            idx = np.random.choice(len(iq_flat), max_points, replace=False)
            iq_flat = iq_flat[idx]
            p_soft_flat = p_soft_flat[idx]

        # Plot IQ points with p_soft color
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(iq_flat.real, iq_flat.imag, c=p_soft_flat,
                              cmap='viridis', s=10, alpha=0.8)

        # Plot GMM centers
        centers = self.models[detector_index]['gmm'].means_
        plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='x', s=80, label='GMM centers')

        plt.xlabel("I (Real part)")
        plt.ylabel("Q (Imag part)")
        plt.title(f"IQ with $p_\\mathrm{{soft}}$ (Detector {detector_index})")
        plt.colorbar(scatter, label="$p_\\mathrm{soft}$")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()