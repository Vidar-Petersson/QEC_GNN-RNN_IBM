import numpy as np
from sklearn.mixture import GaussianMixture

class SoftCalibrator:
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
