import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle

class HMMClassifier:
    """
    HMMDetectionProbability tränar en 2-komponents HMM per detektor
    baserat på komplex IQ-data och uppskattar sannolikheten för varje tillstånd
    vid varje tidssteg och varje shot.

    Inputdata: IQ-mätningar i form av np.ndarray med form (R, S, D)
    där R = antal tidssteg (rounds), S = antal shots, D = antal detektorer.
    """

    def __init__(self, iq_data: np.ndarray = None, n_iter: int = 30):
        """
        Om iq_data inte ges vid init, skapas en tom instans som kan laddas senare från fil.
        """
        self.models = []
        if iq_data is not None:
            if iq_data.ndim != 3:
                raise ValueError("iq_data måste ha formen (R, S, D)")

            self.R, self.S, self.D = iq_data.shape

            # Träna en HMM för varje detektor
            for d in range(self.D):
                iq_det = iq_data[:, :, d]
                feats = np.column_stack([iq_det.real.ravel(), iq_det.imag.ravel()])

                model = hmm.GaussianHMM(
                    n_components=2,
                    covariance_type="full",
                    n_iter=n_iter,
                    random_state=42,
                    tol=1e-2
                )
                model.fit(feats)

                # Bestäm vilken komponent som är state 0 och state 1
                means = model.means_  # (2, 2)
                label_map = np.argsort(means[:, 0])  # sortera efter I-värde

                self.models.append({'hmm': model, 'label_map': label_map})

    def compute_state_probabilities(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Beräkna sannolikhet för att vara i tillstånd 1 vid varje tidssteg,
        för varje shot och varje detektor.
        """
        R, S, D = iq_data.shape
        state_probs = np.zeros((R, S, D))

        for d in range(D):
            model = self.models[d]['hmm']
            label_map = self.models[d]['label_map']

            iq_det = iq_data[:, :, d]
            obs = np.stack([iq_det.real, iq_det.imag], axis=2)  # (R, S, 2)
            obs_flat = obs.reshape(-1, 2)

            # Posteriorer
            post_flat = model.predict_proba(obs_flat)  # (R*S, 2)

            # Mappa om kolumnerna så att kolumn 0 = state 0, kolumn 1 = state 1
            post_flat = post_flat[:, label_map]

            post = post_flat.reshape(R, S, 2)

            # Gör reset justering om reset=True i rep.code
            p0 = post[:-1, :, 0]
            p1 = post[:-1, :, 1]
            q0 = post[1:, :, 0]
            q1 = post[1:, :, 1]
            diff = p0 * q1 + p1 * q0

            first = post[0:1, :, 1]  # P(state=1) vid första tidssteget
            reset_adj = np.concatenate([first, diff], axis=0)

            state_probs[:, :, d] = reset_adj

        return state_probs

    def save_model(self, filepath: str):
        """
        Spara hela modellen (alla detektorer) till fil med pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Modell sparad till '{filepath}'")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Ladda modell från fil och returnera en instans av HMMDetectionProbability.
        """
        with open(filepath, 'rb') as f:
            models = pickle.load(f)

        obj = cls()
        obj.models = models

        # Sätt R, S, D till None eller gissas utifrån modellen
        # Här kan vi försöka härleda D från modellen
        obj.D = len(models)
        obj.R = None
        obj.S = None

        print(f"Modell laddad från '{filepath}', {obj.D} detektorer")
        return obj
    


    def plot_clusters_with_probabilities(self, iq_data: np.ndarray, det_index: int = 0, time_index: int = 3, max_shots: int = 20000):
        """
        Visualiserar IQ-klustren och sannolikheterna för tillstånd 1 vid ett givet tidssteg
        för en vald detektor. Punkten färgkodas efter P(state=1).

        Args:
            iq_data: IQ-data med shape (R, S, D)
            det_index: Index för vilken detektor som ska visas
            time_index: Vilket tidssteg (round) som ska plottas
            max_shots: Max antal shots att plottas (för överskådlighet)
        """
        if det_index >= self.D:
            raise ValueError(f"det_index måste vara < {self.D}")
        if time_index >= self.R:
            raise ValueError(f"time_index måste vara < {self.R}")

        S = iq_data.shape[1]
        shots_to_plot = min(S, max_shots)

        # Extrahera IQ-värden
        iq_det = iq_data[time_index, :shots_to_plot, det_index]
        feats = np.column_stack([iq_det.real, iq_det.imag])

        # Beräkna sannolikheter
        model = self.models[det_index]
        posteriors = model.predict_proba(feats)
        probs = posteriors[:, 1]  # P(state = 1)

        # Undvik log(0) genom att lägga till en liten epsilon
        eps = 1e-6
        probs = np.clip(probs, eps, 1.0)

        # Plotta
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(
            feats[:, 0], feats[:, 1],
            c=probs,
            cmap='viridis',
            norm=LogNorm(vmin=eps, vmax=1.0),
            edgecolor='k', alpha=0.7
        )
        plt.colorbar(scatter, label='log(P(state = 1))')
        plt.xlabel('Re(IQ)')
        plt.ylabel('Im(IQ)')
        plt.title(f'Klustervisualisering (log-skala) - Detektor {det_index}, Tidssteg {time_index}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()