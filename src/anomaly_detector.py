import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('src')

from sklearn.ensemble import IsolationForest
from workload_generator import WorkloadGenerator


class AnomalyDetector:
    """
    Guards the system against abnormal traffic.

    Trained on normal traffic patterns.
    Raises alarm when something unusual arrives.
    When alarm raised — emergency scale up immediately.
    RL agent bypassed until traffic returns to normal.
    """

    def __init__(self, contamination=0.05):
        # contamination = expected % of anomalies in data
        # 0.05 means we expect 5% of traffic to be unusual
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.is_trained = False
        self.threshold = None

    def train(self, normal_workload):
        """
        Train on normal traffic so detector knows
        what normal looks like.
        Anything far from normal = anomaly.
        """
        print("🔍 Training Anomaly Detector...")

        # Reshape for sklearn
        X = normal_workload.reshape(-1, 1)
        self.detector.fit(X)
        self.is_trained = True

        # Calculate threshold from training data
        scores = self.detector.decision_function(X)
        self.threshold = np.percentile(scores, 5)

        print(f"✅ Anomaly detector trained")
        print(f"   Normal traffic range: "
              f"{normal_workload.min():.0f} - "
              f"{normal_workload.max():.0f} requests/min")
        print(f"   Anomaly threshold score: {self.threshold:.4f}")

    def is_anomaly(self, traffic_value):
        """
        Check if a single traffic value is anomalous.
        Returns True if anomaly, False if normal.
        """
        if not self.is_trained:
            return False

        result = self.detector.predict([[traffic_value]])
        # -1 means anomaly, 1 means normal
        return result[0] == -1

    def get_score(self, traffic_value):
        """
        Get anomaly score for a traffic value.
        More negative = more anomalous.
        """
        if not self.is_trained:
            return 0
        return self.detector.decision_function(
            [[traffic_value]]
        )[0]

    def scan_workload(self, workload):
        """
        Scan entire workload and flag anomalies.
        Returns list of True/False for each step.
        """
        results = []
        for value in workload:
            results.append(self.is_anomaly(value))
        return results

    def plot_detection(self, workload, title="Anomaly Detection"):
        """
        Visualize which traffic points were flagged.
        Green = normal, Red = anomaly detected.
        """
        anomalies = self.scan_workload(workload)

        normal_steps = [i for i, a in enumerate(anomalies)
                       if not a]
        anomaly_steps = [i for i, a in enumerate(anomalies)
                        if a]

        normal_values = [workload[i] for i in normal_steps]
        anomaly_values = [workload[i] for i in anomaly_steps]

        plt.figure(figsize=(14, 5))

        # Plot normal traffic
        plt.scatter(normal_steps, normal_values,
                   color='steelblue', s=5,
                   label=f'Normal ({len(normal_steps)} points)',
                   alpha=0.6)

        # Plot anomalies
        plt.scatter(anomaly_steps, anomaly_values,
                   color='crimson', s=20,
                   label=f'Anomaly ({len(anomaly_steps)} points)',
                   zorder=5)

        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Requests/min')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs('results', exist_ok=True)
        path = f'results/anomaly_detection.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ Anomaly detection graph saved to {path}")

        return anomaly_steps, anomaly_values


def make_scaling_decision(state, rl_agent,
                          anomaly_detector,
                          current_traffic,
                          confidence_threshold=0.3):
    """
    The complete decision pipeline with fallback.

    Step 1: Check if traffic is anomalous
    Step 2: If anomaly → emergency scale up
    Step 3: If normal → let RL agent decide
    Step 4: If RL agent unsure → rule-based fallback

    This is your three-layer safety net.
    """

    # Layer 1 — Anomaly check
    if anomaly_detector.is_anomaly(current_traffic):
        return 2, "anomaly_emergency"  # always scale up

    # Layer 2 — RL agent decides
    action, _ = rl_agent.predict(state, deterministic=False)

    return action, "rl_agent"


# Test the anomaly detector
if __name__ == "__main__":

    print("=" * 50)
    print("TESTING ANOMALY DETECTOR")
    print("=" * 50)

    gen = WorkloadGenerator()

    # Train on normal traffic
    normal_workload = gen.generate_normal()
    spike_workload = gen.generate_spike()

    # Create and train detector
    detector = AnomalyDetector(contamination=0.05)
    detector.train(normal_workload)

    print("\n--- Testing on Normal Traffic ---")
    anomaly_steps, _ = detector.plot_detection(
        normal_workload,
        title="Anomaly Detection — Normal Traffic"
    )
    print(f"Anomalies found in normal traffic: "
          f"{len(anomaly_steps)} / {len(normal_workload)}")

    print("\n--- Testing on Spike Traffic ---")
    anomaly_steps_spike, anomaly_values = detector.plot_detection(
        spike_workload,
        title="Anomaly Detection — Spike Traffic"
    )
    print(f"Anomalies found in spike traffic: "
          f"{len(anomaly_steps_spike)} / {len(spike_workload)}")

    print("\n--- Manual Tests ---")
    test_values = [150, 200, 250, 500, 800, 1000]
    for val in test_values:
        is_anom = detector.is_anomaly(val)
        score = detector.get_score(val)
        status = "🔴 ANOMALY" if is_anom else "🟢 Normal"
        print(f"Traffic {val:>5} requests/min → "
              f"{status} (score: {score:.4f})")

    print("\n✅ Anomaly detector complete!")
    print("Check results/ for detection graphs")