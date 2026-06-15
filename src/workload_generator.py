import numpy as np
import matplotlib.pyplot as plt


class WorkloadGenerator:
    """
    Generates three types of realistic cloud traffic patterns.
    Seed is fixed so results are identical every run - important for demo.
    """

    def __init__(self, total_steps=1000, seed=42):
        self.total_steps = total_steps
        np.random.seed(seed)

    def generate_normal(self):
        """Gradual rise and fall — typical weekday traffic."""
        t = np.linspace(0, 4 * np.pi, self.total_steps)
        base = 200 + 100 * np.sin(t)
        noise = np.random.normal(0, 15, self.total_steps)
        return np.clip(base + noise, 50, 500).astype(np.float32)

    def generate_spike(self):
        """Sudden massive burst — flash sale or viral event."""
        np.random.seed(42)
        workload = self.generate_normal()
        spike_start = self.total_steps // 2
        spike_end = spike_start + 80
        workload[spike_start:spike_end] += 600
        return np.clip(workload, 50, 1000).astype(np.float32)

    def generate_wave(self):
        """Multiple peaks — morning and evening traffic."""
        np.random.seed(42)
        t = np.linspace(0, 8 * np.pi, self.total_steps)
        base = 150 + 80 * np.sin(t) + 60 * np.sin(2 * t)
        noise = np.random.normal(0, 10, self.total_steps)
        return np.clip(base + noise, 50, 500).astype(np.float32)

    def save_all(self):
        """Save all three patterns to /data folder."""
        normal = self.generate_normal()
        spike = self.generate_spike()
        wave = self.generate_wave()

        np.save('data/workload_normal.npy', normal)
        np.save('data/workload_spike.npy', spike)
        np.save('data/workload_wave.npy', wave)

        print("✅ Workloads saved to /data folder")
        return normal, spike, wave

    def plot_all(self):
        """Plot all three patterns — your first visual test."""
        normal = self.generate_normal()
        spike = self.generate_spike()
        wave = self.generate_wave()

        fig, axes = plt.subplots(3, 1, figsize=(14, 8))
        fig.suptitle(
            'Cloud Traffic Patterns — Workload Generator',
            fontsize=14,
            fontweight='bold'
        )

        axes[0].plot(normal, color='steelblue', linewidth=1)
        axes[0].set_title('Normal Traffic (Gradual Rise/Fall)')
        axes[0].set_ylabel('Requests/min')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(spike, color='crimson', linewidth=1)
        axes[1].set_title('Spike Traffic (Flash Sale / Viral Event)')
        axes[1].set_ylabel('Requests/min')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(wave, color='forestgreen', linewidth=1)
        axes[2].set_title('Wave Traffic (Morning/Evening Peaks)')
        axes[2].set_ylabel('Requests/min')
        axes[2].set_xlabel('Time Steps')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            'results/workload_patterns.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.show()
        print("✅ Graph saved to results/workload_patterns.png")


if __name__ == "__main__":
    generator = WorkloadGenerator(total_steps=1000)
    generator.plot_all()
    generator.save_all()