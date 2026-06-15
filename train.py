"""
train.py — Master Training Script
===================================
Runs the full pipeline in correct order:
  Step 1 → Generate & save workloads
  Step 2 → Train LSTM predictor
  Step 3 → Train single PPO agent
  Step 4 → Train multi-agent system
  Step 5 → Evaluate all three systems
  Step 6 → Generate all comparison plots

Run from project root:
    python train.py
"""

import os
import sys
import time

sys.path.append('src')

from workload_generator import WorkloadGenerator
from lstm_predictor import LSTMPredictor
from rl_agent import (
    train_ppo_agent,
    evaluate_agent,
    evaluate_rule_based,
    plot_comparison,
    plot_training_curve,
    save_agent
)
from multi_agent_scaler import (
    MultiAgentScaler,
    evaluate_single_agent,
    evaluate_rule_based as mas_rule_based,
    plot_three_way_comparison,
    print_final_summary
)
from anomaly_detector import AnomalyDetector


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():

    start_time = time.time()

    # ─────────────────────────────────────────────────
    # STEP 1 — Generate workloads
    # ─────────────────────────────────────────────────
    print_section("STEP 1 — Generating Workloads")

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    gen = WorkloadGenerator(total_steps=1000, seed=42)
    normal_workload, spike_workload, wave_workload = gen.save_all()
    gen.plot_all()

    print(f"✅ Workloads ready — {len(spike_workload)} steps each")

    # ─────────────────────────────────────────────────
    # STEP 2 — Train LSTM predictor
    # ─────────────────────────────────────────────────
    print_section("STEP 2 — Training LSTM Predictor")

    lstm = LSTMPredictor(sequence_length=10)
    losses = lstm.train(spike_workload, epochs=50)
    lstm.save('models/lstm_model.pth')

    # Plot training loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(losses, color='steelblue', linewidth=1.5)
    plt.title('LSTM Training Loss Over Epochs',
              fontsize=13, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/lstm_training_loss.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Plot predictions on all three traffic patterns
    lstm.plot_predictions(
        spike_workload,
        title='Spike Traffic Predictions'
    )
    lstm.plot_predictions(
        normal_workload,
        title='Normal Traffic Predictions'
    )
    lstm.plot_predictions(
        wave_workload,
        title='Wave Traffic Predictions'
    )

    print("✅ LSTM training complete — model saved")

    # ─────────────────────────────────────────────────
    # STEP 3 — Train anomaly detector
    # ─────────────────────────────────────────────────
    print_section("STEP 3 — Training Anomaly Detector")

    detector = AnomalyDetector(contamination=0.05)
    detector.train(normal_workload)
    detector.plot_detection(
        spike_workload,
        title="Anomaly Detection — Spike Traffic"
    )

    print("✅ Anomaly detector trained")

    # ─────────────────────────────────────────────────
    # STEP 4 — Train single PPO agent
    # ─────────────────────────────────────────────────
    print_section("STEP 4 — Training Single PPO Agent")
    print("  This will take ~15–30 minutes on CPU...")

    model, callback = train_ppo_agent(
        spike_workload,
        lstm,
        total_timesteps=200000,
        name="spike_agent"
    )
    save_agent(model, "ppo_spike_agent")
    plot_training_curve(callback, name="spike_agent")

    print("✅ Single PPO agent trained and saved")

    # ─────────────────────────────────────────────────
    # STEP 5 — Train multi-agent system
    # ─────────────────────────────────────────────────
    print_section("STEP 5 — Training Multi-Agent System")
    print("  Training 3 agents — ~45–60 minutes on CPU...")

    mas = MultiAgentScaler(
        workload=spike_workload,
        lstm_predictor=lstm,
        n_agents=3
    )
    mas.train(timesteps=200000)
    mas.save('models')

    print("✅ Multi-agent system trained and saved")

    # ─────────────────────────────────────────────────
    # STEP 6 — Evaluate all three systems
    # ─────────────────────────────────────────────────
    print_section("STEP 6 — Evaluating All Systems")

    # Single agent vs rule-based
    rl_results = evaluate_agent(
        model, spike_workload, lstm,
        agent_name="PPO Agent"
    )
    rule_results_single = evaluate_rule_based(spike_workload)

    plot_comparison(
        rl_results,
        rule_results_single,
        spike_workload,
        name="spike"
    )

    # Three-way comparison
    multi_results = mas.evaluate()
    single_results = evaluate_single_agent(spike_workload, lstm)
    rule_results_3way = mas_rule_based(spike_workload)

    plot_three_way_comparison(
        multi_results,
        single_results,
        rule_results_3way,
        spike_workload
    )

    print_final_summary(
        multi_results,
        single_results,
        rule_results_3way
    )

    # ─────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60

    print_section("TRAINING COMPLETE")
    print(f"  Total time: {elapsed:.1f} minutes")
    print(f"  Models saved to:  models/")
    print(f"  Results saved to: results/")
    print(f"\n  To launch the dashboard:")
    print(f"  cd dashboard && streamlit run dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    main()