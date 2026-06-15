"""
validate_agents.py — RL Agent Generalization Test
===================================================
Tests trained RL agents on Shard 2 data
they were NEVER trained on.

Training data: Shard 0 + Shard 1 combined
Test data:     Shard 2 only (never seen during training)

Run from project root:
    python validate_agents.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')

from stable_baselines3 import PPO
from cloud_env import CloudEnvironment
from lstm_predictor import LSTMPredictor
from multi_agent_scaler import MultiAgentScaler


def run_rule_based(workload):
    env = CloudEnvironment(workload)
    state, _ = env.reset()
    vm_hist, resp_hist, sla_hist = [], [], []
    done = False
    while not done:
        action = env.rule_based_action(state)
        state, _, done, _, info = env.step(action)
        vm_hist.append(info['vms'])
        resp_hist.append(info['response_time'])
        sla_hist.append(info['sla_violated'])
    return {
        'vm_history': vm_hist,
        'response_history': resp_hist,
        'sla_history': sla_hist,
        'metrics': env.get_metrics()
    }


def run_single_agent(workload, lstm):
    model = PPO.load('models/ppo_spike_agent')
    env = CloudEnvironment(workload, lstm)
    state, _ = env.reset()
    vm_hist, resp_hist, sla_hist = [], [], []
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, _, done, _, info = env.step(action)
        vm_hist.append(info['vms'])
        resp_hist.append(info['response_time'])
        sla_hist.append(info['sla_violated'])
    return {
        'vm_history': vm_hist,
        'response_history': resp_hist,
        'sla_history': sla_hist,
        'metrics': env.get_metrics()
    }


def run_multi_agent(workload, lstm):
    mas = MultiAgentScaler(
        workload=workload,
        lstm_predictor=lstm,
        n_agents=3
    )
    mas.load('models')
    all_vms, all_resp, all_sla = [], [], []
    total_sla, total_cost, total_scale = 0, 0, 0
    for agent, segment in zip(mas.agents, mas.segments):
        env = CloudEnvironment(segment, lstm)
        state, _ = env.reset()
        vm_hist, resp_hist, sla_hist = [], [], []
        done = False
        while not done:
            action, _ = agent.predict(state, deterministic=True)
            state, _, done, _, info = env.step(action)
            vm_hist.append(info['vms'])
            resp_hist.append(info['response_time'])
            sla_hist.append(info['sla_violated'])
        m = env.get_metrics()
        total_sla += m['total_sla_violations']
        total_cost += m['total_cost']
        total_scale += m['scaling_actions']
        all_vms.extend(vm_hist)
        all_resp.extend(resp_hist)
        all_sla.extend(sla_hist)
    return {
        'vm_history': all_vms,
        'response_history': all_resp,
        'sla_history': all_sla,
        'metrics': {
            'total_sla_violations': total_sla,
            'total_cost': round(total_cost, 2),
            'scaling_actions': total_scale
        }
    }


def evaluate_all_systems(workload, lstm, label):
    print(f"\n  Running Rule-Based on {label}...")
    rule = run_rule_based(workload)
    print(f"  Running Single RL on {label}...")
    single = run_single_agent(workload, lstm)
    print(f"  Running Multi-Agent on {label}...")
    multi = run_multi_agent(workload, lstm)

    print(f"\n  Results on {label}:")
    print(f"  {'System':<15} {'SLA Violations':>15} "
          f"{'Cost':>10} {'Scaling':>10}")
    print(f"  {'-'*52}")
    for name, r in [('Rule-Based', rule),
                    ('Single RL', single),
                    ('Multi-Agent', multi)]:
        m = r['metrics']
        print(f"  {name:<15} "
              f"{m['total_sla_violations']:>15} "
              f"{m['total_cost']:>10.1f} "
              f"{m['scaling_actions']:>10}")
    return rule, single, multi


def plot_cross_shard_comparison(
    s0_rule, s0_single, s0_multi,
    s2_rule, s2_single, s2_multi,
    shard0_workload, shard2_workload
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Agent Generalization Test\n'
        'Training Data (Shard 0+1) vs Test Data (Shard 2 — Never Seen)',
        fontsize=13, fontweight='bold'
    )

    def plot_row(axes_row, rule, single, multi, workload, title):
        ax = axes_row[0]
        ax.plot(multi['vm_history'], color='steelblue',
                linewidth=1.5, label='Multi-Agent RL')
        ax.plot(single['vm_history'], color='darkorange',
                linewidth=1.5, linestyle='--', label='Single RL')
        ax.plot(rule['vm_history'], color='crimson',
                linewidth=1.5, linestyle=':', label='Rule-Based')
        ax.set_title(f'{title} — VM Count')
        ax.set_ylabel('VMs')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes_row[1]
        ax.plot(multi['response_history'], color='steelblue',
                linewidth=1, label='Multi-Agent RL')
        ax.plot(single['response_history'], color='darkorange',
                linewidth=1, linestyle='--', label='Single RL')
        ax.plot(rule['response_history'], color='crimson',
                linewidth=1, linestyle=':', label='Rule-Based')
        ax.axhline(y=2.0, color='black', linestyle=':',
                   linewidth=2, label='SLA Threshold', alpha=0.7)
        ax.set_title(f'{title} — Response Time')
        ax.set_ylabel('Seconds')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes_row[2]
        systems = ['Rule\nBased', 'Single\nRL', 'Multi\nAgent']
        violations = [
            rule['metrics']['total_sla_violations'],
            single['metrics']['total_sla_violations'],
            multi['metrics']['total_sla_violations']
        ]
        colors = ['crimson', 'darkorange', 'steelblue']
        bars = ax.bar(systems, violations, color=colors,
                      alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, violations):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, str(val),
                    ha='center', fontweight='bold', fontsize=10)
        ax.set_title(f'{title} — SLA Violations')
        ax.set_ylabel('Violations')
        ax.grid(True, alpha=0.3, axis='y')

    plot_row(axes[0], s0_rule, s0_single, s0_multi,
             shard0_workload, 'Training Data (Shard 0+1)')
    plot_row(axes[1], s2_rule, s2_single, s2_multi,
             shard2_workload, 'Test Data (Shard 2 — Never Seen)')

    plt.tight_layout()
    path = 'results/agent_generalization_test.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Generalization graph saved to {path}")


def print_generalization_verdict(
    s0_rule, s0_single, s0_multi,
    s2_rule, s2_single, s2_multi
):
    print("\n" + "=" * 60)
    print("GENERALIZATION VERDICT")
    print("Training Data vs Shard 2 (Never Seen)")
    print("=" * 60)

    s0_m = s0_multi['metrics']['total_sla_violations']
    s2_m = s2_multi['metrics']['total_sla_violations']
    s0_r = s0_rule['metrics']['total_sla_violations']
    s2_r = s2_rule['metrics']['total_sla_violations']

    improvement_s0 = ((s0_r - s0_m) / s0_r * 100) if s0_r > 0 else 0
    improvement_s2 = ((s2_r - s2_m) / s2_r * 100) if s2_r > 0 else (
        0 if s2_m > 0 else 100
    )
    consistency = abs(improvement_s2 - improvement_s0)

    print(f"\n  Multi-Agent improvement over Rule-Based:")
    print(f"  Training data (Shard 0+1): "
          f"{improvement_s0:+.1f}% SLA reduction")
    print(f"  Test data (Shard 2):       "
          f"{improvement_s2:+.1f}% SLA reduction")
    print(f"  Consistency gap:           {consistency:.1f}%")

    print(f"\n  Detailed comparison:")
    print(f"  {'Metric':<25} {'Training':>12} {'Shard 2':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Rule-Based Violations':<25} "
          f"{s0_r:>12} {s2_r:>12}")
    print(f"  {'Single RL Violations':<25} "
          f"{s0_single['metrics']['total_sla_violations']:>12} "
          f"{s2_single['metrics']['total_sla_violations']:>12}")
    print(f"  {'Multi-Agent Violations':<25} "
          f"{s0_m:>12} {s2_m:>12}")
    print(f"  {'Rule-Based Cost':<25} "
          f"{s0_rule['metrics']['total_cost']:>12.1f} "
          f"{s2_rule['metrics']['total_cost']:>12.1f}")
    print(f"  {'Multi-Agent Cost':<25} "
          f"{s0_multi['metrics']['total_cost']:>12.1f} "
          f"{s2_multi['metrics']['total_cost']:>12.1f}")

    if improvement_s2 > 20 and consistency < 20:
        verdict = "STRONG GENERALIZATION"
        detail = (
            "Multi-Agent beats Rule-Based on BOTH training "
            "and unseen test data. The improvement is genuine "
            "and not dataset-specific. Research finding validated."
        )
    elif improvement_s2 > 0 and consistency < 40:
        verdict = "GOOD GENERALIZATION"
        detail = (
            "Multi-Agent still wins on completely unseen Shard 2. "
            "Some variation across shards is normal and expected."
        )
    elif improvement_s2 > 0:
        verdict = "PARTIAL GENERALIZATION"
        detail = (
            "Multi-Agent wins on unseen data but with reduced "
            "margin. Training on more diverse shards would help."
        )
    else:
        verdict = "NEEDS MORE TRAINING DATA"
        detail = (
            "Multi-Agent doesn't beat Rule-Based on Shard 2. "
            "The model needs more diverse training patterns."
        )

    print(f"\n  Verdict: {verdict}")
    print(f"  {detail}")

    print(f"\n  Statement for your presentation:")
    print(f"  We trained on two Google Cluster Trace shards")
    print(f"  (Shard 0 + Shard 1, 400,000 real records)")
    print(f"  and evaluated on a completely independent")
    print(f"  Shard 2 the models never saw during training.")
    print(f"  Multi-Agent RL achieved {improvement_s2:.0f}% SLA")
    print(f"  reduction vs rule-based on held-out test data,")
    print(f"  compared to {improvement_s0:.0f}% on training data.")


def main():
    print("=" * 60)
    print("AGENT GENERALIZATION TEST")
    print("Training (Shard 0+1) vs Test (Shard 2)")
    print("=" * 60)

    # Load LSTM
    print("\nLoading LSTM...")
    lstm = LSTMPredictor()
    lstm.load('models/lstm_model.pth')

    # Load training workloads
    print("\nLoading training workloads (Shard 0+1)...")
    shard0 = {
        'spike': np.load('data/workload_spike.npy'),
        'normal': np.load('data/workload_normal.npy'),
        'wave': np.load('data/workload_wave.npy')
    }
    print("Training workloads loaded")

    # Load Shard 2 test workloads
    print("\nLoading Shard 2 test workloads...")
    test_spike = 'data/test/workload_spike.npy'
    test_normal = 'data/test/workload_normal.npy'
    test_wave = 'data/test/workload_wave.npy'

    if not all(os.path.exists(p) for p in
               [test_spike, test_normal, test_wave]):
        print("Shard 2 test data not found.")
        print("Run first: python data_loader.py --process-test")
        sys.exit(1)

    shard2 = {
        'spike': np.load(test_spike),
        'normal': np.load(test_normal),
        'wave': np.load(test_wave)
    }
    print("Shard 2 test data loaded from data/test/")
    print("These workloads were never seen by any trained model")

    # Evaluate on training data
    print("\n" + "=" * 60)
    print("EVALUATING ON TRAINING DATA (Shard 0+1)")
    print("=" * 60)
    s0_rule, s0_single, s0_multi = evaluate_all_systems(
        shard0['spike'], lstm, "Training Data"
    )

    # Evaluate on Shard 2
    print("\n" + "=" * 60)
    print("EVALUATING ON SHARD 2 (Never Seen)")
    print("=" * 60)
    s2_rule, s2_single, s2_multi = evaluate_all_systems(
        shard2['spike'], lstm, "Shard 2 Test"
    )

    # Plot
    plot_cross_shard_comparison(
        s0_rule, s0_single, s0_multi,
        s2_rule, s2_single, s2_multi,
        shard0['spike'], shard2['spike']
    )

    # Verdict
    print_generalization_verdict(
        s0_rule, s0_single, s0_multi,
        s2_rule, s2_single, s2_multi
    )

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("Graph saved: results/agent_generalization_test.png")
    print("=" * 60)


if __name__ == "__main__":
    main()