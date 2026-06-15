import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('src')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from workload_generator import WorkloadGenerator
from cloud_env import CloudEnvironment
from lstm_predictor import LSTMPredictor


class MetricsCallback(BaseCallback):
    """
    Watches the agent training and records metrics.
    This gives us the learning curve graph.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.sla_violations = []

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
        return True


def make_env(workload, lstm_predictor):
    """Create a monitored environment for training."""
    env = CloudEnvironment(workload, lstm_predictor)
    env = Monitor(env)
    return env


def train_ppo_agent(workload, lstm_predictor,
                    total_timesteps=200000, name="agent"):
    """
    Train a single PPO agent on the given workload.
    Returns trained agent and training metrics.
    """
    print(f"\n🤖 Training PPO Agent: {name}")
    print(f"   Total timesteps: {total_timesteps}")
    print(f"   This will take 15-30 minutes...")
    print(f"   Progress updates every 10,000 steps\n")

    # Create environment
    env = make_env(workload, lstm_predictor)

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    # Train with callback
    callback = MetricsCallback()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    print(f"\n✅ Training complete for {name}!")
    return model, callback


def evaluate_agent(model, workload, lstm_predictor,
                   agent_name="PPO Agent"):
    """
    Run trained agent on workload and collect metrics.
    This is what we compare against rule-based.
    """
    env = CloudEnvironment(workload, lstm_predictor)
    state, _ = env.reset()

    vm_history = []
    traffic_history = []
    response_history = []
    sla_violations = 0

    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, _, info = env.step(action)

        vm_history.append(info['vms'])
        traffic_history.append(info['traffic'])
        response_history.append(info['response_time'])

        if info['sla_violated']:
            sla_violations += 1

    metrics = env.get_metrics()
    metrics['sla_violations'] = sla_violations

    print(f"\n📊 {agent_name} Results:")
    print(f"   SLA Violations:  {sla_violations}")
    print(f"   Total Cost:      {metrics['total_cost']}")
    print(f"   Scaling Actions: {metrics['scaling_actions']}")

    return {
        'vm_history': vm_history,
        'traffic_history': traffic_history,
        'response_history': response_history,
        'metrics': metrics
    }


def evaluate_rule_based(workload):
    """
    Run rule-based scaler on same workload.
    This is your baseline comparison.
    """
    env = CloudEnvironment(workload)
    state, _ = env.reset()

    vm_history = []
    response_history = []
    sla_violations = 0

    done = False
    while not done:
        # Simple rules — no intelligence
        action = env.rule_based_action(state)
        state, reward, done, _, info = env.step(action)

        vm_history.append(info['vms'])
        response_history.append(info['response_time'])

        if info['sla_violated']:
            sla_violations += 1

    metrics = env.get_metrics()

    print(f"\n📊 Rule-Based Results:")
    print(f"   SLA Violations:  {sla_violations}")
    print(f"   Total Cost:      {metrics['total_cost']}")
    print(f"   Scaling Actions: {metrics['scaling_actions']}")

    return {
        'vm_history': vm_history,
        'response_history': response_history,
        'metrics': metrics
    }


def plot_training_curve(callback, name="agent"):
    """Plot how the agent improved during training."""
    if len(callback.episode_rewards) == 0:
        print("No episode data to plot")
        return

    rewards = callback.episode_rewards

    # Smooth the curve with rolling average
    window = min(50, len(rewards) // 10)
    if window > 1:
        smoothed = np.convolve(
            rewards,
            np.ones(window)/window,
            mode='valid'
        )
    else:
        smoothed = rewards

    plt.figure(figsize=(12, 5))
    plt.plot(rewards, alpha=0.3, color='steelblue',
             label='Episode Reward')
    plt.plot(smoothed, color='steelblue', linewidth=2,
             label=f'Smoothed (window={window})')
    plt.title(f'PPO Training Curve — {name}',
              fontsize=13, fontweight='bold')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    path = f'results/training_curve_{name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Training curve saved to {path}")


def plot_comparison(rl_results, rule_results, workload, name="spike"):
    """
    Side by side comparison — your killer demo graph.
    Shows RL agent vs rule-based on same traffic.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f'RL Agent vs Rule-Based Comparison — {name} Traffic',
        fontsize=14, fontweight='bold'
    )

    steps = range(len(workload))

    # Graph 1 — Traffic
    axes[0].plot(workload, color='gray',
                 linewidth=1, label='Actual Traffic')
    axes[0].set_title('Incoming Traffic')
    axes[0].set_ylabel('Requests/min')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graph 2 — VM count
    axes[1].plot(rl_results['vm_history'],
                 color='steelblue', linewidth=1.5,
                 label='RL Agent VMs')
    axes[1].plot(rule_results['vm_history'],
                 color='crimson', linewidth=1.5,
                 linestyle='--', label='Rule-Based VMs')
    axes[1].set_title('Virtual Machines in Use')
    axes[1].set_ylabel('VM Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Graph 3 — Response time
    axes[2].plot(rl_results['response_history'],
                 color='steelblue', linewidth=1,
                 label='RL Agent Response Time')
    axes[2].plot(rule_results['response_history'],
                 color='crimson', linewidth=1,
                 linestyle='--',
                 label='Rule-Based Response Time')
    axes[2].axhline(y=2.0, color='black', linestyle=':',
                    linewidth=1.5, label='SLA Threshold')
    axes[2].set_title('Response Time (below dotted line = SLA met)')
    axes[2].set_ylabel('Seconds')
    axes[2].set_xlabel('Time Steps')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    path = f'results/comparison_{name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Comparison graph saved to {path}")


def save_agent(model, name="ppo_agent"):
    """Save trained agent to models folder."""
    os.makedirs('models', exist_ok=True)
    path = f'models/{name}'
    model.save(path)
    print(f"✅ Agent saved to {path}.zip")


# Main training and evaluation
if __name__ == "__main__":

    print("=" * 50)
    print("TRAINING RL AGENT")
    print("=" * 50)

    # Load traffic data
    gen = WorkloadGenerator()
    spike_workload = gen.generate_spike()
    normal_workload = gen.generate_normal()

    # Load trained LSTM
    lstm = LSTMPredictor()
    lstm.load('models/lstm_model.pth')

    # Train PPO agent on spike traffic
    model, callback = train_ppo_agent(
        spike_workload,
        lstm,
        total_timesteps=200000,
        name="spike_agent"
    )

    # Save the agent
    save_agent(model, "ppo_spike_agent")

    # Plot training curve
    plot_training_curve(callback, name="spike_agent")

    # Evaluate RL agent
    print("\n" + "=" * 50)
    print("EVALUATING AGENTS")
    print("=" * 50)

    rl_results = evaluate_agent(
        model, spike_workload, lstm,
        agent_name="PPO Agent"
    )

    rule_results = evaluate_rule_based(spike_workload)

    # Plot comparison
    plot_comparison(
        rl_results,
        rule_results,
        spike_workload,
        name="spike"
    )

    # Summary table
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<25} {'Rule-Based':>12} {'PPO Agent':>12}")
    print("-" * 50)
    print(f"{'SLA Violations':<25} "
          f"{rule_results['metrics']['total_sla_violations']:>12} "
          f"{rl_results['metrics']['total_sla_violations']:>12}")
    print(f"{'Total Cost':<25} "
          f"{rule_results['metrics']['total_cost']:>12} "
          f"{rl_results['metrics']['total_cost']:>12}")
    print(f"{'Scaling Actions':<25} "
          f"{rule_results['metrics']['scaling_actions']:>12} "
          f"{rl_results['metrics']['scaling_actions']:>12}")
    print("=" * 50)