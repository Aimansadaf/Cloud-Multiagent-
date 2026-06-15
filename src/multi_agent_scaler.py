import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('src')

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from workload_generator import WorkloadGenerator
from cloud_env import CloudEnvironment
from lstm_predictor import LSTMPredictor


class MultiAgentScaler:
    """
    Three independent PPO agents each managing
    their own cluster (traffic segment).

    Agent 1 → first third of traffic
    Agent 2 → middle third of traffic
    Agent 3 → final third of traffic

    No overlap. No conflict. Each agent is an
    expert on its own portion of the workload.
    """

    def __init__(self, workload, lstm_predictor, n_agents=3):
        self.workload = workload
        self.lstm_predictor = lstm_predictor
        self.n_agents = n_agents
        self.agents = []
        self.segments = []
        self.is_trained = False

        # Split workload into equal segments
        segment_size = len(workload) // n_agents
        for i in range(n_agents):
            start = i * segment_size
            # Last agent gets any remaining steps
            end = (i + 1) * segment_size if i < n_agents - 1 else len(workload)
            self.segments.append(workload[start:end])

        print(f"Workload split into {n_agents} segments:")
        for i, seg in enumerate(self.segments):
            print(f"  Agent {i+1}: {len(seg)} steps")

    def train(self, timesteps=200000):
        """
        Train each agent on its own segment.
        Agent 1 becomes expert on first third.
        Agent 2 becomes expert on middle third.
        Agent 3 becomes expert on final third.
        """
        print("\n" + "=" * 50)
        print("TRAINING MULTI-AGENT SYSTEM")
        print(f"{self.n_agents} agents on separate clusters")
        print("=" * 50)

        self.agents = []

        for i in range(self.n_agents):
            print(f"\n🤖 Training Agent {i+1} of {self.n_agents}")
            print(f"   Cluster: steps {i * len(self.segments[0])} "
                  f"to {(i+1) * len(self.segments[0])}")

            # Each agent gets its OWN segment
            env = CloudEnvironment(
                self.segments[i],
                self.lstm_predictor
            )
            env = Monitor(env)

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0
            )

            model.learn(total_timesteps=timesteps)
            self.agents.append(model)
            print(f"✅ Agent {i+1} trained on its cluster")

        self.is_trained = True
        print("\n✅ All agents trained on separate clusters!")

    def save(self, folder='models'):
        """Save all agents."""
        os.makedirs(folder, exist_ok=True)
        for i, agent in enumerate(self.agents):
            path = f"{folder}/multi_agent_{i+1}"
            agent.save(path)
        print(f"✅ All {self.n_agents} agents saved")

    def load(self, folder='models'):
        """Load all agents."""
        self.agents = []
        for i in range(self.n_agents):
            path = f"{folder}/multi_agent_{i+1}"
            model = PPO.load(path)
            self.agents.append(model)
        self.is_trained = True
        print(f"✅ All {self.n_agents} agents loaded")

    def evaluate(self):
        """
        Run each agent on its own segment.
        Combine results to get full picture.
        """
        if not self.is_trained:
            print("Agents not trained yet.")
            return None

        print("\n📊 Evaluating Multi-Agent System...")

        all_vm_histories = []
        all_response_histories = []
        total_sla_violations = 0
        total_cost = 0
        total_scaling = 0

        for i, (agent, segment) in enumerate(
            zip(self.agents, self.segments)
        ):
            # Each agent runs on its own segment only
            env = CloudEnvironment(
                segment,
                self.lstm_predictor
            )
            state, _ = env.reset()

            vm_history = []
            response_history = []

            done = False
            while not done:
                action, _ = agent.predict(
                    state, deterministic=True
                )
                state, reward, done, _, info = env.step(action)
                vm_history.append(info['vms'])
                response_history.append(info['response_time'])

            metrics = env.get_metrics()
            total_sla_violations += metrics['total_sla_violations']
            total_cost += metrics['total_cost']
            total_scaling += metrics['scaling_actions']

            all_vm_histories.append(vm_history)
            all_response_histories.append(response_history)

        # Combine all segments into one continuous result
        combined_vms = []
        combined_response = []
        for vh, rh in zip(all_vm_histories, all_response_histories):
            combined_vms.extend(vh)
            combined_response.extend(rh)

        print(f"\n📊 Multi-Agent Results (3 clusters combined):")
        print(f"   SLA Violations:  {total_sla_violations}")
        print(f"   Total Cost:      {round(total_cost, 2)}")
        print(f"   Scaling Actions: {total_scaling}")

        return {
            'vm_history': combined_vms,
            'response_history': combined_response,
            'metrics': {
                'total_sla_violations': total_sla_violations,
                'total_cost': round(total_cost, 2),
                'scaling_actions': total_scaling
            }
        }


def evaluate_rule_based(workload):
    """Rule-based baseline."""
    env = CloudEnvironment(workload)
    state, _ = env.reset()

    vm_history = []
    response_history = []

    done = False
    while not done:
        action = env.rule_based_action(state)
        state, reward, done, _, info = env.step(action)
        vm_history.append(info['vms'])
        response_history.append(info['response_time'])

    metrics = env.get_metrics()

    print(f"\n📊 Rule-Based Results:")
    print(f"   SLA Violations:  {metrics['total_sla_violations']}")
    print(f"   Total Cost:      {metrics['total_cost']}")
    print(f"   Scaling Actions: {metrics['scaling_actions']}")

    return {
        'vm_history': vm_history,
        'response_history': response_history,
        'metrics': metrics
    }


def evaluate_single_agent(workload, lstm):
    """Single agent for comparison."""
    model = PPO.load('models/ppo_spike_agent')
    env = CloudEnvironment(workload, lstm)
    state, _ = env.reset()

    vm_history = []
    response_history = []

    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, _, info = env.step(action)
        vm_history.append(info['vms'])
        response_history.append(info['response_time'])

    metrics = env.get_metrics()

    print(f"\n📊 Single Agent Results:")
    print(f"   SLA Violations:  {metrics['total_sla_violations']}")
    print(f"   Total Cost:      {metrics['total_cost']}")
    print(f"   Scaling Actions: {metrics['scaling_actions']}")

    return {
        'vm_history': vm_history,
        'response_history': response_history,
        'metrics': metrics
    }


def plot_three_way_comparison(multi_results, single_results,
                               rule_results, workload):
    """Three-way comparison graph."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle(
        'Three-Way Comparison — Spike Traffic\n'
        'Rule-Based vs Single RL vs Multi-Agent RL',
        fontsize=13, fontweight='bold'
    )

    # Traffic
    axes[0].plot(workload, color='gray',
                 linewidth=1, label='Actual Traffic')
    axes[0].set_title('Incoming Traffic')
    axes[0].set_ylabel('Requests/min')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # VM count
    axes[1].plot(multi_results['vm_history'],
                 color='steelblue', linewidth=1.5,
                 label='Multi-Agent RL')
    axes[1].plot(single_results['vm_history'],
                 color='darkorange', linewidth=1.5,
                 linestyle='--', label='Single RL Agent')
    axes[1].plot(rule_results['vm_history'],
                 color='crimson', linewidth=1.5,
                 linestyle=':', label='Rule-Based')
    axes[1].set_title('Virtual Machines in Use')
    axes[1].set_ylabel('VM Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Response time
    axes[2].plot(multi_results['response_history'],
                 color='steelblue', linewidth=1,
                 label='Multi-Agent RL')
    axes[2].plot(single_results['response_history'],
                 color='darkorange', linewidth=1,
                 linestyle='--', label='Single RL Agent')
    axes[2].plot(rule_results['response_history'],
                 color='crimson', linewidth=1,
                 linestyle=':', label='Rule-Based')
    axes[2].axhline(y=2.0, color='black', linestyle=':',
                    linewidth=2, label='SLA Threshold')
    axes[2].set_title(
        'Response Time (below dotted line = SLA met)'
    )
    axes[2].set_ylabel('Seconds')
    axes[2].set_xlabel('Time Steps')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'results/three_way_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Three-way comparison saved to {path}")


def print_final_summary(multi_results, single_results,
                         rule_results):
    """Final results table."""
    print("\n" + "=" * 60)
    print("FINAL THREE-WAY RESULTS SUMMARY")
    print("=" * 60)

    rm = rule_results['metrics']
    sm = single_results['metrics']
    mm = multi_results['metrics']

    print(f"{'Metric':<25} {'Rule-Based':>12} "
          f"{'Single RL':>12} {'Multi-Agent':>12}")
    print("-" * 60)
    print(f"{'SLA Violations':<25} "
          f"{rm['total_sla_violations']:>12} "
          f"{sm['total_sla_violations']:>12} "
          f"{mm['total_sla_violations']:>12}")
    print(f"{'Total Cost':<25} "
          f"{rm['total_cost']:>12} "
          f"{sm['total_cost']:>12} "
          f"{mm['total_cost']:>12}")
    print(f"{'Scaling Actions':<25} "
          f"{rm['scaling_actions']:>12} "
          f"{sm['scaling_actions']:>12} "
          f"{mm['scaling_actions']:>12}")
    print("=" * 60)


# Main
if __name__ == "__main__":

    print("Loading data and models...")

    gen = WorkloadGenerator()
    spike_workload = gen.generate_spike()

    lstm = LSTMPredictor()
    lstm.load('models/lstm_model.pth')

    # Train multi-agent on separate clusters
    mas = MultiAgentScaler(
        workload=spike_workload,
        lstm_predictor=lstm,
        n_agents=3
    )
    mas.train(timesteps=200000)
    mas.save()

    # Evaluate all three systems
    print("\n" + "=" * 50)
    print("EVALUATING ALL THREE SYSTEMS")
    print("=" * 50)

    multi_results = mas.evaluate()
    single_results = evaluate_single_agent(spike_workload, lstm)
    rule_results = evaluate_rule_based(spike_workload)

    # Plot and summarize
    plot_three_way_comparison(
        multi_results,
        single_results,
        rule_results,
        spike_workload
    )

    print_final_summary(
        multi_results,
        single_results,
        rule_results
    )