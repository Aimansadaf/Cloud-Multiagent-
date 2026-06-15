import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CloudEnvironment(gym.Env):
    """
    Simulates a cloud computing environment.
    
    Think of this as the kitchen from our story:
    - VMs are the chefs
    - Requests are the food orders
    - Queue is orders waiting to be processed
    - Response time is how long customers wait
    """

    def __init__(self, workload, lstm_predictor=None):
        super().__init__()

        self.workload = workload
        self.total_steps = len(workload)
        self.lstm_predictor = lstm_predictor

        self.max_vms = 20
        self.min_vms = 1
        self.vm_capacity = 50
        self.sla_threshold = 2.0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.vms = 5
        self.queue = 0
        self.last_action = 1

        self.total_sla_violations = 0
        self.total_cost = 0
        self.scaling_actions = 0

        return self._get_state(), {}

    def step(self, action):

        previous_vms = self.vms

        if action == 2:
            self.vms = min(self.vms + 1, self.max_vms)
        elif action == 0:
            self.vms = max(self.vms - 1, self.min_vms)

        if self.vms != previous_vms:
            self.scaling_actions += 1

        current_traffic = self.workload[self.current_step]
        capacity = self.vms * self.vm_capacity

        self.queue = max(0, self.queue + current_traffic - capacity)

        cpu = min(current_traffic / capacity, 1.0)

        response_time = 0.1 + (self.queue / max(capacity, 1)) * 2.0
        response_time = min(response_time, 10.0)

        sla_violated = response_time > self.sla_threshold
        if sla_violated:
            self.total_sla_violations += 1

        self.total_cost += self.vms * 0.1

        reward = self._calculate_reward(
            response_time, sla_violated, action, previous_vms
        )

        self.last_action = action
        self.current_step += 1
        done = self.current_step >= self.total_steps - 1

        state = self._get_state(cpu, response_time)

        info = {
            'response_time': response_time,
            'sla_violated': sla_violated,
            'vms': self.vms,
            'queue': self.queue,
            'cpu': cpu,
            'cost': self.total_cost,
            'traffic': current_traffic
        }

        return state, reward, done, False, info

    def _calculate_reward(self, response_time, sla_violated,
                          action, previous_vms):
        """
        FINAL VERSION:
        - SLA penalty = -10 (stable learning)
        - Cost penalty = 0.1 (don't over-optimize cost)
        - Positive rewards increased so agent naturally
          chases good response time instead of just
          avoiding SLA violations
        """

        reward = 0.0

        # SLA violation — -10 is stable for learning
        if sla_violated:
            reward -= 10.0

        # Cost — small penalty, don't over-optimize
        traffic = self.workload[self.current_step - 1]
        vms_needed = max(1, int(np.ceil(
            traffic / self.vm_capacity
        )))
        excess_vms = max(0, self.vms - vms_needed - 1)
        reward -= excess_vms * 0.1

        # Stability — small penalty for changing too often
        if action != self.last_action and action != 1:
            reward -= 0.2

        # Positive reward — increased significantly
        # This is the key change — agent now strongly
        # rewarded for keeping response time low
        if response_time < 0.5:
            reward += 5.0
        elif response_time < 1.0:
            reward += 3.0
        elif response_time < self.sla_threshold:
            reward += 1.0

        return reward

    def _get_state(self, cpu=0.5, response_time=0.5):

        if (self.lstm_predictor is not None and
                self.current_step >= 10):
            history = self.workload[
                max(0, self.current_step-10):self.current_step
            ]
            prediction = self.lstm_predictor.predict_next(history)
            normalized_prediction = min(prediction / 1000.0, 1.0)
        else:
            normalized_prediction = 0.5

        return np.array([
            float(cpu),
            float(self.vms / self.max_vms),
            float(min(self.queue / 1000.0, 1.0)),
            float(min(response_time / 10.0, 1.0)),
            float(normalized_prediction)
        ], dtype=np.float32)

    def get_metrics(self):
        return {
            'total_sla_violations': self.total_sla_violations,
            'total_cost': round(self.total_cost, 2),
            'scaling_actions': self.scaling_actions,
            'final_vms': self.vms
        }

    def rule_based_action(self, state):
        cpu = state[0]
        if cpu > 0.8:
            return 2
        elif cpu < 0.3:
            return 0
        else:
            return 1