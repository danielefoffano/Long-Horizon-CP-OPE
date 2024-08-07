from Agents.Agent import Agent
from Utils.Experience import Experience
import numpy as np

class QlearningAgent(Agent):
    def __init__(self, ns: int, na: int, discount_factor: float, alpha: float, explorative_policy = None):
        super().__init__(ns, na, discount_factor)
        self.q_function = np.zeros((self.ns, self.na))
        self.alpha = alpha
        self.explorative_policy = explorative_policy

    def _forward_logic(self, state: int, step: int) -> int:
        if self.explorative_policy is not None:
            state_sum = np.sum(self.explorative_policy[state] + 1/self.ns)
            probs = (self.explorative_policy[state] + 1/self.ns)/state_sum
            action = np.random.choice(range(self.na),1,p=probs)[0]
        else:
            eps = 1 if self.num_visits_state[state] <= 2 * self.na else max(0.5, 1 / (self.num_visits_state[state] - 2*self.na))
            action = np.random.choice(self.na) if np.random.uniform() < eps else self.q_function[state].argmax()
        return action

    def greedy_action(self, state: int) -> int:
        return self.q_function[state].argmax()

    def _backward_logic(self, experience: Experience):
        state, action, reward, next_state, done = list(experience)
        target = reward + (1-done) * self.discount_factor * self.q_function[next_state].max()
        lr = 1 / (self.num_visits_actions[state][action] ** self.alpha)
        self.q_function[state][action] += lr * (target - self.q_function[state][action])