from Policies.Policy import Policy
import numpy as np

class EpsilonGreedyPolicy(Policy):
    def __init__(self, q_table, epsilon, na):
        super().__init__()
        self.epsilon = epsilon
        self.na = na
        self.q_function = q_table
        self.probabilities = np.zeros_like(q_table)
        for s in range(q_table.shape[0]):
            self.probabilities[s, q_table[s].argmax()] = 1

        self.probabilities = epsilon * np.ones_like(q_table)/na + (1-epsilon) * self.probabilities


    def get_action(self, state: int) -> int:
        return np.random.choice(self.na, p=self.probabilities[state])

    def get_action_prob(self, state: int, action: int) -> float:
        return self.probabilities[state, action]