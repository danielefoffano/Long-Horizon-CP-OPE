import numpy as np
from abc import ABC
from Utils.Experience import Experience

class Agent(ABC):
    ns: int # Number of states
    na: int # Number of actions
    discount_factor: float # Discount factor

    def __init__(self, ns: int, na: int, discount_factor: float):
        self.ns = ns
        self.na = na
        self.discount_factor = discount_factor
        self.num_visits_state = np.zeros(self.ns)
        self.num_visits_actions = np.zeros((self.ns, self.na))
        self.last_visit_state = np.zeros(self.ns)

    def forward(self, state: int, step: int) -> int:
        self.num_visits_state[state] += 1
        self.last_visit_state[state] = step
        action = self._forward_logic(state, step)
        self.num_visits_actions[state][action] += 1
        return action

    def backward(self, experience: Experience):
        self._backward_logic(experience)

    def _forward_logic(self, state: int, step: int) -> int:
        raise NotImplementedError

    def _backward_logic(self, experience: Experience):
        raise NotImplementedError

    def greedy_action(self, state: int) -> int:
        raise NotImplementedError