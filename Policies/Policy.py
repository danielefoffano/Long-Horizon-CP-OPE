from abc import abstractmethod
from abc import ABC

class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, state: int) -> int:
        pass

    @abstractmethod
    def get_action_prob(self, state: int, action: int) -> float:
        pass