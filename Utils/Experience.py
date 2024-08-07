from typing import NamedTuple

class Experience(NamedTuple):
    state: int
    action: int
    reward: float
    next_state: int
    done: bool