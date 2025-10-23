from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def choose_action(self):
        pass

    @abstractmethod
    def update(self, action, observation):
        pass

    @abstractmethod
    def update_transition_matrix(self, transition_matrix):
        pass
