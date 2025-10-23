import numpy as np


class StochasticMemorylessPolicy:
    def __init__(self, num_actions, no_info=False):

        self.num_actions = num_actions
        self.uniform_action_distribution = (
            np.ones(shape=self.num_actions) / self.num_actions
        )

        self.observation_action_probability = None
        self.no_info = no_info

    def choose_action(self, observation):
        if self.no_info is True:
            chosen_action = np.random.multinomial(
                n=1, pvals=self.uniform_action_distribution
            ).argmax()
        else:
            optimal_action_distribution = self.observation_action_probability[
                observation
            ]
            chosen_action = np.random.multinomial(
                n=1, pvals=optimal_action_distribution
            ).argmax()
        return chosen_action

    def update_policy_info(self, observation_action_probability):
        self.observation_action_probability = observation_action_probability
        self.no_info = False
