from abc import ABC

import numpy as np

import utils
from policies.policy import Policy


class DiscretizedBeliefBasedPolicy(Policy, ABC):
    def __init__(
        self,
        num_states,
        num_actions,
        num_obs,
        discretized_beliefs,
        estimated_state_action_transition_matrix,
        belief_action_mapping,
        estimated_state_observation_matrix,
        initial_discretized_belief=None,
        initial_discretized_belief_index=None,
        no_info=False,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs

        self.discretized_beliefs = discretized_beliefs

        self.belief_based_policy = belief_action_mapping

        self.state_action_transition_matrix = estimated_state_action_transition_matrix
        self.state_observation_matrix = estimated_state_observation_matrix

        self.uniform_action_distribution = (
            np.ones(shape=self.num_actions) / self.num_actions
        )

        if initial_discretized_belief is not None:
            self.discretized_belief = initial_discretized_belief
            self.discretized_belief_index = initial_discretized_belief_index
        else:
            (
                self.discretized_belief,
                self.discretized_belief_index,
            ) = utils.find_closest_discretized_belief(
                self.discretized_beliefs,
                np.ones(shape=self.num_states) / self.num_states,
            )

        self.no_info = no_info

    def choose_action(self):

        if self.no_info is True:
            chosen_action = np.random.multinomial(
                n=1, pvals=self.uniform_action_distribution
            ).argmax()
        else:
            chosen_action = self.belief_based_policy[self.discretized_belief_index]

        return chosen_action

    def discretized_belief_update(self, current_observation, past_action):
        if self.no_info is True:
            return
        current_transition_matrix = self.state_action_transition_matrix[
            :, past_action, :
        ]
        observation_distribution = self.state_observation_matrix[
            :, current_observation
        ].reshape(-1)

        transitioned_belief = self.discretized_belief @ current_transition_matrix
        scaled_belief = transitioned_belief * observation_distribution
        belief = scaled_belief / scaled_belief.sum()

        (
            self.discretized_belief,
            self.discretized_belief_index,
        ) = utils.find_closest_discretized_belief(self.discretized_beliefs, belief)

    def discretized_belief_update_obs(self, observation_index):
        if self.no_info is True:
            return
        observation_distribution = self.state_observation_matrix[
            :, observation_index
        ].reshape(-1)
        scaled_belief = self.discretized_belief * observation_distribution
        belief = scaled_belief / scaled_belief.sum()

        (
            self.discretized_belief,
            self.discretized_belief_index,
        ) = utils.find_closest_discretized_belief(self.discretized_beliefs, belief)

    def update_policy_infos(
        self,
        state_action_transition_matrix,
        estimated_state_observation_matrix,
        belief_action_mapping,
    ):
        self.state_action_transition_matrix = state_action_transition_matrix
        self.state_observation_matrix = estimated_state_observation_matrix
        self.belief_based_policy = belief_action_mapping
        self.no_info = False

    def update(self, action, observation):
        return None

    def update_transition_matrix(self, transition_matrix):
        return None
