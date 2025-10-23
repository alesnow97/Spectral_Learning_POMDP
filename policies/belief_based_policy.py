from abc import ABC

import numpy as np

import utils
from policies.policy import Policy


class BeliefBasedPolicy(Policy, ABC):
    def __init__(
        self,
        num_states,
        num_actions,
        num_obs,
        min_action_prob,
        state_action_transition_matrix,
        state_action_observation_matrix,
        possible_rewards,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.min_action_prob = min_action_prob

        if state_action_transition_matrix is not None:
            self.state_action_transition_matrix = state_action_transition_matrix
        else:
            self.state_action_transition_matrix = (
                utils.generate_random_transition_matrix(
                    num_states=self.num_states,
                    num_actions=self.num_actions,
                    min_transition_value=0.05,
                )
            )

        self.state_action_observation_matrix = state_action_observation_matrix
        self.possible_rewards = possible_rewards

        self.belief = np.ones(shape=num_states) * 1 / num_states

    def choose_action(self):
        current_state_action_observation = self.state_action_observation_matrix

        current_state_action_reward = (
            current_state_action_observation * self.possible_rewards[None, None, :]
        )

        scaled_matrix = self.belief[:, None, None] * current_state_action_reward

        reduced_scaled_matrix = scaled_matrix.sum(axis=2)
        final_vector = reduced_scaled_matrix.sum(axis=0)
        best_action = np.argmax(final_vector)

        action_probs = np.ones(shape=self.num_actions) * self.min_action_prob
        action_probs[best_action] = 1 - self.min_action_prob * (self.num_actions - 1)

        chosen_action = np.random.multinomial(n=1, pvals=action_probs, size=1)[
            0
        ].argmax()
        return chosen_action

    def update(self, chosen_action, observation_index):
        observation_distribution = self.state_action_observation_matrix[
            :, chosen_action, observation_index
        ].reshape(-1)

        scaled_belief = self.belief * observation_distribution
        current_transition_matrix = self.state_action_transition_matrix[
            :, chosen_action, :
        ]
        transitioned_belief = scaled_belief @ current_transition_matrix.T

        self.belief = transitioned_belief / transitioned_belief.sum()

    def update_transition_matrix(self, estimated_transition_matrix):
        self.state_action_transition_matrix = estimated_transition_matrix

    def reset_belief(self):
        self.belief = np.ones(shape=self.num_states) * 1 / self.num_states
