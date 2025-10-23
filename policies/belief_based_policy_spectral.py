from abc import ABC

import numpy as np

from policies.policy import Policy


class BeliefBasedPolicySpectral(Policy, ABC):
    def __init__(
        self,
        num_states,
        num_actions,
        num_obs,
        min_action_prob,
        observation_matrix,
        possible_rewards,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.min_action_prob = min_action_prob
        self.state_action_transition_matrix = self.generate_random_transition_matrix(
            min_transition_value=0.05
        )
        self.observation_matrix = observation_matrix
        self.possible_rewards = possible_rewards

        self.state_expected_reward = (
            self.observation_matrix * self.possible_rewards[None, :]
        )
        self.state_expected_reward = self.state_expected_reward.sum(axis=1)

        self.action_probs = np.ones(shape=self.num_actions) / self.num_actions
        self.uniform_action_probs = np.ones(shape=self.num_actions) / self.num_actions

        self.belief = np.ones(shape=num_states) / num_states

    def choose_action(self):
        if self.uniform:
            chosen_action = np.random.multinomial(
                n=1, pvals=self.uniform_action_probs
            ).argmax()
        else:

            best_action = np.argmax(self.belief @ self.state_expected_reward)
            current_action_prob = np.ones(shape=self.num_actions) * self.min_action_prob
            current_action_prob[best_action] = 1 - self.min_action_prob * (
                self.num_actions - 1
            )

            chosen_action = np.random.multinomial(
                n=1, pvals=current_action_prob
            ).argmax()

            current_transition_matrix = self.state_action_transition_matrix[
                :, chosen_action, :
            ]
            self.belief = self.belief @ current_transition_matrix

        return chosen_action

    def update_obs(self, observation_index):
        observation_distribution = self.observation_matrix[
            :, observation_index
        ].reshape(-1)
        scaled_belief = self.belief * observation_distribution
        self.belief = scaled_belief / scaled_belief.sum()

    def update_action_probs(self, action_prob):
        self.action_probs = action_prob

    def update(self, action, observation):
        pass

    def change_policy(self, new_transition_matrices, new_possible_rewards):
        if new_transition_matrices is not None:
            self.uniform = False
            self.update_transition_matrix(new_transition_matrices)
            self.possible_rewards = new_possible_rewards
            self.state_expected_reward = (
                self.observation_matrix * self.possible_rewards[None, :]
            )
            self.state_expected_reward = self.state_expected_reward.sum(axis=1)
        else:
            self.uniform = True
        self.reset_belief()

    def update_transition_matrix(self, estimated_transition_matrix):
        self.state_action_transition_matrix = estimated_transition_matrix

    def reset_belief(self):
        self.belief = np.ones(shape=self.num_states) / self.num_states

    def generate_random_transition_matrix(self, min_transition_value):
        # by setting specific design we give more probability to self-loops
        transition_matrix = None
        for state in range(self.num_states):
            state_actions_matrix = np.random.random((self.num_actions, self.num_states))
            state_actions_matrix = (
                state_actions_matrix / state_actions_matrix.sum(axis=1)[:, None]
            )
            if np.any(state_actions_matrix < min_transition_value):
                modified_state_action_matrix = state_actions_matrix.copy()
                modified_state_action_matrix[
                    state_actions_matrix < min_transition_value
                ] += min_transition_value
                modified_state_action_matrix = (
                    modified_state_action_matrix
                    / modified_state_action_matrix.sum(axis=1)[:, None]
                )
                state_actions_matrix = modified_state_action_matrix

            if transition_matrix is None:
                transition_matrix = state_actions_matrix
            else:
                transition_matrix = np.concatenate(
                    [transition_matrix, state_actions_matrix], axis=0
                )

        reshaped_transition_matrix = transition_matrix.reshape(
            (self.num_states, self.num_actions, self.num_states)
        )

        return reshaped_transition_matrix
