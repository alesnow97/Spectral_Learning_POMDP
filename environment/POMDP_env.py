import numpy as np


class POMDP:
    def __init__(
        self,
        num_states,
        num_actions,
        num_observations,
        state_action_transition_matrix,
        observation_matrix,
        possible_rewards,
        real_min_transition_value=None,
        non_normalized_min_transition_value=0.25,
        transition_multiplier=0,
        observation_multiplier=30,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_observations
        self.possible_rewards = possible_rewards

        self.non_normalized_min_transition_value = non_normalized_min_transition_value
        self.min_transition_value = (
            non_normalized_min_transition_value / self.num_states
        )
        self.real_min_transition_value = real_min_transition_value

        self.transition_multiplier = transition_multiplier
        self.observation_multiplier = observation_multiplier
        self.state_action_transition_matrix = state_action_transition_matrix
        self.state_observation_matrix = observation_matrix

        if self.state_action_transition_matrix is not None:
            self.state_action_transition_matrix = state_action_transition_matrix
            self.state_observation_matrix = observation_matrix
        else:
            self.state_action_transition_matrix = self.generate_transition_matrix(
                transition_multiplier=transition_multiplier,
                min_transition_value=self.min_transition_value,
            )
            self.state_observation_matrix = self.generate_observation_matrix(
                observation_multiplier=observation_multiplier
            )

        # self.observation_state_matrix = self.compute_diagonal_observation_state_matrix()
        # self.reference_matrix = self.compute_reference_matrix()
        self.state_reward = self.compute_mean_state_reward()
        # self.reference_matrix_original = self.compute_reference_matrix_original()

    def generate_transition_matrix(self, transition_multiplier, min_transition_value):
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
        self.real_min_transition_value = reshaped_transition_matrix.min()

        for action in range(self.num_actions):
            current_transition_matrix = reshaped_transition_matrix[:, action, :]
            min_svd = self.compute_min_svd(current_transition_matrix)
            print(
                f"MIN SVD OF ACTION TRANSITION MATRIX FOR ACTION {action} IS {min_svd}"
            )

        return reshaped_transition_matrix

    def generate_observation_matrix(self, observation_multiplier):

        perturbation_matrix = np.zeros(shape=(self.num_states, self.num_obs))

        if self.num_states >= self.num_obs:
            for i in range(self.num_states // self.num_obs):
                perturbation_matrix[
                    self.num_obs * i : self.num_obs * (i + 1), :
                ] = observation_multiplier * np.eye(self.num_obs)
        else:
            for i in range(self.num_obs // self.num_states):
                perturbation_matrix[
                    :, self.num_states * i : self.num_states * (i + 1)
                ] = observation_multiplier * np.eye(self.num_states)

        observation_matrix = np.random.random((self.num_states, self.num_obs))

        if self.num_states >= self.num_obs:
            permutation = np.random.permutation(self.num_states)
            permuted_matrix = perturbation_matrix[permutation, :]
        else:
            permutation = np.random.permutation(self.num_obs)
            permuted_matrix = perturbation_matrix[:, permutation]

        observation_matrix += permuted_matrix
        observation_matrix = (
            observation_matrix / observation_matrix.sum(axis=1)[:, None]
        )

        min_svd = self.compute_min_svd(observation_matrix)
        print(f"MIN SVD OF OBSERVATION MATRIX IS {min_svd}")

        return observation_matrix

    def get_next_state(self, state, action):
        return np.random.multinomial(
            n=1, pvals=self.state_action_transition_matrix[state, action]
        ).argmax()

    def get_observation(self, state):
        return np.random.multinomial(
            n=1, pvals=self.state_observation_matrix[state]
        ).argmax()

    def compute_mean_state_reward(self):
        state_reward = np.zeros(shape=self.num_states)
        for state in range(self.num_states):
            mean_reward = np.sum(
                self.possible_rewards * self.state_observation_matrix[state]
            )
            state_reward[state] = mean_reward

        return state_reward

    def compute_reference_matrix(self):
        row_dim = self.num_actions ** 2 * self.num_obs ** 2
        col_dim = self.num_actions ** 2 * self.num_states ** 2
        reference_matrix = np.zeros(shape=(row_dim, col_dim))

        obs_mat = self.state_observation_matrix.T
        kron = np.kron(obs_mat, obs_mat)

        svd = self.compute_min_svd(kron)
        print(f"Basic singular value is {svd}")

        for first_action in range(self.num_actions):
            for second_action in range(self.num_actions):

                row_index = (
                    first_action * self.num_actions + second_action
                ) * self.num_obs ** 2
                col_index = (
                    first_action * self.num_actions + second_action
                ) * self.num_states ** 2

                reference_matrix[
                    row_index : row_index + self.num_obs ** 2,
                    col_index : col_index + self.num_states ** 2,
                ] = kron

        # self.min_svd_reference_matrix = self.compute_min_svd(reference_matrix)
        self.min_svd_reference_matrix = svd
        print(f"Min svd of reference matrix is {self.min_svd_reference_matrix}")

        return reference_matrix

    def generate_pomdp_dict(self):
        save_dict = {}
        save_dict["num_states"] = self.num_states
        save_dict["num_actions"] = self.num_actions
        save_dict["num_obs"] = self.num_obs

        save_dict["transition_multiplier"] = self.transition_multiplier
        save_dict["observation_multiplier"] = self.observation_multiplier
        save_dict[
            "non_normalized_min_transition_value"
        ] = self.non_normalized_min_transition_value
        save_dict["real_min_transition_value"] = self.real_min_transition_value

        save_dict[
            "state_action_transition_matrix"
        ] = self.state_action_transition_matrix.tolist()
        save_dict["observation_matrix"] = self.state_observation_matrix.tolist()
        save_dict["possible_rewards"] = self.possible_rewards.tolist()

        return save_dict
        # # Convert and write JSON object to file
        # with open("sample.json", "w") as outfile:
        #     json.dump(save_dict, outfile)

    def compute_min_svd(self, reference_matrix):
        _, s, _ = np.linalg.svd(reference_matrix, full_matrices=True)
        return min(s)
