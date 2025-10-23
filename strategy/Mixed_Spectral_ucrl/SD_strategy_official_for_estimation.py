import itertools

import numpy as np

from environment.POMDP_env import POMDP
from strategy.Mixed_Spectral_ucrl.SD_utils import (
    compute_estimated_eigenvector_eigenvalue_pair,
)


class SpectralStrategyAllLowRank:
    def __init__(
        self,
        policy,
        num_states,
        num_actions,
        num_obs,
        pomdp: POMDP,
        state_action_matrices_for_policies: np.ndarray,
        possible_rewards_for_policies: np.ndarray,
    ):

        self.policy = policy
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.state_action_matrices_for_policies = state_action_matrices_for_policies
        self.possible_rewards_for_policies = possible_rewards_for_policies

    def run_estimation_error(
        self,
        initial_checkpoint_length,
        num_samples_checkpoint,
        num_checkpoints,
        initial_state,
        L,
        N,
    ):
        self.collect_samples(
            initial_checkpoint_length,
            num_samples_checkpoint,
            num_checkpoints,
            initial_state,
        )

        observation_distance_matrices_errors = np.zeros(
            shape=(num_checkpoints, self.num_obs, self.num_states)
        )
        observation_matrix_error_frobenious_norms = np.zeros(shape=(num_checkpoints))
        transition_matrices_errors = np.zeros(
            shape=(num_checkpoints, self.num_actions, self.num_states, self.num_states)
        )
        transition_matrix_frobenious_norms = np.zeros(
            shape=(num_checkpoints, self.num_actions)
        )
        observation_matrices_corrected = np.zeros(shape=(num_checkpoints), dtype=bool)
        transition_matrices_corrected = np.zeros(
            shape=(num_checkpoints, self.num_actions), dtype=bool
        )

        for checkpoint_index in range(num_checkpoints):
            self.computed_second_and_third_moment(
                initial_checkpoint_len=initial_checkpoint_length,
                checkpoint_index=checkpoint_index,
                num_samples_checkpoint=num_samples_checkpoint,
                first_iteration=checkpoint_index == 0,
            )
            self.whiten_third_order_matrix()
            self.robust_tensor_power_method(L=L, N=N)
            self.compute_reconstruction_error()
            # self.compute_third_moment_estimation_error()
            (
                observation_mat_corr,
                transition_mat_corr,
            ) = self.compute_transition_and_observation_matrices()
            # self.compute_estimation_performance()
            print(f"At Checkpoint {checkpoint_index} we have the following errors")

            (
                observation_distance_matrix_error,
                observation_matrix_error_frobenious_norm,
                transition_matrices_error,
                transition_matrix_frobenious_norm,
            ) = self.compute_model_error()

            print(
                f"Observation distance matrix error is {observation_matrix_error_frobenious_norm}"
            )
            for action in range(self.num_actions):
                print(
                    f"Transition matrix error for action {action} is {transition_matrix_frobenious_norm[action]}"
                )

            observation_distance_matrices_errors[
                checkpoint_index
            ] = observation_distance_matrix_error
            observation_matrix_error_frobenious_norms[
                checkpoint_index
            ] = observation_matrix_error_frobenious_norm
            transition_matrices_errors[checkpoint_index] = transition_matrices_error
            transition_matrix_frobenious_norms[
                checkpoint_index
            ] = transition_matrix_frobenious_norm
            observation_matrices_corrected[checkpoint_index] = observation_mat_corr
            transition_matrices_corrected[checkpoint_index] = transition_mat_corr

        return (
            observation_distance_matrices_errors,
            observation_matrix_error_frobenious_norms,
            transition_matrices_errors,
            transition_matrix_frobenious_norms,
            observation_matrices_corrected,
            transition_matrices_corrected,
        )

    def collect_samples(
        self,
        initial_check_point_len,
        num_samples_checkpoint,
        num_checkpoints,
        initial_state,
    ):
        num_total_samples = initial_check_point_len + num_samples_checkpoint * (
            num_checkpoints - 1
        )
        collected_samples = []

        self.policy.change_policy(
            new_transition_matrices=None, new_possible_rewards=None
        )
        first_state = initial_state
        for i in range(num_total_samples):

            if i % 50000 == 0:
                print(i)

            if (
                i - initial_check_point_len
            ) % num_samples_checkpoint == 0 and i >= initial_check_point_len:
                new_transition_matrices_index = (
                    i - initial_check_point_len
                ) // num_samples_checkpoint
                new_transition_matrices = self.state_action_matrices_for_policies[
                    new_transition_matrices_index
                ]
                new_possible_rewards = self.possible_rewards_for_policies[
                    new_transition_matrices_index
                ]
                self.policy.change_policy(new_transition_matrices, new_possible_rewards)

            first_obs = self.pomdp.get_observation(first_state)
            self.policy.update_obs(first_obs)
            first_action = self.policy.choose_action()
            next_state = self.pomdp.get_next_state(first_state, first_action)

            collected_samples.append((first_obs, first_action, first_state))
            first_state = next_state

        self.all_actions = np.array([elem[1] for elem in collected_samples])
        self.all_obs = np.array([elem[0] for elem in collected_samples])
        self.all_states = np.array([elem[2] for elem in collected_samples])

    # the values of samples to discard and samples per estimate refer to the number of couples,
    #  thus the timesteps need to be doubled
    def computed_second_and_third_moment(
        self,
        initial_checkpoint_len,
        checkpoint_index,
        num_samples_checkpoint,
        first_iteration=False,
    ):

        if first_iteration:
            self.corr_mat_12_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_obs)
            )
            self.corr_mat_21_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_obs)
            )
            self.corr_mat_23_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_obs)
            )
            self.corr_mat_32_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_obs)
            )
            self.corr_mat_13_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_obs)
            )
            self.corr_mat_31_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_obs)
            )

            # third order tensor
            self.tensor_123_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_obs, self.num_obs)
            )

            self.estimated_V1_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_states)
            )
            self.estimated_V3_count = np.zeros(
                shape=(self.num_actions, self.num_obs, self.num_states)
            )

            self.action_state_count = np.zeros(
                shape=(self.num_actions, self.num_states)
            )

        if checkpoint_index == 0:
            starting_index = 0
            ending_index = initial_checkpoint_len
        else:
            starting_index = (
                initial_checkpoint_len + (checkpoint_index - 1) * num_samples_checkpoint
            )
            ending_index = (
                initial_checkpoint_len + checkpoint_index * num_samples_checkpoint
            )

        for i in range(starting_index + 1, ending_index - 1):
            # i represents time t
            v2_obs = self.all_obs[i]
            v2_action = self.all_actions[i]
            v2_state = self.all_states[i]

            v1_obs = self.all_obs[i - 1]

            v3_obs = self.all_obs[i + 1]

            # ESTIMATE VIEWS
            # THESE QUANTITIES ARE NOT OBSERVED BY THE LEARNER SINCE THE STATE IS NOT VISIBLE
            self.estimated_V1_count[v2_action, v1_obs, v2_state] += 1
            self.estimated_V3_count[v2_action, v3_obs, v2_state] += 1

            self.corr_mat_12_count[v2_action, v1_obs, v2_obs] += 1
            self.corr_mat_23_count[v2_action, v2_obs, v3_obs] += 1
            self.corr_mat_13_count[v2_action, v1_obs, v3_obs] += 1

            self.tensor_123_count[v2_action, v1_obs, v2_obs, v3_obs] += 1

            self.action_state_count[v2_action, v2_state] += 1

        ##############################################

        self.corr_mat_12 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs)
        )
        self.corr_mat_21 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs)
        )
        self.corr_mat_23 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs)
        )
        self.corr_mat_32 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs)
        )
        self.corr_mat_13 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs)
        )
        self.corr_mat_31 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs)
        )

        self.tensor_123 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs, self.num_obs)
        )

        self.estimated_V1_from_samples = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_states)
        )
        self.estimated_V3_from_samples = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_states)
        )
        self.real_V3 = np.zeros(shape=(self.num_actions, self.num_obs, self.num_states))
        self.action_state_stationary_distribution = np.zeros(
            shape=(self.num_actions, self.num_states)
        )

        # normalize correlation matrices and produce their inverse
        for action in range(self.num_actions):
            self.corr_mat_12[action] = (
                self.corr_mat_12_count[action] / self.corr_mat_12_count[action].sum()
            )
            self.corr_mat_21[action] = self.corr_mat_12[action].T
            print(
                f"{self.corr_mat_12[action].sum()} and {self.corr_mat_21[action].sum()}"
            )
            self.corr_mat_23[action] = (
                self.corr_mat_23_count[action] / self.corr_mat_23_count[action].sum()
            )
            self.corr_mat_32[action] = self.corr_mat_23[action].T
            print(
                f"{self.corr_mat_23[action].sum()} and {self.corr_mat_32[action].sum()}"
            )
            self.corr_mat_13[action] = (
                self.corr_mat_13_count[action] / self.corr_mat_13_count[action].sum()
            )
            self.corr_mat_31[action] = self.corr_mat_13[action].T
            print(
                f"{self.corr_mat_13[action].sum()} and {self.corr_mat_31[action].sum()}"
            )

            self.tensor_123[action] = (
                self.tensor_123_count[action] / self.tensor_123_count[action].sum()
            )

            self.action_state_stationary_distribution[action] = (
                self.action_state_count[action] / self.action_state_count[action].sum()
            )

            self.estimated_V3_from_samples[action] = (
                self.estimated_V3_count[action]
                / np.sum(self.estimated_V3_count[action], axis=0)[None, :]
            )
            self.real_V3[action] = np.matmul(
                self.pomdp.state_observation_matrix.T,
                self.pomdp.state_action_transition_matrix[:, action, :].T,
            )
            distance_between_real_and_estimated_V3 = np.absolute(
                self.real_V3[action] - self.estimated_V3_from_samples[action]
            )
            frobenious_norm_distance = np.linalg.norm(
                distance_between_real_and_estimated_V3, ord="fro"
            )
            print(
                f"Frobenious norm distance between real and estimated V3 for action {action} is {frobenious_norm_distance}"
            )

        # compute symmetrized M2 and M3 in an optimized way
        self.estimated_sym_M2 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs)
        )
        self.estimated_sym_M3 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_obs, self.num_obs)
        )

        for action in range(self.num_actions):
            # compute symmetrized M2

            left_block = np.matmul(
                self.low_rank(self.corr_mat_32[action]),
                self.low_rank_inverse(self.corr_mat_12[action]),
            )
            right_block = np.matmul(
                self.low_rank_inverse(self.corr_mat_12[action]),
                self.low_rank(self.corr_mat_13[action]),
            )
            first_intermediate = np.matmul(left_block, self.corr_mat_12[action])
            self.estimated_sym_M2[action] = np.matmul(first_intermediate, right_block)

            # compute symmetrized M3
            first_intermediate = np.tensordot(
                self.tensor_123[action], left_block.T, axes=([0], [0])
            )
            first_intermediate = np.transpose(
                first_intermediate, axes=(2, 0, 1)
            )  # (dim_3, dim_2, dim_3)
            second_intermediate = np.tensordot(
                first_intermediate, right_block, axes=([1], [0])
            )
            self.estimated_sym_M3[action] = np.transpose(
                second_intermediate, axes=(0, 2, 1)
            )

    def whiten_third_order_matrix(self):
        # these are intended to be the whitening matrices for view V3
        self.estimated_whitening_matrices = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_states)
        )
        self.estimated_whitened_M3 = np.zeros(
            shape=(self.num_actions, self.num_states, self.num_states, self.num_states)
        )

        # self.real_norm_2_eigenvectors = np.zeros(shape=(self.num_actions, self.num_states, self.num_obs))
        # self.real_norm_2_eigenvalues = np.zeros(shape=(self.num_actions, self.num_states))

        for action in range(self.num_actions):

            # SECOND VERSION OF WHITENING
            self.estimated_sym_M2[action] = (
                self.estimated_sym_M2[action] + self.estimated_sym_M2[action].T
            ) / 2
            w, v = np.linalg.eigh(self.estimated_sym_M2[action])

            # Sort eigenvalues (and eigenvectors) by magnitude
            idx = np.argsort(np.abs(w))[::-1]
            w = w[idx]
            v = v[:, idx]

            # keep only top n_models eigenvalues/eigenvectors
            w = w[: self.num_states]
            v = v[:, : self.num_states]

            # SAFE OPERATION
            if np.any(w < 0):
                w[w < 0] = 1e-10
                print("WARNING: Negative eigenvalues have been clipped to 1e-10")

            # w = np.clip(w, a_min=1e-10, a_max=None)
            # WHITENING MATRIX HAS DIMENSION OxS
            current_whitening_matrix = np.matmul(v, np.diag(w ** (-0.5)))

            identity_mat = np.matmul(
                current_whitening_matrix.T,
                np.matmul(self.estimated_sym_M2[action], current_whitening_matrix),
            )
            # print("This below should be an identity matrix USING THE SECOND WHITENING VERSION")
            # print(identity_mat)

            self.estimated_whitening_matrices[action] = current_whitening_matrix
            first_intermediate = np.tensordot(
                self.estimated_sym_M3[action], current_whitening_matrix, axes=([2], [0])
            )
            second_intermediate = np.tensordot(
                first_intermediate, current_whitening_matrix, axes=([1], [0])
            )
            second_intermediate = np.transpose(second_intermediate, axes=(0, 2, 1))
            third_intermediate = np.tensordot(
                second_intermediate, current_whitening_matrix, axes=([0], [0])
            )
            self.estimated_whitened_M3[action] = np.transpose(
                third_intermediate, axes=(2, 0, 1)
            )

    def robust_tensor_power_method(self, L, N):

        self.estimated_whitened_eigenvectors = np.zeros(
            shape=(self.num_actions, self.num_states, self.num_states)
        )
        self.estimated_whitened_eigenvalues = np.zeros(
            shape=(self.num_actions, self.num_states)
        )

        for action in range(self.num_actions):

            current_action_tensor = self.estimated_whitened_M3[action]

            for component in range(self.num_states):
                (
                    estimated_eigenvector,
                    estimated_eigenvalue,
                ) = compute_estimated_eigenvector_eigenvalue_pair(
                    num_dimensions=self.num_states,
                    num_initial_vectors=L,
                    num_iterations=N,
                    tensor=current_action_tensor,
                )

                outer_product = np.outer(estimated_eigenvector, estimated_eigenvector)
                selected_part = (
                    estimated_eigenvalue
                    * outer_product[:, :, None]
                    * estimated_eigenvector
                )

                deflated_tensor = current_action_tensor - selected_part
                current_action_tensor = deflated_tensor

                self.estimated_whitened_eigenvectors[
                    action, component
                ] = estimated_eigenvector
                self.estimated_whitened_eigenvalues[
                    action, component
                ] = estimated_eigenvalue

    def compute_reconstruction_error(self):
        for action in range(self.num_actions):

            reconstructed_tensor = np.zeros(
                shape=(self.num_states, self.num_states, self.num_states)
            )
            for state in range(self.num_states):
                eigenvector = self.estimated_whitened_eigenvectors[action, state]
                eigenvalue = self.estimated_whitened_eigenvalues[action, state]
                outer_product = np.outer(eigenvector, eigenvector)
                reconstructed_tensor += (
                    eigenvalue * outer_product[:, :, None] * eigenvector
                )

            distance_matrix = np.absolute(
                self.estimated_whitened_M3[action].reshape(-1)
                - reconstructed_tensor.reshape(-1)
            )
            # print(self.estimated_whitened_M3[action])
            # print(reconstructed_tensor)
            distance_value = abs(np.sum(distance_matrix))
            # print(f"distance_value is {distance_value}")

    def compute_transition_and_observation_matrices(self):

        self.estimated_V3 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_states)
        )
        self.estimated_observation_mat = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_states)
        )
        self.estimated_transition_mat = np.zeros(
            shape=(self.num_actions, self.num_states, self.num_states)
        )
        self.aggregated_observation_matrix = np.zeros(
            shape=(self.num_obs, self.num_states)
        )

        permuted_V3 = np.zeros(shape=(self.num_actions, self.num_obs, self.num_states))
        permuted_observation_matrices = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_states)
        )

        observation_matrix_corrected = False
        for action in range(self.num_actions):

            current_whitening_matrix = self.estimated_whitening_matrices[action]
            current_pseudo_inverse = np.linalg.pinv(current_whitening_matrix.T)

            for state in range(self.num_states):
                eigenvector = self.estimated_whitened_eigenvectors[action, state]
                eigenvalue = self.estimated_whitened_eigenvalues[action, state]

                self.estimated_V3[action, :, state] = eigenvalue * np.matmul(
                    current_pseudo_inverse, eigenvector
                )
                self.estimated_observation_mat[action, :, state] = np.matmul(
                    self.low_rank(self.corr_mat_21[action]),
                    np.matmul(
                        self.low_rank_inverse(self.corr_mat_31[action]),
                        self.estimated_V3[action, :, state],
                    ),
                )

            permuted_V3[action] = self.find_best_permutation_debug(
                self.estimated_V3[action], self.real_V3[action]
            )
            distance_matrix = np.absolute(permuted_V3[action] - self.real_V3[action])
            frobenious_norm_distance = np.linalg.norm(distance_matrix, ord="fro")
            # print(f"Frobenious norm distance between real and RECONSTRUCTED V3 for action {action} is {frobenious_norm_distance}")

            if np.any(self.estimated_observation_mat[action] < 0):
                # print(f"ERROR: Observation matrix WITH LOW RANK for action {action} has negative values")
                observation_matrix_corrected = True
                self.estimated_observation_mat[action] = np.clip(
                    self.estimated_observation_mat[action], a_min=0, a_max=None
                )

            if np.any(self.estimated_observation_mat[action].sum(axis=0) == 0):
                mask = self.estimated_observation_mat[action].sum(axis=0) == 0
                self.estimated_observation_mat[action][:, mask] = 1
            self.estimated_observation_mat[action] = (
                self.estimated_observation_mat[action]
                / self.estimated_observation_mat[action].sum(axis=0)[None, :]
            )

            permuted_observation_matrices[action] = self.find_best_permutation_debug(
                self.estimated_observation_mat[action],
                self.pomdp.state_observation_matrix.T,
            )
            distance_matrix = np.absolute(
                permuted_observation_matrices[action]
                - self.pomdp.state_observation_matrix.T
            )
            distance_matrix_frobenious = np.linalg.norm(distance_matrix, ord="fro")
            # print(f"Observation Distance Matrix LOW RANK for action {action} is {distance_matrix_frobenious}")

        (
            permuted_observation_matrices,
            permuted_estimated_V3,
        ) = self._find_best_permutation()

        # Aggregate different observation matrices
        self.num_occurrencies_actions = np.zeros(shape=(self.num_actions))
        self.num_occurrencies_actions = self.action_state_count.sum(axis=1)
        action_percentage = (
            self.num_occurrencies_actions / self.num_occurrencies_actions.sum()
        )
        for action in range(self.num_actions):
            self.aggregated_observation_matrix += (
                permuted_observation_matrices[action] * action_percentage[action]
            )

        for action in range(self.num_actions):
            self.estimated_transition_mat[action] = np.matmul(
                np.linalg.pinv(self.aggregated_observation_matrix),
                permuted_estimated_V3[action],
            ).T

        # SANITY CHECK
        transition_matrices_corrected = np.zeros(shape=(self.num_actions), dtype=bool)
        for action in range(self.num_actions):
            if np.any(self.estimated_transition_mat[action] < 0):
                # print("ERROR: Transition matrix has negative values")
                transition_matrices_corrected[action] = True
                self.estimated_transition_mat[action] = np.clip(
                    self.estimated_transition_mat[action], a_min=1e-10, a_max=None
                )
            self.estimated_transition_mat[action] = (
                self.estimated_transition_mat[action]
                / self.estimated_transition_mat[action].sum(axis=1)[:, None]
            )

        return observation_matrix_corrected, transition_matrices_corrected

    def _find_best_permutation(self):

        permuted_observation_matrices = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_states)
        )
        permuted_estimated_V3 = np.zeros(
            shape=(self.num_actions, self.num_obs, self.num_states)
        )

        real_observation_matrix = self.pomdp.state_observation_matrix.T
        data = [i for i in range(real_observation_matrix.shape[1])]
        observation_matrix_error_metric = lambda x: np.sum(
            np.absolute(real_observation_matrix.reshape(-1) - x.reshape(-1))
        )

        for action in range(self.num_actions):
            current_estimated_observation_matrix = self.estimated_observation_mat[
                action
            ]

            # Generate all permutations of the data
            permutations = itertools.permutations(data)

            # Find the permutation with the lowest error
            min_error = float("inf")
            min_permutation = None
            for permutation in permutations:
                current_perm_matrix = current_estimated_observation_matrix[
                    :, permutation
                ]
                error = observation_matrix_error_metric(current_perm_matrix)
                if error < min_error:
                    min_error = error
                    min_permutation = permutation

            # print("Permutation with lowest error:", min_permutation)
            # print("Lowest error:", min_error)

            permuted_observation_matrices[
                action
            ] = current_estimated_observation_matrix[:, min_permutation]
            permuted_estimated_V3[action] = self.estimated_V3[action][
                :, min_permutation
            ]

        return permuted_observation_matrices, permuted_estimated_V3

    def find_best_permutation_debug(self, estimated, real):

        permuted_quantity = np.zeros(shape=(self.num_obs, self.num_states))
        data = [i for i in range(real.shape[1])]

        error_metric = lambda x: np.sum(np.absolute(real.reshape(-1) - x.reshape(-1)))

        # Generate all permutations of the data
        permutations = itertools.permutations(data)

        # Find the permutation with the lowest error
        min_error = float("inf")
        min_permutation = None
        for permutation in permutations:
            current_perm_matrix = estimated[:, permutation]
            error = error_metric(current_perm_matrix)
            if error < min_error:
                min_error = error
                min_permutation = permutation

            # print("Permutation with the lowest error:", min_permutation)
            # print("Lowest error:", min_error)

            permuted_quantity = estimated[:, min_permutation]

        return permuted_quantity

    def compute_model_error(self):

        observation_distance_matrix_error = np.absolute(
            self.aggregated_observation_matrix - self.pomdp.state_observation_matrix.T
        )
        observation_matrix_error_frobenious_norm = np.linalg.norm(
            observation_distance_matrix_error, ord="fro"
        )

        transition_matrices_error = np.zeros(
            shape=(self.num_actions, self.num_states, self.num_states)
        )
        transition_matrix_frobenious_norm = np.zeros(shape=self.num_actions)

        for action in range(self.num_actions):
            transition_matrices_error[action] = np.absolute(
                self.estimated_transition_mat[action]
                - self.pomdp.state_action_transition_matrix[:, action, :]
            )
            transition_matrix_frobenious_norm[action] = np.linalg.norm(
                transition_matrices_error[action], ord="fro"
            )

        return (
            observation_distance_matrix_error,
            observation_matrix_error_frobenious_norm,
            transition_matrices_error,
            transition_matrix_frobenious_norm,
        )

    def low_rank_inverse(self, original_matrix):
        low_rank_matrix = low_rank_approximation(original_matrix, self.num_states)
        return np.linalg.pinv(low_rank_matrix)

    def low_rank(self, original_matrix):
        return low_rank_approximation(original_matrix, self.num_states)


def low_rank_approximation(A, rank):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_truncated = np.zeros_like(S)
    S_truncated[:rank] = S[:rank]
    return U @ np.diag(S_truncated) @ Vt
