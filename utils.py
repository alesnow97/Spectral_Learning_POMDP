import json
import os
import time
from itertools import product

import numpy as np

from environment.POMDP_env import POMDP


def load_pomdp(pomdp_to_load_path, pomdp_num):

    if os.path.exists(pomdp_to_load_path):

        f = open(pomdp_to_load_path + f"/pomdp{pomdp_num}/pomdp_info.json")
        data = json.load(f)
        f.close()

        loaded_pomdp = POMDP(
            num_states=data["num_states"],
            num_actions=data["num_actions"],
            num_observations=data["num_obs"],
            state_action_transition_matrix=np.array(
                data["state_action_transition_matrix"]
            ),
            observation_matrix=np.array(data["observation_matrix"]),
            real_min_transition_value=data["real_min_transition_value"],
            non_normalized_min_transition_value=data[
                "non_normalized_min_transition_value"
            ],
            possible_rewards=np.array(data["possible_rewards"]),
            transition_multiplier=data["transition_multiplier"],
            observation_multiplier=data["observation_multiplier"],
        )

        return loaded_pomdp
    return None


def load_pomdp_basic_info(pomdp_to_load_path, pomdp_num, state_discretization_step):

    base_path = pomdp_to_load_path + f"/pomdp{pomdp_num}"
    basic_info_path = f"/{state_discretization_step}stst_basic_info.json"
    file_to_open_path = base_path + basic_info_path

    if os.path.exists(base_path):

        f = open(file_to_open_path)
        data = json.load(f)

        return (
            np.array(data["discretized_belief_states"]),
            np.array(data["real_belief_action_belief"]),
            np.array(data["real_optimal_belief_action_mapping"]),
            np.array(data["initial_discretized_belief"]),
            data["initial_discretized_belief_index"],
        )
    return None


def compute_second_eigenvalue(transition_matrix):
    evals, _ = np.linalg.eig(transition_matrix)
    evals = evals.real
    evals[np.isclose(evals, 1)] = -1
    return max(evals)


def discretize_continuous_space(array_size, epsilon):

    print("Discretizing continuous state space")
    num_bins = int((1 / epsilon))
    all_combinations = product(range(num_bins + 1), repeat=array_size)
    start_time = time.time()

    good_combinations = np.array(
        list(filter(lambda x: sum(x) == num_bins, all_combinations)), dtype=np.float16
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time is {execution_time}")
    print(good_combinations.shape)
    discretized_beliefs = good_combinations / num_bins

    return discretized_beliefs


def find_closest_discretized_belief(discretized_beliefs, continuous_belief):

    belief_diff = discretized_beliefs - continuous_belief.reshape(1, -1)[:, None]
    belief_diff = belief_diff.reshape(-1, discretized_beliefs.shape[1])
    absolute_diff = np.absolute(belief_diff).sum(axis=1).reshape(-1)
    closest_belief_index = np.argmin(absolute_diff)
    closest_belief = discretized_beliefs[closest_belief_index]
    return closest_belief, closest_belief_index


def generate_random_transition_matrix(num_states, num_actions, min_transition_value):
    # by setting specific design we give more probability to self-loops
    transition_matrix = None
    for state in range(num_states):
        state_actions_matrix = np.random.random((num_actions, num_states))
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
        (num_states, num_actions, num_states)
    )

    return reshaped_transition_matrix


def compute_min_svd(matrix):
    _, s, _ = np.linalg.svd(matrix, full_matrices=True)
    return min(s)


def set_base_path():
    current_path = os.getcwd()
    while not os.path.basename(current_path) == "NeurIPS_Average_Reward_POMDP":
        print("I did not find the path. Current path is {}".format(current_path))
        parent_path = os.path.dirname(current_path)
        current_path = parent_path

    os.chdir(current_path)


def get_base_path():
    current_path = os.getcwd()

    while not os.path.basename(current_path) == "NeurIPS_Average_Reward_POMDP":
        parent_path = os.path.dirname(current_path)
        current_path = parent_path

    return current_path
