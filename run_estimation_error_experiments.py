import numpy as np

from environment.POMDP_env import POMDP
from simulations.estimation_error.simulation_spectral_estimation_error import (
    POMDPSimulationSpectralEstimationError,
)
from utils import load_pomdp


def run_estimation_error_experiment():

    run_settings = "0"

    if (
        run_settings == "0"
    ):  # it corresponds to the case where the POMDP instance is created from scratch
        save_pomdp_info = True
        save_basic_info = True
        save_results = True
        to_load = False
    elif (
        run_settings == "1"
    ):  # it corresponds to the case where the POMDP instance is loaded from memory
        save_pomdp_info = False
        save_basic_info = False
        save_results = True
        to_load = True
    else:
        save_pomdp_info = False
        save_basic_info = False
        save_results = True
        to_load = True

    num_states = 3
    num_actions = 2
    num_observations = 5
    num_experiments = 10

    # estimation error experiment
    num_initial_checkpoint_length = (
        100000  # this is the value associated with the first episode
    )
    num_samples_checkpoint = 100000
    num_checkpoints = 30

    non_normalized_min_transition_value = 0.25
    min_transition_probability_for_policy = 0.25 / num_states
    min_action_prob = 0.25

    # these infos are required if "run_settings == 1", otherwise they can be ignored
    pomdp_to_load_path = f"NeurIPS_experiments/{num_states}states_{num_actions}actions_{num_observations}obs/"
    pomdp_num = 5

    if to_load:
        pomdp = load_pomdp(pomdp_to_load_path, pomdp_num)
    else:
        possible_rewards = np.random.permutation(
            np.linspace(start=0.0, stop=1.0, num=num_observations)
        )
        pomdp = POMDP(
            num_states=num_states,
            num_actions=num_actions,
            num_observations=num_observations,
            possible_rewards=possible_rewards,
            real_min_transition_value=None,
            non_normalized_min_transition_value=non_normalized_min_transition_value,
            state_action_transition_matrix=None,
            observation_matrix=None,
            observation_multiplier=10,
        )

    simulation = POMDPSimulationSpectralEstimationError(
        pomdp,
        loaded_pomdp=to_load,
        pomdp_num=pomdp_num,
        save_pomdp_info=save_pomdp_info,
        save_basic_info=save_basic_info,
        save_results=save_results,
    )

    simulation.run(
        num_experiments=num_experiments,
        initial_checkpoint_length=num_initial_checkpoint_length,
        num_samples_checkpoint=num_samples_checkpoint,
        num_checkpoints=num_checkpoints,
        min_action_prob=min_action_prob,
        min_transition_prob_for_policy=min_transition_probability_for_policy,
    )


if __name__ == "__main__":

    run_estimation_error_experiment()

