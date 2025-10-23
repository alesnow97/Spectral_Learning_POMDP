import json
import os

import numpy as np

import utils
from policies.belief_based_policy_spectral import BeliefBasedPolicySpectral
from environment.POMDP_env import POMDP
from strategy.Mixed_Spectral_ucrl.SD_strategy_official_for_estimation import (
    SpectralStrategyAllLowRank,
)


class POMDPSimulationSpectralEstimationError:
    def __init__(
        self,
        pomdp: POMDP,
        loaded_pomdp,
        pomdp_num=0,
        save_pomdp_info=False,
        save_basic_info=False,
        save_results=False,
    ):

        self.pomdp = pomdp
        self.loaded_pomdp = loaded_pomdp
        self.pomdp_num = pomdp_num
        self.num_states = self.pomdp.num_states
        self.num_actions = self.pomdp.num_actions
        self.num_obs = self.pomdp.num_obs

        self.save_pomdp_info = save_pomdp_info
        self.save_basic_info = save_basic_info
        self.save_results = save_results

    def generate_dirs(self, experiment_type):

        base_base = "NeurIPS_experiments"

        dir_name = f"{base_base}/{self.num_states}states_{self.num_actions}actions_{self.num_obs}obs"

        if os.path.exists(dir_name):
            if self.loaded_pomdp:
                self.pomdp_dir_path = dir_name + f"/pomdp{self.pomdp_num}"
                self.exp_type_path = self.pomdp_dir_path + f"/{experiment_type}"
                self.new_exp_index = len(os.listdir(self.exp_type_path))
                print(self.new_exp_index)
            else:
                self.new_pomdp_index = len(os.listdir(dir_name))
                self.pomdp_dir_path = dir_name + f"/pomdp{self.new_pomdp_index}"
                os.mkdir(self.pomdp_dir_path)
                est_error_exp_path = self.pomdp_dir_path + "/estimation_error"
                regret_exp_path = self.pomdp_dir_path + "/regret"
                os.mkdir(est_error_exp_path)
                os.mkdir(regret_exp_path)
                self.exp_type_path = self.pomdp_dir_path + f"/{experiment_type}"
                self.new_exp_index = 0
        else:
            os.mkdir(dir_name)
            self.new_pomdp_index = 0
            self.pomdp_dir_path = dir_name + f"/pomdp{self.new_pomdp_index}"
            os.mkdir(self.pomdp_dir_path)
            est_error_exp_path = self.pomdp_dir_path + "/estimation_error"
            regret_exp_path = self.pomdp_dir_path + "/regret"
            os.mkdir(est_error_exp_path)
            os.mkdir(regret_exp_path)
            self.exp_type_path = self.pomdp_dir_path + f"/{experiment_type}"
            self.new_exp_index = 0

    def run(
        self,
        num_experiments: int,
        initial_checkpoint_length: int,
        num_samples_checkpoint: int,
        num_checkpoints: int,
        min_action_prob: float,
        min_transition_prob_for_policy: float,
    ):

        self.generate_dirs(experiment_type="estimation_error")

        result_dict = {
            "initial_checkpoint_length": initial_checkpoint_length,
            "num_checkpoints": num_checkpoints,
            "num_samples_checkpoint": num_samples_checkpoint,
            "num_experiments": num_experiments,
        }

        self.real_action_state_dist = np.zeros(
            shape=(
                num_experiments,
                num_checkpoints,
                self.num_actions,
                self.num_actions,
                self.num_states,
                self.num_states,
            )
        )

        # used policy
        self.policy = BeliefBasedPolicySpectral(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            min_action_prob=min_action_prob,
            observation_matrix=self.pomdp.state_observation_matrix,
            possible_rewards=self.pomdp.possible_rewards,
        )

        # these matrices are used to update the policies
        (
            state_action_matrices_for_policies,
            possible_rewards_for_policies,
        ) = self.generate_set_of_policies(
            num_different_policies=(
                num_checkpoints - 1
            ),  # This -1 is due to the fact that the first policy is uniform
            min_transition_value=min_transition_prob_for_policy,
        )

        observation_distance_matrices_errors = np.zeros(
            shape=(num_experiments, num_checkpoints, self.num_obs, self.num_states)
        )
        observation_matrix_error_frobenious_norms = np.zeros(
            shape=(num_experiments, num_checkpoints)
        )
        observation_matrices_corrected = np.zeros(
            shape=(num_experiments, num_checkpoints), dtype=bool
        )
        transition_matrices_errors = np.zeros(
            shape=(
                num_experiments,
                num_checkpoints,
                self.num_actions,
                self.num_states,
                self.num_states,
            )
        )
        transition_matrix_frobenious_norms = np.zeros(
            shape=(num_experiments, num_checkpoints, self.num_actions)
        )
        transition_matrices_corrected = np.zeros(
            shape=(num_experiments, num_checkpoints, self.num_actions), dtype=bool
        )

        #########################################
        # RUN EXPERIMENTS
        for n in range(num_experiments):
            print("Experiment_n: " + str(n))

            initial_state = np.random.random_integers(low=0, high=self.num_states - 1)

            self.policy.reset_belief()

            self.spectral_strategy = SpectralStrategyAllLowRank(
                num_states=self.num_states,
                num_actions=self.num_actions,
                num_obs=self.num_obs,
                pomdp=self.pomdp,
                policy=self.policy,
                state_action_matrices_for_policies=state_action_matrices_for_policies,
                possible_rewards_for_policies=possible_rewards_for_policies,
            )

            (
                exp_observation_distance,
                exp_observation_frob,
                exp_transition_distance,
                exp_transition_frob,
                exp_obs_corrected,
                exp_trans_corrected,
            ) = self.spectral_strategy.run_estimation_error(
                initial_checkpoint_length=initial_checkpoint_length,
                num_samples_checkpoint=num_samples_checkpoint,
                num_checkpoints=num_checkpoints,
                initial_state=initial_state,
                L=100,
                N=100,
            )

            observation_distance_matrices_errors[n] = exp_observation_distance
            observation_matrix_error_frobenious_norms[n] = exp_observation_frob
            observation_matrices_corrected[n] = exp_obs_corrected
            transition_matrices_errors[n] = exp_transition_distance
            transition_matrix_frobenious_norms[n] = exp_transition_frob
            transition_matrices_corrected[n] = exp_trans_corrected

        ###########################################

        ###########################################
        # STORE RESULTS

        if self.save_pomdp_info:
            pomdp_info_dict = self.pomdp.generate_pomdp_dict()
            f = open(self.pomdp_dir_path + "/pomdp_info.json", "w")
            json_file = json.dumps(pomdp_info_dict)
            f.write(json_file)
            f.close()

        if self.save_results:
            result_dict[
                "observation_distance_matrices_errors"
            ] = observation_distance_matrices_errors.tolist()
            result_dict[
                "observation_matrix_error_frobenious_norms"
            ] = observation_matrix_error_frobenious_norms.tolist()
            result_dict[
                "transition_matrices_errors"
            ] = transition_matrices_errors.tolist()
            result_dict[
                "transition_matrix_frobenious_norms"
            ] = transition_matrix_frobenious_norms.tolist()

            exp_dir_path = os.path.join(
                self.exp_type_path, f"{self.new_exp_index}.json"
            )
            f = open(exp_dir_path, "w")
            json_file = json.dumps(result_dict)
            f.write(json_file)
            f.close()

    def generate_set_of_policies(
        self, num_different_policies: int, min_transition_value: float
    ):

        transition_matrices = np.empty(
            shape=(
                num_different_policies,
                self.num_states,
                self.num_actions,
                self.num_states,
            )
        )
        possible_rewards = np.empty(shape=(num_different_policies, self.num_obs))

        for policy in range(num_different_policies):
            generated_matrix = utils.generate_random_transition_matrix(
                num_states=self.num_states,
                num_actions=self.num_actions,
                min_transition_value=min_transition_value,
            )
            current_possible_rewards = np.random.uniform(
                low=0.0, high=1.0, size=self.num_obs
            )
            transition_matrices[policy] = generated_matrix
            possible_rewards[policy] = current_possible_rewards

        return transition_matrices, possible_rewards
