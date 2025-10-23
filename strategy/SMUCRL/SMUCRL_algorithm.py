import json
import os

import numpy as np

import utils
from policies.stochastic_memoryless_policy import StochasticMemorylessPolicy
from environment.POMDP_env import POMDP
from strategy import strategy_helper
from strategy.SMUCRL.SD_strategy_SMUCRL_for_regret import (
    SpectralStrategyForSMUCRLAlgorithm,
)


class SMUCRLAlgorithm:
    def __init__(
        self,
        num_states,
        num_actions,
        num_obs,
        pomdp: POMDP,
        min_action_prob,
        ext_v_i_stopping_cond=0.02,
        delta=0.1,
        num_sampled_MDPs=5,
        save_path=None,
    ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.delta = delta
        self.min_action_prob = min_action_prob
        self.ext_v_i_stopping_cond = ext_v_i_stopping_cond
        self.num_sampled_MDPs = num_sampled_MDPs

        # this path should be common to all the algorithms
        self.save_path = save_path

    # the values of samples to discard and samples per estimate refer to the number of couples,
    #  thus the timestamps need to be doubled
    def run(self, T_0, horizon_length, experiment_num, initial_state):
        self.init_policy()
        self.T_0 = T_0

        self.n_for_stopping_condition = np.zeros(shape=self.num_actions)
        self.N_for_stopping_condition = np.zeros(shape=self.num_actions)

        self.spectral_estimator = SpectralStrategyForSMUCRLAlgorithm(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
        )

        all_collected_samples = None
        episode_collected_actions = []
        episode_collected_observations = []
        episode_collected_rewards = []
        episode_num = 0
        t = 0
        first_state = initial_state
        chosen_action = None
        first_sample_of_new_episode = True

        # INITIAL INTERACTION FOR t < T_0
        while t < horizon_length:

            # A slightly more precise thing to do here would be to do this control
            # before taking the actions
            if t == T_0 or (
                t > T_0
                and not np.all(
                    self.n_for_stopping_condition <= 2 * self.N_for_stopping_condition
                )
            ):

                self.n_for_stopping_condition[chosen_action] -= 1

                # Dimension will be 3 x n
                episode_collected_samples = np.stack(
                    [
                        episode_collected_actions,
                        episode_collected_observations,
                        episode_collected_rewards,
                    ]
                )

                # update model
                (
                    estimated_obs,
                    estimated_trans,
                    estimates_info,
                ) = self.spectral_estimator.update_model_estimates(
                    samples_from_new_episode=episode_collected_samples
                )

                self.N_for_stopping_condition += self.n_for_stopping_condition
                self.n_for_stopping_condition = np.zeros(shape=self.num_actions)

                if all_collected_samples is None:
                    all_collected_samples = episode_collected_samples
                else:
                    all_collected_samples = np.hstack(
                        (all_collected_samples, episode_collected_samples)
                    )

                episode_collected_actions = []
                episode_collected_observations = []
                episode_collected_rewards = []
                first_sample_of_new_episode = True
                episode_num += 1

                estimated_state_action_trans = np.transpose(
                    estimated_trans, axes=(1, 0, 2)
                )
                estimated_state_observation_matrix = np.transpose(estimated_obs)

                self.change_episode(
                    estimated_state_observation_matrix=estimated_state_observation_matrix,
                    estimated_state_action_trans=estimated_state_action_trans,
                    estimated_parameters_info=estimates_info,
                    new_episode_num=episode_num,
                    experiment_num=experiment_num,
                    episode_collected_samples=episode_collected_samples,
                )

            first_obs = self.pomdp.get_observation(first_state)

            chosen_action = self.policy.choose_action(observation=first_obs)

            next_state = self.pomdp.get_next_state(first_state, chosen_action)

            episode_collected_actions.append(int(chosen_action))
            episode_collected_observations.append(int(first_obs))
            episode_collected_rewards.append(
                float(self.pomdp.possible_rewards[first_obs])
            )

            first_state = next_state
            t += 1

            if first_sample_of_new_episode is True:
                first_sample_of_new_episode = False
            else:
                self.n_for_stopping_condition[chosen_action] += 1

        episode_collected_samples = np.stack(
            [
                episode_collected_actions,
                episode_collected_observations,
                episode_collected_rewards,
            ]
        )

        # LAST EPISODE
        self.save_last_episode(
            episode_num=episode_num + 1,
            experiment_num=experiment_num,
            episode_collected_samples=episode_collected_samples,
        )

    def change_episode(
        self,
        estimated_state_observation_matrix: np.ndarray,
        estimated_state_action_trans: np.ndarray,
        estimated_parameters_info: dict,
        new_episode_num: int,
        experiment_num: int,
        episode_collected_samples: np.ndarray,
    ):

        (
            transition_mat_confidence_bounds,
            observation_mat_confidence_bound,
        ) = self.compute_confidence_bound()

        (
            optimistic_transition_model,
            optimistic_observation_model,
            optimistic_observation_action_mapping,
        ) = strategy_helper.compute_optimistic_observation_MDP_from_estimates(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            state_action_transition_matrix=estimated_state_action_trans,
            state_observation_matrix=estimated_state_observation_matrix,
            observation_reward_mapping=self.pomdp.possible_rewards,
            ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
            transition_mat_confidence_bounds=transition_mat_confidence_bounds,
            observation_mat_confidence_bound=observation_mat_confidence_bound,
            num_sampled_MDPs=self.num_sampled_MDPs,
            min_transition_value=self.pomdp.min_transition_value,
            min_action_prob=self.min_action_prob,
        )

        self.policy.update_policy_info(
            observation_action_probability=optimistic_observation_action_mapping
        )

        self.save_results(
            episode_num=new_episode_num,
            experiment_num=experiment_num,
            estimated_state_observation_matrix=estimated_state_observation_matrix,
            estimated_transition_matrix=estimated_state_action_trans,
            estimated_parameters_info=estimated_parameters_info,
            episode_collected_samples=episode_collected_samples,
        )

    def compute_confidence_bound(self):
        transition_matrices_confidence_bounds = np.empty(shape=self.num_actions)
        sigma_min_obs = self.pomdp.compute_min_svd(self.pomdp.state_observation_matrix)
        min_transition_value = self.pomdp.min_transition_value
        log_term = np.log(1 / self.delta)
        C_T = 10 ** (-15)
        C_O = 10 ** (-15)

        # Set sigma_min_31 and sigma_min_trans_mat
        sigma_min_31_per_action = np.empty(shape=self.num_actions)
        sigma_min_trans_matrices = np.empty(shape=self.num_actions)
        for action in range(self.num_actions):
            corr_mat_31 = self.spectral_estimator.corr_mat_31[action]
            sigma_min_31_per_action[action] = utils.compute_min_svd(corr_mat_31)

            current_trans_mat = self.pomdp.state_action_transition_matrix[:, action, :]
            sigma_min_trans_matrices[action] = utils.compute_min_svd(current_trans_mat)

        sigma_min_31_index = np.argmin(sigma_min_31_per_action)
        sigma_min_31 = sigma_min_31_per_action[sigma_min_31_index]
        sigma_min_trans_mat_index = np.argmin(sigma_min_trans_matrices)
        sigma_min_trans_mat = sigma_min_trans_matrices[sigma_min_trans_mat_index]

        lambda_term = (
            sigma_min_obs
            * self.min_action_prob ** 2
            * sigma_min_31
            * min_transition_value ** (3 / 2)
            * (sigma_min_trans_mat * sigma_min_obs) ** 3
        )

        main_term_conf_bound_tr = (C_T * self.num_states / lambda_term) * np.sqrt(
            self.num_obs * log_term
        )

        # Confidence bounds for the transition matrices
        for action in range(self.num_actions):
            N_L_action = self.spectral_estimator.action_count[action]
            current_conf_bound = main_term_conf_bound_tr / np.sqrt(N_L_action)
            transition_matrices_confidence_bounds[action] = current_conf_bound

        # confidence bounds for the observation matrix
        N_L_max_index = np.argmax(self.spectral_estimator.action_count)
        N_L_max = self.spectral_estimator.action_count[N_L_max_index]
        observation_matrix_confidence_bound = (C_O / lambda_term) * np.sqrt(
            self.num_obs * log_term / N_L_max
        )

        print(
            f"Confidence bounds for the SMUCRL algorithm at episode with {self.spectral_estimator.action_count} actions counts:"
        )
        print(
            f"Confidence Bound of Transition Matrices is {transition_matrices_confidence_bounds}"
        )
        print(
            f"Confidence Bound of Observation Matrix is {observation_matrix_confidence_bound}"
        )

        return (
            transition_matrices_confidence_bounds,
            observation_matrix_confidence_bound,
        )

    def save_results(
        self,
        episode_num,
        experiment_num,
        estimated_state_observation_matrix,
        estimated_transition_matrix,
        estimated_parameters_info,
        episode_collected_samples,
    ):

        result_dict = {
            "estimated_state_observation_matrix": estimated_state_observation_matrix.tolist(),
            "estimated_transition_matrix": estimated_transition_matrix.tolist(),
            "observation_matrix_frobenious_norm": estimated_parameters_info[
                "observation_matrix_frobenious_norm"
            ].tolist(),
            "transition_matrix_frobenious_norm": estimated_parameters_info[
                "transition_matrix_frobenious_norm"
            ].tolist(),
            "collected_samples": episode_collected_samples.tolist(),
        }

        # dir_to_create_path = os.path.join(self.save_path, f"Exxp{experiment_num}__Ep_{episode_num}")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        f = open(
            self.save_path + f"/SMUCRL_{experiment_num}Exp__{episode_num}_Ep.json", "w"
        )
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(
            f"SMUCRL Results of episode {episode_num} and experiment {experiment_num} have been saved"
        )

    def save_last_episode(self, episode_num, experiment_num, episode_collected_samples):

        result_dict = {"collected_samples": episode_collected_samples.tolist()}

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        f = open(
            self.save_path + f"/SMUCRL_{experiment_num}Exp__{episode_num}_Ep.json", "w"
        )
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(
            f"SMUCRL Results of episode {episode_num} and experiment {experiment_num} have been saved"
        )

    def init_policy(self):

        # used policy
        self.policy = StochasticMemorylessPolicy(
            num_actions=self.num_actions, no_info=True
        )
