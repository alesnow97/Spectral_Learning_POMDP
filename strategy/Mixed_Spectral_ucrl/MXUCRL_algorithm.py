import json
import os

import numpy as np

import utils
from policies.discretized_deterministic_belief_based_policy import (
    DiscretizedBeliefBasedPolicy,
)
from environment.POMDP_env import POMDP
from strategy import strategy_helper
from strategy.Mixed_Spectral_ucrl.SD_strategy_mixed_spectral_for_regret import (
    SpectralStrategyForRegretExp,
)


class MixedSpectralUCRLAlgorithm:
    def __init__(
        self,
        num_states,
        num_actions,
        num_obs,
        pomdp: POMDP,
        ext_v_i_stopping_cond=0.02,
        epsilon_state=0.2,
        delta=0.1,
        num_sampled_belief_MDPs=5,
        discretized_belief_states=None,
        save_path=None,
    ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.ext_v_i_stopping_cond = ext_v_i_stopping_cond
        self.epsilon_state = epsilon_state
        self.delta = delta
        self.num_sampled_belief_MDPs = num_sampled_belief_MDPs

        # this path should be common to all the algorithms
        self.save_path = save_path

        if discretized_belief_states is None:
            self.discretized_belief_states = utils.discretize_continuous_space(
                self.num_states, epsilon=epsilon_state
            )
        else:
            self.discretized_belief_states = discretized_belief_states

        self.len_discretized_beliefs = self.discretized_belief_states.shape[0]

    # the values of samples to discard and samples per estimate refer to the number of couples,
    #  thus the timestamps need to be doubled
    def run(self, T_0, horizon_length, experiment_num, initial_state):
        self.init_policy()
        self.T_0 = T_0

        self.n_for_stopping_condition = np.zeros(shape=self.num_actions)
        self.N_for_stopping_condition = np.zeros(shape=self.num_actions)

        self.mixed_spectral_estimator = SpectralStrategyForRegretExp(
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
        chosen_action = None
        t = 0
        first_state = initial_state
        chosen_action = None
        first_sample_of_new_episode = True

        while t < horizon_length:

            if t == T_0 or (
                t > T_0
                and not np.all(
                    self.n_for_stopping_condition <= self.N_for_stopping_condition
                )
            ):

                self.n_for_stopping_condition[chosen_action] -= 1

                # The dimension will be 3 x n
                episode_collected_samples = np.stack(
                    [
                        episode_collected_actions,
                        episode_collected_observations,
                        episode_collected_rewards,
                    ]
                )

                # update model
                print("ESTIMATION ERROR FOR THE MIXED SPECTRAL UCRL ALGORITHM")
                (
                    estimated_obs,
                    estimated_trans,
                    estimates_info,
                ) = self.mixed_spectral_estimator.update_model_estimates(
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
            if chosen_action is not None:
                self.policy.discretized_belief_update(
                    current_observation=first_obs, past_action=chosen_action
                )
            else:
                self.policy.discretized_belief_update_obs(first_obs)

            chosen_action = self.policy.choose_action()
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
        ) = self.compute_confidence_bound(episode_num=new_episode_num)

        (
            optimistic_transition_model,
            optimistic_observation_model,
            optimistic_belief_based_policy,
        ) = strategy_helper.compute_optimistic_belief_MDP_from_estimates(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            state_action_transition_matrix=estimated_state_action_trans,
            state_observation_matrix=estimated_state_observation_matrix,
            observation_reward_mapping=self.pomdp.possible_rewards,
            ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
            transition_mat_confidence_bounds=transition_mat_confidence_bounds,
            observation_mat_confidence_bound=observation_mat_confidence_bound,
            num_sampled_belief_MDPs=self.num_sampled_belief_MDPs,
            min_transition_value=self.pomdp.min_transition_value,
            discretized_belief_states=self.discretized_belief_states,
            len_discretized_beliefs=self.len_discretized_beliefs,
        )

        self.policy.update_policy_infos(
            state_action_transition_matrix=optimistic_transition_model,
            estimated_state_observation_matrix=optimistic_observation_model,
            belief_action_mapping=optimistic_belief_based_policy,
        )

        self.save_results(
            episode_num=new_episode_num,
            experiment_num=experiment_num,
            # optimistic_belief_action_belief_matrix=optimistic_belief_action_belief_matrix,
            optimistic_transition_matrix_mdp=optimistic_transition_model,
            optimistic_belief_based_policy=optimistic_belief_based_policy,
            estimated_state_observation_matrix=estimated_state_observation_matrix,
            estimated_transition_matrix=estimated_state_action_trans,
            estimated_parameters_info=estimated_parameters_info,
            episode_collected_samples=episode_collected_samples,
        )

    def compute_confidence_bound(self, episode_num: int):
        transition_matrices_confidence_bounds = np.empty(shape=self.num_actions)
        sigma_min_obs = self.pomdp.compute_min_svd(self.pomdp.state_observation_matrix)
        log_term = np.log(self.num_actions * episode_num / self.delta)
        C_T = 10 ** (-11)
        C_O = 10 ** (-11)

        # Set sigma_min_31 and sigma_min_trans_mat
        sigma_min_31_per_action = np.empty(shape=self.num_actions)
        sigma_min_trans_matrices = np.empty(shape=self.num_actions)
        for action in range(self.num_actions):
            corr_mat_31 = self.mixed_spectral_estimator.corr_mat_31[action]
            sigma_min_31_per_action[action] = utils.compute_min_svd(corr_mat_31)

            current_trans_mat = self.pomdp.state_action_transition_matrix[:, action, :]
            sigma_min_trans_matrices[action] = utils.compute_min_svd(current_trans_mat)

        sigma_min_31_index = np.argmin(sigma_min_31_per_action)
        sigma_min_31 = sigma_min_31_per_action[sigma_min_31_index]
        sigma_min_trans_mat_index = np.argmin(sigma_min_trans_matrices)
        sigma_min_trans_mat = sigma_min_trans_matrices[sigma_min_trans_mat_index]

        zeta = (
            sigma_min_31
            * (
                np.sqrt(self.pomdp.min_transition_value)
                * sigma_min_trans_mat
                * sigma_min_obs
            )
            ** 3
        )
        main_term_conf_bound_tr = (
            C_T * self.num_states / (sigma_min_obs * zeta)
        ) * np.sqrt(self.num_actions * episode_num * log_term)

        # Confidence bounds for the transition matrices
        for action in range(self.num_actions):
            N_L_action = self.mixed_spectral_estimator.action_count[action]
            current_conf_bound = main_term_conf_bound_tr / np.sqrt(N_L_action)
            transition_matrices_confidence_bounds[action] = current_conf_bound

        # confidence bounds for the observation matrix
        N_L = np.sum(self.mixed_spectral_estimator.action_count)
        observation_matrix_confidence_bound = (C_O / zeta) * np.sqrt(
            self.num_states * self.num_actions * episode_num * log_term / N_L
        )

        print(
            f"Confidence bounds for the MXUCRL algorithm at episode {episode_num} with {self.mixed_spectral_estimator.action_count} actions counts:"
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
        optimistic_transition_matrix_mdp,
        optimistic_belief_based_policy,
        estimated_state_observation_matrix,
        estimated_transition_matrix,
        estimated_parameters_info,
        episode_collected_samples,
    ):

        result_dict = {
            "optimistic_transition_matrix_mdp": optimistic_transition_matrix_mdp.tolist(),
            "optimistic_belief_action_mapping": optimistic_belief_based_policy.tolist(),
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

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        f = open(
            self.save_path
            + f"/MixedSpectral_{experiment_num}Exp__{episode_num}_Ep.json",
            "w",
        )
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(
            f"Mixed Spectral Results of episode {episode_num} and experiment {experiment_num} have been saved"
        )

    def save_last_episode(self, episode_num, experiment_num, episode_collected_samples):

        result_dict = {"collected_samples": episode_collected_samples.tolist()}

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        f = open(
            self.save_path
            + f"/MixedSpectral_{experiment_num}Exp__{episode_num}_Ep.json",
            "w",
        )
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(
            f"Mixed Spectral Results of episode {episode_num} and experiment {experiment_num} have been saved"
        )

    def init_policy(self):

        # used policy
        self.policy = DiscretizedBeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            initial_discretized_belief=None,
            initial_discretized_belief_index=None,
            discretized_beliefs=self.discretized_belief_states,
            estimated_state_action_transition_matrix=None,
            estimated_state_observation_matrix=None,
            belief_action_mapping=None,
            no_info=True,
        )

    def generate_basic_info_dict(self):
        experiment_basic_info = {
            "discretized_belief_states": self.discretized_belief_states.tolist(),
            "ext_v_i_stopping_cond": self.ext_v_i_stopping_cond,
            "epsilon_state": self.epsilon_state,
        }
        return experiment_basic_info
