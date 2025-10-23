import json
import os

import numpy as np

import utils
from policies.discretized_deterministic_belief_based_policy import (
    DiscretizedBeliefBasedPolicy,
)
from strategy import strategy_helper
from strategy.Mixed_Spectral_ucrl.SD_strategy_mixed_spectral_for_regret import (
    SpectralStrategyForRegretExp,
)


class SEEUAlgorithm:
    def __init__(
        self,
        num_states,
        num_actions,
        num_obs,
        pomdp,
        ext_v_i_stopping_cond=0.02,
        epsilon_state=0.2,
        delta=0.1,
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
        self.num_sampled_belief_MDPs = 5

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
    def run(self, tau_1, tau_2, horizon_length, experiment_num, initial_state):

        # if starting_episode_num == 0:
        self.init_policy()

        # COMPUTE NUMBER OF EPISODES
        num_episodes = 0
        num_samples = 0
        while num_samples < horizon_length:
            num_samples += int(tau_1 * self.num_actions)
            num_samples += int(tau_2 * np.sqrt(num_episodes + 1))
            num_episodes += 1
        print("Number of episodes is ", num_episodes)

        self.mixed_spectral_estimator = SpectralStrategyForRegretExp(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
        )

        for episode_num in range(num_episodes):

            (
                episode_collected_samples_exploration,
                last_state,
            ) = self.collect_samples_during_exploration(
                starting_state=initial_state, tau_1=tau_1
            )

            print("ESTIMATION ERROR FOR THE SEEU ALGORITHM")
            (
                estimated_observation_matrix,
                estimated_transition_matrix,
                model_estimates_dict,
            ) = self.mixed_spectral_estimator.update_model_estimates(
                samples_from_new_episode=episode_collected_samples_exploration
            )

            estimated_state_action_trans = np.transpose(
                estimated_transition_matrix, axes=(1, 0, 2)
            )
            estimated_state_observation_matrix = np.transpose(
                estimated_observation_matrix
            )

            self.update_discretized_model(
                estimated_state_observation_matrix=estimated_state_observation_matrix,
                estimated_state_action_trans=estimated_state_action_trans,
                new_episode_num=episode_num,
            )

            (
                episode_collected_samples_exploitation,
                current_state,
            ) = self.collect_samples_during_exploitation(
                starting_state=last_state, tau_2=tau_2, episode_num=episode_num
            )

            episode_collected_samples = np.hstack(
                [
                    episode_collected_samples_exploration,
                    episode_collected_samples_exploitation,
                ]
            )

            episode_num += 1

            self.save_results(
                tau_1=tau_1,
                tau_2=tau_2,
                episode_num=episode_num,
                experiment_num=experiment_num,
                estimated_state_action_trans=estimated_state_action_trans,
                estimated_state_observation_matrix=estimated_state_observation_matrix,
                model_estimates_dict=model_estimates_dict,
                episode_collected_samples=episode_collected_samples,
            )

    def collect_samples_during_exploration(self, starting_state, tau_1):

        first_state = starting_state

        # for convenience these numbers are even
        num_total_samples = int(tau_1)
        episode_collected_samples = np.zeros(shape=(3, num_total_samples))
        uniform_action_distribution = np.ones(shape=self.num_actions) / self.num_actions
        action_count = np.zeros(shape=self.num_actions)

        for sample_num in range(num_total_samples):

            first_obs = self.pomdp.get_observation(first_state)

            chosen_action = np.random.multinomial(
                n=1, pvals=uniform_action_distribution
            ).argmax()

            next_state = self.pomdp.get_next_state(first_state, chosen_action)

            episode_collected_samples[:, sample_num] = [
                chosen_action,
                first_obs,
                self.pomdp.possible_rewards[first_obs],
            ]
            first_state = next_state

            action_count[chosen_action] += 1
        print("The number of chosen actions is ", action_count)

        return episode_collected_samples, first_state

    def collect_samples_during_exploitation(self, starting_state, tau_2, episode_num):

        first_state = starting_state

        # for convenience these numbers are even
        num_total_samples = int(tau_2 * np.sqrt(episode_num + 1))
        episode_collected_samples = np.zeros(shape=(3, num_total_samples))
        chosen_action = None

        for sample_num in range(num_total_samples):

            first_obs = self.pomdp.get_observation(first_state)
            if chosen_action is not None:
                self.policy.discretized_belief_update(
                    current_observation=first_obs, past_action=chosen_action
                )
            else:
                self.policy.discretized_belief_update_obs(first_obs)

            chosen_action = self.policy.choose_action()
            next_state = self.pomdp.get_next_state(first_state, chosen_action)

            episode_collected_samples[:, sample_num] = [
                chosen_action,
                first_obs,
                self.pomdp.possible_rewards[first_obs],
            ]
            first_state = next_state

        return episode_collected_samples, first_state

    def compute_confidence_bound(self):

        transition_matrices_confidence_bounds = np.empty(shape=self.num_actions)
        sigma_min_obs = self.pomdp.compute_min_svd(self.pomdp.state_observation_matrix)
        log_term = np.log(6 * (self.num_obs ** 2 + self.num_obs) / self.delta)
        C = 10 ** (-15)

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

        multiplicative_term_C_12 = (
            1
            + 8
            * np.sqrt(2)
            / (
                self.pomdp.min_transition_value ** 2
                * (sigma_min_trans_mat * sigma_min_obs) ** 3
            )
            + 256
            / self.pomdp.min_transition_value ** 2
            * (sigma_min_trans_mat * sigma_min_obs) ** 3
        )
        C_12 = C / np.sqrt(self.pomdp.min_transition_value) * multiplicative_term_C_12

        C_2 = (
            4
            / sigma_min_obs
            * (np.sqrt(self.num_states + 21 * self.num_states / sigma_min_31))
            * C_12
        )
        # Confidence bounds for the transition matrices
        for action in range(self.num_actions):
            N_L_action = self.mixed_spectral_estimator.action_count[action]
            current_conf_bound = C_2 / np.sqrt(log_term / N_L_action)
            transition_matrices_confidence_bounds[action] = current_conf_bound

        # confidence bounds for the observation matrix
        C_1 = 21 * np.sqrt(self.num_obs) / sigma_min_31 * C_12
        N_L_max_index = np.argmax(self.mixed_spectral_estimator.action_count)
        N_L_max = self.mixed_spectral_estimator.action_count[N_L_max_index]
        observation_matrix_confidence_bound = C_1 * np.sqrt(log_term / N_L_max)

        print(
            f"Confidence bounds for the SEEU algorithm at episode with {self.mixed_spectral_estimator.action_count} actions counts:"
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

    def update_discretized_model(
        self,
        estimated_state_observation_matrix,
        estimated_state_action_trans,
        new_episode_num,
    ):

        (
            transition_mat_conf_bounds,
            observation_mat_conf_bound,
        ) = self.compute_confidence_bound()

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
            transition_mat_confidence_bounds=transition_mat_conf_bounds,
            observation_mat_confidence_bound=observation_mat_conf_bound,
            num_sampled_belief_MDPs=self.num_sampled_belief_MDPs,
            min_transition_value=self.pomdp.min_transition_value,
            discretized_belief_states=self.discretized_belief_states,
            len_discretized_beliefs=self.len_discretized_beliefs,
        )

        self.policy.update_policy_infos(
            state_action_transition_matrix=optimistic_transition_model,
            estimated_state_observation_matrix=estimated_state_observation_matrix,
            belief_action_mapping=optimistic_belief_based_policy,
        )

    def save_results(
        self,
        tau_1,
        tau_2,
        episode_num,
        experiment_num,
        estimated_state_action_trans,
        estimated_state_observation_matrix,
        model_estimates_dict,
        episode_collected_samples,
    ):

        result_dict = {
            "tau_1": tau_1,
            "tau_2": tau_2,
            "estimated_transition_matrix": estimated_state_action_trans.tolist(),
            "estimated_state_observation_matrix": estimated_state_observation_matrix.tolist(),
            "observation_matrix_frobenious_norm": model_estimates_dict[
                "observation_matrix_frobenious_norm"
            ].tolist(),
            "transition_matrix_frobenious_norm": model_estimates_dict[
                "transition_matrix_frobenious_norm"
            ].tolist(),
            "collected_samples": episode_collected_samples.tolist(),
        }

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        f = open(
            self.save_path + f"/SEEU_{experiment_num}Exp__{episode_num}_Ep.json", "w"
        )
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(
            f"SEEU Results of episode {episode_num} and experiment {experiment_num} have been saved"
        )

    # def restore_infos(self,
    #                   loaded_data):
    #     self.policy = DiscretizedBeliefBasedPolicy(
    #         num_states=self.num_states,
    #         num_actions=self.num_actions,
    #         num_obs=self.num_obs,
    #         initial_discretized_belief=None,
    #         initial_discretized_belief_index=None,
    #         discretized_beliefs=self.discretized_belief_states,
    #         estimated_state_action_transition_matrix=np.array(loaded_data["optimistic_transition_matrix_mdp"]),
    #         belief_action_mapping=np.array(loaded_data["optimistic_belief_action_mapping"]),
    #         state_action_observation_matrix=self.pomdp.state_action_observation_matrix,
    #         no_info=False
    #     )
    #
    #     self.policy.discretized_belief = np.array(loaded_data["discretized_belief"])
    #     self.policy.discretized_belief_index = loaded_data["discretized_belief_index"]

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
