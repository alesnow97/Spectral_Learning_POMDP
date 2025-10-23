import json
import os
import time

import numpy as np

import utils
from policies.discretized_deterministic_belief_based_policy import (
    DiscretizedBeliefBasedPolicy,
)
from environment.POMDP_env import POMDP
from strategy import strategy_helper


class DiscretizedOracleStrategy:
    def __init__(
        self,
        num_states,
        num_actions,
        num_obs,
        pomdp: POMDP,
        ext_v_i_stopping_cond=0.02,
        epsilon_state=0.2,
        discretized_belief_states=None,
        real_belief_action_belief=None,
        real_optimal_belief_action_mapping=None,
        initial_discretized_belief=None,
        initial_discretized_belief_index=None,
        to_save_basic_info=True,
        basic_info_path=None,
        save_path=None,
    ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.pomdp = pomdp
        self.ext_v_i_stopping_cond = ext_v_i_stopping_cond
        self.epsilon_state = epsilon_state

        self.basic_info_path = basic_info_path
        self.save_path = save_path
        self.to_save_basic_info = to_save_basic_info

        if discretized_belief_states is None:
            self.discretized_belief_states = utils.discretize_continuous_space(
                self.num_states, epsilon=epsilon_state
            )
            self.len_discretized_beliefs = self.discretized_belief_states.shape[0]

            start_time = time.time()
            self.real_belief_action_belief = strategy_helper.compute_belief_action_belief_matrix(
                num_actions=self.num_actions,
                num_obs=self.num_obs,
                discretized_belief_states=self.discretized_belief_states,
                len_discretized_beliefs=self.len_discretized_beliefs,
                state_action_transition_matrix=self.pomdp.state_action_transition_matrix,
                state_observation_matrix=self.pomdp.state_observation_matrix,
            )

            end_time = time.time()
            compute_belief_action_list_time = end_time - start_time
            print(
                f"Compute_belief_action_list time is {compute_belief_action_list_time}"
            )

            start_time = time.time()

            self.real_optimal_belief_action_mapping = (
                strategy_helper.compute_optimal_POMDP_policy(
                    num_actions=self.num_actions,
                    discretized_belief_states=self.discretized_belief_states,
                    len_discretized_beliefs=self.len_discretized_beliefs,
                    ext_v_i_stopping_cond=self.ext_v_i_stopping_cond,
                    state_reward=self.pomdp.state_reward,
                    belief_action_belief_matrix=self.real_belief_action_belief,
                )
            )
            end_time = time.time()
            optimal_POMDP_policy_time = end_time - start_time
            print(f"Optimal_POMDP_policy_time is {optimal_POMDP_policy_time}")

            uniform_initial_belief = np.ones(shape=self.num_states) / self.num_states
            (
                self.initial_discretized_belief,
                self.initial_discretized_belief_index,
            ) = utils.find_closest_discretized_belief(
                self.discretized_belief_states, uniform_initial_belief
            )

            # if a basic_info_path parameter is provided, then basic infos are saved
            if self.to_save_basic_info is True:
                basic_info = self.generate_basic_info_dict()
                self.store_basic_info(basic_info)

        else:
            # data come from a loaded file
            self.discretized_belief_states = discretized_belief_states
            self.len_discretized_beliefs = self.discretized_belief_states.shape[0]
            self.real_belief_action_belief = real_belief_action_belief
            self.real_optimal_belief_action_mapping = real_optimal_belief_action_mapping
            self.initial_discretized_belief = initial_discretized_belief
            self.initial_discretized_belief_index = initial_discretized_belief_index

        self.oracle_policy = DiscretizedBeliefBasedPolicy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            initial_discretized_belief=self.initial_discretized_belief,
            initial_discretized_belief_index=self.initial_discretized_belief_index,
            discretized_beliefs=self.discretized_belief_states,
            estimated_state_action_transition_matrix=self.pomdp.state_action_transition_matrix,
            belief_action_mapping=self.real_optimal_belief_action_mapping,
            estimated_state_observation_matrix=self.pomdp.state_observation_matrix,
            no_info=False,
        )

        self.experiment_info = self.generate_basic_info_dict()

    def run(self, horizon_length, experiment_num, initial_state):
        first_state = initial_state

        # self.estimated_action_state_dist_per_episode = np.zeros(shape=(
        #     num_episodes, self.num_actions, self.num_actions,
        #     self.num_states, self.num_states))

        self.collected_samples = np.empty(shape=(3, horizon_length))
        chosen_action = None

        for t in range(horizon_length):

            first_obs = self.pomdp.get_observation(first_state)
            if chosen_action is not None:
                self.oracle_policy.discretized_belief_update(
                    current_observation=first_obs, past_action=chosen_action
                )
            else:
                self.oracle_policy.discretized_belief_update_obs(first_obs)

            chosen_action = self.oracle_policy.choose_action()
            next_state = self.pomdp.get_next_state(first_state, chosen_action)

            self.collected_samples[:, t] = np.array(
                [
                    int(chosen_action),
                    int(first_obs),
                    float(self.pomdp.possible_rewards[first_obs]),
                ]
            )

            first_state = next_state
            t += 1

        self.save_results(
            experiment_num=experiment_num,
        )

    def save_results(self, experiment_num):

        result_dict = {"collected_samples": self.collected_samples.tolist()}

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        f = open(self.save_path + f"/ORACLE_{experiment_num}.json", "w")
        json_file = json.dumps(result_dict)
        f.write(json_file)
        f.close()
        print(f"Oracle Results of experiment {experiment_num} have been saved")

    # def restore_infos(self,
    #                   loaded_data):
    #     self.oracle_policy.discretized_belief = np.array(loaded_data["discretized_belief"])
    #     self.oracle_policy.discretized_belief_index = loaded_data["discretized_belief_index"]

    def generate_basic_info_dict(self):

        if isinstance(self.initial_discretized_belief_index, int):
            index_to_store = self.initial_discretized_belief_index
        else:
            index_to_store = self.initial_discretized_belief_index.tolist()

        experiment_basic_info = {
            "discretized_belief_states": self.discretized_belief_states.tolist(),
            "real_belief_action_belief": self.real_belief_action_belief,  # .tolist(),
            "real_optimal_belief_action_mapping": self.real_optimal_belief_action_mapping.tolist(),
            "initial_discretized_belief": self.initial_discretized_belief.tolist(),
            "initial_discretized_belief_index": index_to_store,
            "ext_v_i_stopping_cond": self.ext_v_i_stopping_cond,
            "epsilon_state": self.epsilon_state,
        }

        return experiment_basic_info

    def store_basic_info(self, basic_info_file):
        if not os.path.exists(self.basic_info_path):
            f = open(
                os.path.join(
                    self.basic_info_path, f"/{self.epsilon_state}stst_basic_info.json"
                ),
                "w",
            )
            json_file = json.dumps(basic_info_file)
            f.write(json_file)
            f.close()
            print("BASIC INFO have been saved")
