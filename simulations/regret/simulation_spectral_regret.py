import json
import os

import numpy as np

from environment.POMDP_env import POMDP
from strategy.Mixed_Spectral_ucrl.MXUCRL_algorithm import MixedSpectralUCRLAlgorithm
from strategy.Oracle.discretizedOracleStrategy import DiscretizedOracleStrategy
from strategy.SEEU.SEEU_algorithm import SEEUAlgorithm
from strategy.SMUCRL.SMUCRL_algorithm import SMUCRLAlgorithm


class POMDPSimulationSpectralRegret:
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

    def generate_dirs(self, last_directory_name):

        base_base = "NeurIPS_experiments"

        dir_name = f"{base_base}/{self.num_states}states_{self.num_actions}actions_{self.num_obs}obs"

        if os.path.exists(dir_name):
            if self.loaded_pomdp:
                self.pomdp_dir_path = dir_name + f"/pomdp{self.pomdp_num}"
                self.exp_type_path = self.pomdp_dir_path + "/regret"
                self.save_path = self.exp_type_path + f"/{last_directory_name}"
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                print(f"Save path is {self.save_path}")
            else:
                self.new_pomdp_index = len(os.listdir(dir_name))
                self.pomdp_dir_path = dir_name + f"/pomdp{self.new_pomdp_index}"
                os.mkdir(self.pomdp_dir_path)
                est_error_exp_path = self.pomdp_dir_path + "/estimation_error"
                regret_exp_path = self.pomdp_dir_path + "/regret"
                os.mkdir(est_error_exp_path)
                os.mkdir(regret_exp_path)
                self.exp_type_path = self.pomdp_dir_path + f"/regret"
                self.save_path = self.exp_type_path + f"/{last_directory_name}"
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                print(f"Save path is {self.save_path}")
        else:
            os.mkdir(dir_name)
            self.new_pomdp_index = 0
            self.pomdp_dir_path = dir_name + f"/pomdp{self.new_pomdp_index}"
            os.mkdir(self.pomdp_dir_path)
            est_error_exp_path = self.pomdp_dir_path + "/estimation_error"
            regret_exp_path = self.pomdp_dir_path + "/regret"
            os.mkdir(est_error_exp_path)
            os.mkdir(regret_exp_path)
            self.exp_type_path = self.pomdp_dir_path + f"/regret"
            self.save_path = self.exp_type_path + f"/{last_directory_name}"
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            print(f"Save path is {self.save_path}")

    def run_regret_experiment(
        self,
        num_experiments: int,
        T_0: int,
        horizon_length: int,
        ext_v_i_stopping_cond: float,
        state_discretization_step: float,
        delta: float,
        run_oracle: bool,
        run_seeu: bool,
        run_smucrl: bool,
        run_mixed_spectral: bool,
        seeu_hyperparameters_dict: dict,
        sm_ucrl_hyperparameters_dict: dict,
        last_directory_name: str = None,
        discretized_belief_states: np.ndarray = None,
        real_belief_action_belief: np.ndarray = None,
        real_optimal_belief_action_mapping: np.ndarray = None,
        initial_discretized_belief: np.ndarray = None,
        initial_discretized_belief_index: int = None,
    ):

        self.generate_dirs(last_directory_name=last_directory_name)

        self.state_discretization_step = state_discretization_step
        self.sm_ucrl_min_action_prob = sm_ucrl_hyperparameters_dict["min_action_prob"]

        pomdp_info_dict = self.pomdp.generate_pomdp_dict()

        self.oracle_strategy = DiscretizedOracleStrategy(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            epsilon_state=state_discretization_step,
            discretized_belief_states=discretized_belief_states,
            real_belief_action_belief=real_belief_action_belief,
            real_optimal_belief_action_mapping=real_optimal_belief_action_mapping,
            initial_discretized_belief=initial_discretized_belief,
            initial_discretized_belief_index=initial_discretized_belief_index,
            to_save_basic_info=self.save_basic_info,
            basic_info_path=self.pomdp_dir_path,
            save_path=self.save_path,
        )

        self.mixed_spectral_ucrl = MixedSpectralUCRLAlgorithm(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            epsilon_state=state_discretization_step,
            delta=delta,
            discretized_belief_states=self.oracle_strategy.discretized_belief_states,
            save_path=self.save_path,
        )

        self.seeu_algorithm = SEEUAlgorithm(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            epsilon_state=state_discretization_step,
            delta=delta,
            discretized_belief_states=self.oracle_strategy.discretized_belief_states,
            save_path=self.save_path,
        )

        self.sm_ucrl_strategy = SMUCRLAlgorithm(
            num_states=self.num_states,
            num_actions=self.num_actions,
            num_obs=self.num_obs,
            pomdp=self.pomdp,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            min_action_prob=self.sm_ucrl_min_action_prob,
            save_path=self.save_path,
        )

        for n in range(num_experiments):
            print("Experiment_n: " + str(n))

            initial_state = np.random.multinomial(
                1, np.ones(shape=self.num_states) / self.num_states
            ).argmax()

            if run_oracle is True:
                self.oracle_strategy.run(
                    horizon_length=horizon_length,
                    experiment_num=n,
                    initial_state=initial_state,
                )

            if run_mixed_spectral is True:
                self.mixed_spectral_ucrl.run(
                    T_0=T_0,
                    horizon_length=horizon_length,
                    experiment_num=n,
                    initial_state=initial_state,
                )

            if run_seeu is True:
                self.seeu_algorithm.run(
                    tau_1=seeu_hyperparameters_dict["tau_1"],
                    tau_2=seeu_hyperparameters_dict["tau_2"],
                    horizon_length=horizon_length,
                    experiment_num=n,
                    initial_state=initial_state,
                )

            if run_smucrl is True:
                self.sm_ucrl_strategy.run(
                    T_0=T_0,
                    horizon_length=horizon_length,
                    experiment_num=n,
                    initial_state=initial_state,
                )

        if not self.loaded_pomdp and self.save_pomdp_info:
            f = open(self.pomdp_dir_path + "/pomdp_info.json", "w")
            json_file = json.dumps(pomdp_info_dict)
            f.write(json_file)
            f.close()

    # def restore_infos(self, T_0, starting_episode_num, run_oracle, run_optimistic, experiment_num):
    #
    #     self.new_exp_index = len(os.listdir(self.exp_type_path))
    #
    #     basic_info_path = f"/{self.state_discretization_step}stst_{self.min_action_prob}_minac/{T_0}_init"
    #     dir_to_read_path = self.exp_type_path + basic_info_path
    #
    #     oracle_starting_state = None
    #     optimistic_starting_state = None
    #     if run_oracle is True:
    #         oracle_file_to_read_path = dir_to_read_path + f'/oracle_{starting_episode_num-1}Ep_{experiment_num}Exp.json'
    #         f = open(oracle_file_to_read_path)
    #         data = json.load(f)
    #         self.oracle_strategy.restore_infos(loaded_data=data)
    #         oracle_starting_state = data["starting_state"]
    #
    #     if run_optimistic is True:
    #         optimistic_file_to_read_path = dir_to_read_path + f'/optimistic_{starting_episode_num - 1}Ep_{experiment_num}Exp.json'
    #         f = open(optimistic_file_to_read_path)
    #         data = json.load(f)
    #         self.optimistic_algorithm_strategy.restore_infos(loaded_data=data)
    #         optimistic_starting_state = data["starting_state"]
    #
    #     return oracle_starting_state, optimistic_starting_state
