import numpy as np

from environment.POMDP_env import POMDP
from simulations.regret.simulation_spectral_regret import POMDPSimulationSpectralRegret
from utils import load_pomdp, load_pomdp_basic_info

def set_algorithm_hyperparameters(
    initial_episode_length=1000, tau_1=1000, tau_2=1000, min_action_prob_smucrl=0.1
):
    seeu_hyperparameters_dict = {
        "initial_episode_length": initial_episode_length,
        "tau_1": tau_1,
        "tau_2": tau_2,
    }
    sm_ucrl_hyperparameters_dict = {
        "initial_episode_length": initial_episode_length,
        "min_action_prob": min_action_prob_smucrl,
    }
    mixed_spectral_ucrl_hyperparameters_dict = {
        "initial_episode_length": initial_episode_length,
    }

    return (
        seeu_hyperparameters_dict,
        sm_ucrl_hyperparameters_dict,
        mixed_spectral_ucrl_hyperparameters_dict,
    )


def run_regret_experiment():

    run_settings = "0"

    if (
        run_settings == "0"
    ):  # it corresponds to the case where the POMDP instance is created from scratch
        save_pomdp_info = True
        save_basic_info = True
        save_results = True
        to_load = False
        to_load_pomdp_basic_info = False
    elif (
        run_settings == "1"
    ):  # it corresponds to the case where the POMDP instance is loaded from memory
        save_pomdp_info = False
        save_basic_info = False
        save_results = True
        to_load = True
        to_load_pomdp_basic_info = False
    else:
        save_pomdp_info = False
        save_basic_info = True
        save_results = True
        to_load = True
        to_load_pomdp_basic_info = False

    run_oracle = True
    run_mixed_spectral = True
    run_smucrl = True
    run_seeu = True

    # POMDP PARAMETERS
    num_states = 3
    num_actions = 3
    num_observations = 4
    non_normalized_min_transition_value = 0.1

    # OTHER PARAMETERS
    ext_v_i_stopping_cond = 0.005
    state_discretization_step = 0.04
    delta = 0.9

    # SEEU
    tau_1 = 10000
    tau_2 = 30000

    # SM UCRL
    min_action_prob_smucrl = 0.02

    ############################################
    # SET EXPERIMENT LENGTH
    initial_episode_length = 30000
    num_episodes = 13
    num_experiments = 10

    # Here we compute the maximum horizon length given the parameters
    # of the different algorithms
    spectral_num_samples = 0
    for episode_num in range(num_episodes):
        spectral_num_samples += tau_1
        spectral_num_samples += int(tau_2 * np.sqrt(episode_num + 1))

    horizon_length = spectral_num_samples
    print(f"Horizon length is {horizon_length}")

    ############################################

    # SET HYPERPARAMETERS
    seeu_hp, smucrl_hp, mixeducrl_hp = set_algorithm_hyperparameters(
        initial_episode_length=initial_episode_length,
        tau_1=tau_1,
        tau_2=tau_2,
        min_action_prob_smucrl=min_action_prob_smucrl,
    )

    ###############################
    # CREATE LAST DIRECTORY NAME
    if run_seeu:
        tau_1, tau_2 = seeu_hp["tau_1"], seeu_hp["tau_2"]
    else:
        tau_1, tau_2 = 0, 0

    if run_smucrl:
        sm_ucrl_min_action_prob = smucrl_hp["min_action_prob"]
        sm_T0 = smucrl_hp["initial_episode_length"]
    else:
        sm_ucrl_min_action_prob = 0
        sm_T0 = 0

    if run_mixed_spectral:
        mxs_T0 = mixeducrl_hp["initial_episode_length"]
    else:
        mxs_T0 = 0

    last_directory_name = f"{num_episodes}Ep_{state_discretization_step}discr_{tau_1}tau1_{tau_2}tau2_{sm_ucrl_min_action_prob}SMAC_{sm_T0}SMT0_{mxs_T0}MXTO"
    ###############################

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

    if to_load_pomdp_basic_info:
        (
            discretized_belief_states,
            real_belief_action_belief,
            real_optimal_belief_action_mapping,
            initial_discretized_belief,
            initial_discretized_belief_index,
        ) = load_pomdp_basic_info(
            state_discretization_step=state_discretization_step,
            pomdp_to_load_path=pomdp_to_load_path,
            pomdp_num=pomdp_num,
        )
    else:
        discretized_belief_states = None
        real_belief_action_belief = None
        real_optimal_belief_action_mapping = None
        initial_discretized_belief = None
        initial_discretized_belief_index = None

    simulation = POMDPSimulationSpectralRegret(
        pomdp,
        loaded_pomdp=to_load,
        pomdp_num=pomdp_num,
        save_pomdp_info=save_pomdp_info,
        save_basic_info=save_basic_info,
        save_results=save_results,
    )

    simulation.run_regret_experiment(
        num_experiments=num_experiments,
        seeu_hyperparameters_dict=seeu_hp,
        sm_ucrl_hyperparameters_dict=smucrl_hp,
        last_directory_name=last_directory_name,
        T_0=initial_episode_length,
        horizon_length=horizon_length,
        ext_v_i_stopping_cond=ext_v_i_stopping_cond,
        state_discretization_step=state_discretization_step,
        delta=delta,
        run_oracle=run_oracle,
        run_seeu=run_seeu,
        run_smucrl=run_smucrl,
        run_mixed_spectral=run_mixed_spectral,
        discretized_belief_states=discretized_belief_states,
        real_belief_action_belief=real_belief_action_belief,
        real_optimal_belief_action_mapping=real_optimal_belief_action_mapping,
        initial_discretized_belief=initial_discretized_belief,
        initial_discretized_belief_index=initial_discretized_belief_index,
    )


if __name__ == "__main__":

    run_regret_experiment()
