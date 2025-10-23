import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t

import utils

plt.style.use("fivethirtyeight")
sns.set_style(rc={"figure.facecolor": "white"})


def ci2(mean, std, n, conf=0.90):
    # Calculate the t-value
    t_value = t.ppf(1 - conf, n - 1)

    # Calculate the margin of error
    margin_error = t_value * std / math.sqrt(n)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound


np.random.seed(200)


def plot_estimation_error(
    num_states: int,
    num_actions: int,
    num_obs: int,
    pomdp_num: int,
    save_figure: bool = False,
    save_csv: bool = False,
):
    base_path = f"NeurIPS_experiments/{num_states}states_{num_actions}actions_{num_obs}obs/pomdp{pomdp_num}"

    fig, axs = plt.subplots(1, 1, figsize=(15, 10))  # , sharex=True, sharey=True)
    # plot_titles = ['(a)', '(b)']

    # first exp
    file_name = "3"
    basic_info_path = base_path + f"/estimation_error/{file_name}.json"
    basic_info_path = os.path.join(utils.get_base_path(), basic_info_path)

    save_figure_path = base_path + "/estimation_error"
    save_figure_path = os.path.join(utils.get_base_path(), save_figure_path)

    f = open(basic_info_path)
    data = json.load(f)
    f.close()

    num_checkpoints = data["num_checkpoints"]
    num_experiments = data["num_experiments"]

    observation_matrix_error_frobenious_norms = np.array(
        data["observation_matrix_error_frobenious_norms"]
    )
    transition_matrix_frobenious_norms = np.array(
        data["transition_matrix_frobenious_norms"]
    )
    num_considered_episodes = 10
    confidence = 10

    # PLOT OBSERVATION MODEL ERROR
    mean_frobenious_observation = observation_matrix_error_frobenious_norms[
        :, :num_considered_episodes
    ].mean(axis=0)
    std_frobenious_observation = observation_matrix_error_frobenious_norms[
        :, :num_considered_episodes
    ].std(axis=0)
    lower_bound_observation, upper_bound_observation = ci2(
        mean_frobenious_observation,
        std_frobenious_observation,
        # observation_matrix_error_frobenious_norms[:, :num_considered_episodes].shape[0]
        confidence,
    )
    x_axis = np.array([(i + 1) for i in range(mean_frobenious_observation.shape[0])])
    axs.plot(x_axis, mean_frobenious_observation, label="Observation Model Error")
    axs.fill_between(
        x_axis, lower_bound_observation, upper_bound_observation, alpha=0.2
    )

    result_dict = {
        "x_axis": x_axis,
        f"mean_frobenious_observation": mean_frobenious_observation,
        f"lower_bound_observation": lower_bound_observation,
        f"upper_bound_observation": upper_bound_observation,
    }

    # PLOT TRANSITION MODEL ERROR
    for action in range(num_actions):
        mean_frobenious = transition_matrix_frobenious_norms[
            :, :num_considered_episodes, action
        ].mean(axis=0)
        std_frobenious = transition_matrix_frobenious_norms[
            :, :num_considered_episodes, action
        ].std(axis=0)
        lower_bound, upper_bound = ci2(mean_frobenious, std_frobenious, confidence)
        axs.plot(x_axis, mean_frobenious, label=f"Action {action} Error")
        axs.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2)
        result_dict[f"mean_frobenious_transition_{action}"] = mean_frobenious
        result_dict[f"lower_bound_transition_{action}"] = lower_bound
        result_dict[f"upper_bound_transition_{action}"] = upper_bound

    plt.legend()
    plt.tight_layout()

    if save_figure:
        plt.savefig(
            os.path.join(save_figure_path, "estimation_error_plot_NON_AGGR.png")
        )  # Save the figure

    first_df = pd.DataFrame(result_dict)

    if save_csv:
        path_to_save_file = os.path.join(save_figure_path, f"{file_name}_NON_AGGR.csv")
        first_df.to_csv(path_to_save_file, index=False)

    plt.show()


def plot_custom_estimation_error(
    num_states: int,
    num_actions: int,
    num_obs: int,
    pomdp_num: int,
    save_figure: bool = False,
    save_csv: bool = False,
):
    base_path = f"NeurIPS_experiments/{num_states}states_{num_actions}actions_{num_obs}obs/pomdp{pomdp_num}"

    # get real transition matrix
    # pomdp_path = os.path.join(utils.get_base_path(), base_path, 'pomdp_info.json')
    # f = open(pomdp_path)
    # pomdp_data = json.load(f)
    # f.close()
    # real_transition_matrix = np.array(pomdp_data["state_action_transition_matrix"])

    fig, axs = plt.subplots(1, 1, figsize=(15, 10))  # , sharex=True, sharey=True)
    # plot_titles = ['(a)', '(b)']

    # first exp
    file_name_1 = "1_first_exps"
    file_name_2 = "2_last_exps"
    basic_info_path_1 = base_path + f"/estimation_error/{file_name_1}.json"
    basic_info_path_2 = base_path + f"/estimation_error/{file_name_2}.json"
    basic_info_path_1 = os.path.join(utils.get_base_path(), basic_info_path_1)
    basic_info_path_2 = os.path.join(utils.get_base_path(), basic_info_path_2)

    save_figure_path = base_path + "/estimation_error"
    save_figure_path = os.path.join(utils.get_base_path(), save_figure_path)

    # FIRST FILE
    f = open(basic_info_path_1)
    data_1 = json.load(f)
    f.close()

    initial_checkpoint_length = data_1["initial_checkpoint_length"]
    num_checkpoints = data_1["num_checkpoints"]
    num_samples_checkpoint = data_1["num_samples_checkpoint"]
    num_experiments_1 = data_1["num_experiments"]

    observation_matrix_error_frobenious_norms_1 = np.array(
        data_1["observation_matrix_error_frobenious_norms"]
    )
    transition_matrix_frobenious_norms_1 = np.array(
        data_1["transition_matrix_frobenious_norms"]
    )

    # SECOND FILE
    f = open(basic_info_path_2)
    data_2 = json.load(f)
    f.close()

    num_experiments_2 = data_2["num_experiments"]

    observation_matrix_error_frobenious_norms_2 = np.array(
        data_2["observation_matrix_error_frobenious_norms"]
    )
    transition_matrix_frobenious_norms_2 = np.array(
        data_2["transition_matrix_frobenious_norms"]
    )

    observation_matrix_error_frobenious_norms = np.concatenate(
        (
            observation_matrix_error_frobenious_norms_1,
            observation_matrix_error_frobenious_norms_2,
        ),
        axis=0,
    )
    transition_matrix_frobenious_norms = np.concatenate(
        (transition_matrix_frobenious_norms_1, transition_matrix_frobenious_norms_2),
        axis=0,
    )

    num_considered_episodes = 30
    confidence = 20

    # PLOT OBSERVATION MODEL ERROR
    mean_frobenious_observation = observation_matrix_error_frobenious_norms[
        :, :num_considered_episodes
    ].mean(axis=0)
    std_frobenious_observation = observation_matrix_error_frobenious_norms[
        :, :num_considered_episodes
    ].std(axis=0)
    lower_bound_observation, upper_bound_observation = ci2(
        mean_frobenious_observation,
        std_frobenious_observation,
        # observation_matrix_error_frobenious_norms[:, :num_considered_episodes].shape[0]
        confidence,
    )
    x_axis = np.array([(i + 1) for i in range(mean_frobenious_observation.shape[0])])
    axs.plot(x_axis, mean_frobenious_observation, label="Observation Model Error")
    axs.fill_between(
        x_axis, lower_bound_observation, upper_bound_observation, alpha=0.2
    )

    result_dict = {
        "x_axis": x_axis,
        f"mean_frobenious_observation": mean_frobenious_observation,
        f"lower_bound_observation": lower_bound_observation,
        f"upper_bound_observation": upper_bound_observation,
    }

    # PLOT TRANSITION MODEL ERROR
    for action in range(num_actions):
        mean_frobenious = transition_matrix_frobenious_norms[
            :, :num_considered_episodes, action
        ].mean(axis=0)
        std_frobenious = transition_matrix_frobenious_norms[
            :, :num_considered_episodes, action
        ].std(axis=0)
        # lower_bound, upper_bound = ci2(mean_frobenious, std_frobenious, transition_matrix_frobenious_norms[:, :num_considered_episodes, action].shape[0])
        lower_bound, upper_bound = ci2(mean_frobenious, std_frobenious, confidence)
        axs.plot(x_axis, mean_frobenious, label=f"Action {action} Error")
        axs.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2)
        result_dict[f"mean_frobenious_transition_{action}"] = mean_frobenious
        result_dict[f"lower_bound_transition_{action}"] = lower_bound
        result_dict[f"upper_bound_transition_{action}"] = upper_bound

    plt.legend()
    plt.tight_layout()

    if save_figure:
        plt.savefig(
            os.path.join(save_figure_path, "estimation_error_plot.png")
        )  # Save the figure

    if save_csv:
        first_df = pd.DataFrame(result_dict)
        path_to_save_file = os.path.join(save_figure_path, f"{file_name_1}.csv")
        first_df.to_csv(path_to_save_file, index=False)

    plt.show()


if __name__ == "__main__":

    num_states = 3
    num_actions = 2
    num_obs = 5
    pomdp_num = 5
    save_figure = False
    save_csv = False

    plot_estimation_error(
        num_states=num_states,
        num_actions=num_actions,
        num_obs=num_obs,
        pomdp_num=pomdp_num,
        save_figure=save_figure,
        save_csv=save_csv,
    )
