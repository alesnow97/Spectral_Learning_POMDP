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


def ci2(mean, std, n, conf=0.85):
    # Calculate the t-value
    t_value = t.ppf(1 - conf, n - 1)

    # Calculate the margin of error
    margin_error = t_value * std / math.sqrt(n)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound


np.random.seed(200)


def plot_regret(
    discretization_step: float,
    min_action_prob: float,
    num_states: int,
    num_actions: int,
    num_obs: int,
    initial_episode_length: int,
    pompd_num: int,
    tau_1: int,
    tau_2: int,
    num_experiments: int,
    num_episodes: int,
    save_figure: bool = False,
    save_csv: bool = False,
):

    base_path = (
        f"NeurIPS_experiments/{num_states}states_{num_actions}actions_{num_obs}obs/"
        f"pomdp{pompd_num}/regret/{num_episodes}Ep_{discretization_step}discr_{tau_1}tau1_{tau_2}"
        f"tau2_{min_action_prob}SMAC_{initial_episode_length}SMT0_{initial_episode_length}MXTO"
    )

    base_path = os.path.join(utils.get_base_path(), base_path)

    fig, axs = plt.subplots(1, 1, figsize=(20, 6))  # , sharex=True, sharey=True)
    # plot_titles = ['(a)', '(b)']

    oracle_samples = None
    mixed_spectral_ucrl_samples = None
    seeu_samples = None
    smucrl_samples = None

    for experiment_num in range(num_experiments):

        current_exp_mixed_spectral_ucrl_data = None
        current_exp_seeu_data = None
        current_exp_smucrl_data = None

        # oracle
        f = open(base_path + f"/ORACLE_{experiment_num}.json")
        data = json.load(f)
        f.close()
        current_exp_oracle_data = np.array(data["collected_samples"])[2]

        if oracle_samples is None:
            oracle_samples = current_exp_oracle_data
        else:
            oracle_samples = np.vstack([oracle_samples, current_exp_oracle_data])

        # MIXED SPECTRAL DATA
        num_episodes_mixed = sum(
            1
            for entry in os.scandir(base_path)
            if entry.is_file() and f"MixedSpectral_{experiment_num}Exp" in entry.name
        )
        for episode_num in range(num_episodes_mixed):
            f = open(
                base_path
                + f"/MixedSpectral_{experiment_num}Exp__{episode_num+1}_Ep.json"
            )
            data = json.load(f)
            f.close()
            episode_mixed_spectral_ucrl_data = np.array(data["collected_samples"])

            if current_exp_mixed_spectral_ucrl_data is None:
                current_exp_mixed_spectral_ucrl_data = episode_mixed_spectral_ucrl_data[
                    2, :
                ].reshape(-1)
            else:
                current_exp_mixed_spectral_ucrl_data = np.hstack(
                    [
                        current_exp_mixed_spectral_ucrl_data,
                        episode_mixed_spectral_ucrl_data[2, :].reshape(-1),
                    ]
                )

        if mixed_spectral_ucrl_samples is None:
            mixed_spectral_ucrl_samples = current_exp_mixed_spectral_ucrl_data
        else:
            mixed_spectral_ucrl_samples = np.vstack(
                [mixed_spectral_ucrl_samples, current_exp_mixed_spectral_ucrl_data]
            )

        # SEEU DATA
        num_episodes_seeu = sum(
            1
            for entry in os.scandir(base_path)
            if entry.is_file() and f"SEEU_{experiment_num}Exp" in entry.name
        )
        for episode_num in range(num_episodes_seeu):
            f = open(base_path + f"/SEEU_{experiment_num}Exp__{episode_num+1}_Ep.json")
            data = json.load(f)
            f.close()
            episode_seeu_data = np.array(data["collected_samples"])

            if current_exp_seeu_data is None:
                current_exp_seeu_data = episode_seeu_data[2, :].reshape(-1)
            else:
                current_exp_seeu_data = np.hstack(
                    [current_exp_seeu_data, episode_seeu_data[2, :].reshape(-1)]
                )

        if seeu_samples is None:
            seeu_samples = current_exp_seeu_data
        else:
            seeu_samples = np.vstack([seeu_samples, current_exp_seeu_data])

        # SMUCRL DATA
        num_episodes_smucrl = sum(
            1
            for entry in os.scandir(base_path)
            if entry.is_file() and f"SMUCRL_{experiment_num}Exp" in entry.name
        )
        for episode_num in range(num_episodes_smucrl):
            f = open(
                base_path + f"/SMUCRL_{experiment_num}Exp__{episode_num+1}_Ep.json"
            )
            data = json.load(f)
            f.close()
            episode_smucrl_data = np.array(data["collected_samples"])

            if current_exp_smucrl_data is None:
                current_exp_smucrl_data = episode_smucrl_data[2, :].reshape(-1)
            else:
                current_exp_smucrl_data = np.hstack(
                    [current_exp_smucrl_data, episode_smucrl_data[2, :].reshape(-1)]
                )

        if smucrl_samples is None:
            smucrl_samples = current_exp_smucrl_data
        else:
            smucrl_samples = np.vstack([smucrl_samples, current_exp_smucrl_data])

    # define min length
    min_num_samples = min(
        oracle_samples.shape[1],
        mixed_spectral_ucrl_samples.shape[1],
        seeu_samples.shape[1],
        smucrl_samples.shape[1],
    )

    x_axis = np.array([i for i in range(min_num_samples)])
    oracle_samples = oracle_samples[:, :min_num_samples]
    mixed_spectral_ucrl_samples = mixed_spectral_ucrl_samples[:, :min_num_samples]
    seeu_samples = seeu_samples[:, :min_num_samples]
    smucrl_samples = smucrl_samples[:, :min_num_samples]

    x_axis_mask = np.array([(i % 10000 == 0) for i in range(min_num_samples)])

    # MIXED SPECTRAL REGRET
    mixed_spectral_ucrl_regret = oracle_samples - mixed_spectral_ucrl_samples
    cumulative_mixed_spectral_ucrl_regret = np.cumsum(
        mixed_spectral_ucrl_regret, axis=1
    )
    mean_cumulated_mixed_spectral_ucrl_regret = np.mean(
        cumulative_mixed_spectral_ucrl_regret, axis=0
    )
    std_cumulative_mixed_spectral_ucrl_regret = np.std(
        cumulative_mixed_spectral_ucrl_regret, axis=0
    )
    lower_bound_mixed, upper_bound_mixed = ci2(
        mean_cumulated_mixed_spectral_ucrl_regret,
        std_cumulative_mixed_spectral_ucrl_regret,
        num_experiments,
    )

    axs.plot(
        mean_cumulated_mixed_spectral_ucrl_regret, "c", label="Mixed Spectral UCRL"
    )
    axs.fill_between(
        x_axis[x_axis_mask],
        lower_bound_mixed[x_axis_mask],
        upper_bound_mixed[x_axis_mask],
        color="c",
        alpha=0.2,
    )

    # SEEU REGRET
    seeu_regret = oracle_samples - seeu_samples
    cumulative_seeu_regret = np.cumsum(seeu_regret, axis=1)
    mean_cumulated_seeu_regret = np.mean(cumulative_seeu_regret, axis=0)
    std_cumlated_seeu_regret = np.std(cumulative_seeu_regret, axis=0)
    lower_bound_seeu, upper_bound_seeu = ci2(
        mean_cumulated_seeu_regret, std_cumlated_seeu_regret, num_experiments
    )

    axs.plot(mean_cumulated_seeu_regret, "r", label="SEEU")
    axs.fill_between(
        x_axis[x_axis_mask],
        lower_bound_seeu[x_axis_mask],
        upper_bound_seeu[x_axis_mask],
        color="r",
        alpha=0.2,
    )

    # SMUCRL REGRET
    smucrl_regret = oracle_samples - smucrl_samples
    cumulative_smucrl_regret = np.cumsum(smucrl_regret, axis=1)
    mean_cumulated_smucrl_regret = np.mean(cumulative_smucrl_regret, axis=0)
    std_cumulated_smucrl_regret = np.std(cumulative_smucrl_regret, axis=0)
    lower_bound_smucrl, upper_bound_smucrl = ci2(
        mean_cumulated_smucrl_regret, std_cumulated_smucrl_regret, num_experiments
    )
    axs.plot(mean_cumulated_smucrl_regret, "g", label="SMUCRL")
    axs.fill_between(
        x_axis[x_axis_mask],
        lower_bound_smucrl[x_axis_mask],
        upper_bound_smucrl[x_axis_mask],
        color="g",
        alpha=0.2,
    )

    axs.set_title(f"Regret Pomdp_num {pompd_num}")
    axs.legend()

    plt.tight_layout()
    if save_figure:
        plt.savefig(os.path.join(base_path, "regret_plot.png"))  # Save the figure
    plt.show()

    if save_csv:
        x_axis_mask = np.array([(i % 10000 == 0) for i in range(int(min_num_samples))])
        path_to_save_file = os.path.join(base_path, "regret.csv")
        result_dict = {"x_axis": x_axis[x_axis_mask] / 10 ** 5}
        result_dict["mean_cumulated_mixed_spectral_ucrl_regret"] = (
            mean_cumulated_mixed_spectral_ucrl_regret[x_axis_mask] / 10 ** 4
        )
        result_dict["lower_bound_mixed"] = lower_bound_mixed[x_axis_mask] / 10 ** 4
        result_dict["upper_bound_mixed"] = upper_bound_mixed[x_axis_mask] / 10 ** 4
        result_dict["mean_cumulated_seeu_regret"] = (
            mean_cumulated_seeu_regret[x_axis_mask] / 10 ** 4
        )
        result_dict["lower_bound_seeu"] = lower_bound_seeu[x_axis_mask] / 10 ** 4
        result_dict["upper_bound_seeu"] = upper_bound_seeu[x_axis_mask] / 10 ** 4
        result_dict["mean_cumulated_smucrl_regret"] = (
            mean_cumulated_smucrl_regret[x_axis_mask] / 10 ** 4
        )
        result_dict["lower_bound_smucrl"] = lower_bound_smucrl[x_axis_mask] / 10 ** 4
        result_dict["upper_bound_smucrl"] = upper_bound_smucrl[x_axis_mask] / 10 ** 4

        pd.DataFrame(result_dict).to_csv(path_to_save_file, index=False)


if __name__ == "__main__":

    save_fig = False
    save_csv = False

    discretization_step = 0.04
    min_action_prob = 0.02

    num_states = 3
    num_actions = 3
    num_obs = 4

    initial_episode_length = 30000
    tau_1 = 10000
    tau_2 = 30000
    pomdp_num = 5
    num_experiments = 5
    num_episodes = 13

    plot_regret(
        discretization_step=discretization_step,
        min_action_prob=min_action_prob,
        num_states=num_states,
        num_actions=num_actions,
        num_obs=num_obs,
        initial_episode_length=initial_episode_length,
        pompd_num=pomdp_num,
        tau_1=tau_1,
        tau_2=tau_2,
        num_experiments=num_experiments,
        num_episodes=num_episodes,
        save_figure=save_fig,
        save_csv=save_csv,
    )
