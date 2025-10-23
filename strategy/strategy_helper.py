import math

import numpy as np

import utils
from policies.discretized_deterministic_belief_based_policy import (
    DiscretizedBeliefBasedPolicy,
)
from policies.stochastic_memoryless_policy import StochasticMemorylessPolicy
from environment.POMDP_env import POMDP


def compute_belief_action_belief_matrix(
    num_actions,
    num_obs,
    discretized_belief_states,
    len_discretized_beliefs,
    state_action_transition_matrix,
    state_observation_matrix,
):

    belief_action_belief_list = []

    for i in range(len_discretized_beliefs):

        current_belief = discretized_belief_states[i]

        current_belief_list = []

        # future_discretized_belief_index, probability
        for action in range(num_actions):

            current_belief_action_dict = {}

            # just for debug purposes
            obs_probabilities = np.zeros(shape=num_obs)
            for obs in range(num_obs):
                observation_distribution = state_observation_matrix[:, obs].reshape(-1)

                scaled_belief = current_belief * observation_distribution
                obs_probability = np.sum(scaled_belief)
                obs_probabilities[obs] = obs_probability
                # current_transition_matrix = self.pomdp.state_action_transition_matrix[
                #                             :, action, :]
                current_transition_matrix = state_action_transition_matrix[:, action, :]
                transitioned_belief = scaled_belief @ current_transition_matrix

                if transitioned_belief.sum() == 0:
                    num_states = transitioned_belief.shape[0]
                    transitioned_belief = np.ones(shape=num_states) / num_states
                transitioned_belief = transitioned_belief / transitioned_belief.sum()

                (
                    discretized_transitioned_belief,
                    discretized_transitioned_belief_index,
                ) = utils.find_closest_discretized_belief(
                    discretized_belief_states, transitioned_belief
                )

                if discretized_transitioned_belief_index in current_belief_action_dict:
                    current_belief_action_dict[
                        discretized_transitioned_belief_index.tolist()
                    ] += obs_probability
                else:
                    current_belief_action_dict[
                        discretized_transitioned_belief_index.tolist()
                    ] = obs_probability

            current_belief_list.append(current_belief_action_dict)
            # belief_action_belief_matrix[i, action, discretized_transitioned_belief_index] += obs_probability
        belief_action_belief_list.append(current_belief_list)

    return belief_action_belief_list


def compute_optimal_POMDP_policy(
    num_actions,
    discretized_belief_states,
    len_discretized_beliefs,
    ext_v_i_stopping_cond,
    state_reward,
    belief_action_belief_matrix,
):

    u = np.zeros(shape=len_discretized_beliefs)
    best_action_per_u = np.zeros(shape=len_discretized_beliefs)

    du = np.zeros(shape=len_discretized_beliefs)
    du[0], du[-1] = np.inf, -np.inf
    counter = 0
    # print(f"Computing optimal POMDP policy. Len discretized belief is {len_discretized_beliefs}")
    while not du.max() - du.min() < ext_v_i_stopping_cond:
        if counter >= 300:
            print("Convergence issues")
            print(f"Current difference is {du.max() - du.min()}")
            ext_v_i_stopping_cond += 1 + du.max() - du.min()
        counter += 1
        new_u = np.zeros(shape=len_discretized_beliefs)
        for j in range(len_discretized_beliefs):
            current_belief = discretized_belief_states[j]

            current_u_a = np.zeros(shape=num_actions)
            for action in range(num_actions):
                mean_action_reward = np.multiply(current_belief, state_reward).sum()
                current_belief_action_dict = belief_action_belief_matrix[j][action]
                q_function = 0
                for key in current_belief_action_dict:
                    q_function += current_belief_action_dict[key] * u[key]
                # current_belief_action_probs = belief_action_belief_list[j, action, :]
                # future_value_function_array = np.multiply(
                #     current_belief_action_probs, u.reshape(1, -1))
                # future_value_function = np.sum(future_value_function_array)
                current_u_a[action] = mean_action_reward + q_function

            best_action_index = np.argmax(current_u_a)

            future_value_function = current_u_a[best_action_index]
            best_action_per_u[j] = best_action_index
            new_u[j] = future_value_function

        du = new_u - u
        # print(f"Difference between max and min is {du.max() - du.min()}")
        u = new_u

    return best_action_per_u.astype(int)


def inner_maximization(p_sa_hat, confidence_bound_p_sa, rank, min_transition_value):
    """
    Find the best local transition p(.|s, a) within the plausible set of transitions as bounded by the confidence bound for some state action pair.
    Arg:
        p_sa_hat : (n_states)-shaped float array. MLE estimate for p(.|s, a).
        confidence_bound_p_sa : scalar. The confidence bound for p(.|s, a) in L1-norm.
        rank : (n_states)-shaped int array. The sorted list of states in descending order of value.
    Return:
        (n_states)-shaped float array. The optimistic transition p(.|s, a).
    """
    # print('rank', rank)
    p_sa = np.array(p_sa_hat)
    max_value = 1 - min_transition_value * (p_sa.shape[0] - 1)
    p_sa[rank[0]] = min(max_value, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
    rank_dup = list(rank)
    last = rank_dup.pop()
    # Reduce until it is a distribution (equal to one within numerical tolerance)
    while sum(p_sa) > 1 + 1e-9:
        # print('inner', last, p_sa)
        p_sa[last] = max(min_transition_value, 1 - sum(p_sa) + p_sa[last])
        # p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
        last = rank_dup.pop()
    # print('p_sa', p_sa)
    return p_sa


def compute_stochastic_memoryless_policy(
    num_states,
    num_actions,
    min_action_prob,
    state_action_transition_matrix,
    ext_v_i_stopping_cond,
    state_reward,
    confidence_bound,
    min_transition_value,
):

    # Initial values (an optimal 0-step non-stationary policy's values)
    u = np.zeros(num_states)
    new_u = np.zeros(num_states)
    du = np.zeros(num_states)
    du[0], du[-1] = np.inf, -np.inf

    # Optimistic MDP and its epsilon-optimal policy
    p_tilde = np.zeros((num_states, num_actions, num_states))
    best_action_dist_per_state = np.zeros(shape=(num_states, num_actions))

    # print("Computing the optimistic MDP")
    counter = 0
    while not du.max() - du.min() < ext_v_i_stopping_cond:
        if counter == 10 or counter == 1000:
            print("CONVERGENCE ISSUES")
        if counter % 200 == 0 and counter != 0:
            print(f"Current difference is {du.max() - du.min()}")
            print("CONVERGENCE ISSUES")
            # when a convergence issue is present, the stopping condition is increased
            ext_v_i_stopping_cond = 1 + du.max() - du.min()
        counter += 1
        # Sort the states by their values in descending order
        rank = np.argsort(-u)
        for st in range(num_states):
            q_s_per_action = np.zeros(shape=num_actions)
            for ac in range(num_actions):
                # Optimistic transitions
                p_sa_tilde = inner_maximization(
                    state_action_transition_matrix[st, ac],
                    confidence_bound,
                    rank,
                    min_transition_value,
                )
                q_sa = state_reward[st] + (p_sa_tilde * u).sum()
                q_s_per_action[ac] = q_sa
                p_tilde[st, ac] = p_sa_tilde

            best_action_index = np.argmax(q_s_per_action)
            best_action_dist = np.ones(shape=num_actions) * min_action_prob
            best_action_dist[best_action_index] = 1 - (
                (num_actions - 1) * min_action_prob
            )

            best_q_sa = np.sum(np.multiply(best_action_dist, q_s_per_action))
            new_u[st] = best_q_sa
            best_action_dist_per_state[st] = best_action_dist

        du = new_u - u
        u = new_u
        new_u = np.zeros(num_states)

    return p_tilde, best_action_dist_per_state


def compute_optimistic_belief_MDP_from_estimates(
    num_states,
    num_actions,
    num_obs,
    state_action_transition_matrix,
    state_observation_matrix,
    observation_reward_mapping,
    ext_v_i_stopping_cond,
    transition_mat_confidence_bounds,
    observation_mat_confidence_bound,
    num_sampled_belief_MDPs,
    min_transition_value,
    discretized_belief_states,
    len_discretized_beliefs,
):

    sampled_transition_models = []
    sampled_observation_models = []
    optimistic_belief_based_policies = []
    optimistic_average_rewards = np.empty(shape=num_sampled_belief_MDPs)

    for sampled_belief_MDP in range(num_sampled_belief_MDPs):

        # sample belief MDP using confidence bound
        sampled_transition_model, sampled_observation_model = sample_belief_MDP(
            num_states=num_states,
            num_actions=num_actions,
            num_obs=num_obs,
            estimated_state_action_transition_matrix=state_action_transition_matrix,
            estimated_state_observation_matrix=state_observation_matrix,
            transition_matrices_confidence_bounds=transition_mat_confidence_bounds,
            observation_matrix_confidence_bound=observation_mat_confidence_bound,
            min_transition_value=min_transition_value,
        )

        sampled_state_reward = compute_estimated_state_reward(
            num_states=num_states,
            associated_rewards=observation_reward_mapping,
            estimated_observation_matrix=sampled_observation_model,
        )

        # compute belief action belief matrix from the sampled mdp
        belief_action_belief_matrix = compute_belief_action_belief_matrix(
            num_actions=num_actions,
            num_obs=num_obs,
            discretized_belief_states=discretized_belief_states,
            len_discretized_beliefs=len_discretized_beliefs,
            state_action_transition_matrix=sampled_transition_model,
            state_observation_matrix=sampled_observation_model,
        )

        belief_action_mapping = compute_optimal_POMDP_policy(
            num_actions=num_actions,
            discretized_belief_states=discretized_belief_states,
            len_discretized_beliefs=len_discretized_beliefs,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            state_reward=sampled_state_reward,
            belief_action_belief_matrix=belief_action_belief_matrix,
        )

        # generate sampled POMDP
        sampled_pomdp = POMDP(
            num_states=num_states,
            num_actions=num_actions,
            num_observations=num_obs,
            state_action_transition_matrix=sampled_transition_model,
            observation_matrix=sampled_observation_model,
            possible_rewards=observation_reward_mapping,
        )

        # simulate each policy on its own sampled MDP to get the average reward
        average_reward = simulate_belief_based_policy(
            discretized_beliefs=discretized_belief_states,
            belief_based_policy=belief_action_mapping,
            pomdp=sampled_pomdp,
        )

        sampled_transition_models.append(sampled_transition_model)
        sampled_observation_models.append(sampled_observation_model)
        optimistic_belief_based_policies.append(belief_action_mapping)
        optimistic_average_rewards[sampled_belief_MDP] = average_reward

    best_model_index = np.argmax(optimistic_average_rewards)

    return (
        sampled_transition_models[best_model_index],
        sampled_observation_models[best_model_index],
        optimistic_belief_based_policies[best_model_index],
    )


def compute_optimistic_observation_MDP_from_estimates(
    num_states,
    num_actions,
    num_obs,
    state_action_transition_matrix,
    state_observation_matrix,
    observation_reward_mapping,
    ext_v_i_stopping_cond,
    transition_mat_confidence_bounds,
    observation_mat_confidence_bound,
    num_sampled_MDPs,
    min_transition_value,
    min_action_prob,
):

    sampled_transition_models = []
    sampled_observation_models = []
    optimistic_memoryless_policies = []
    optimistic_average_rewards = np.empty(shape=num_sampled_MDPs)

    for sampled_MDP in range(num_sampled_MDPs):

        # sample belief MDP using confidence bound
        sampled_transition_model, sampled_observation_model = sample_belief_MDP(
            num_states=num_states,
            num_actions=num_actions,
            num_obs=num_obs,
            estimated_state_action_transition_matrix=state_action_transition_matrix,
            estimated_state_observation_matrix=state_observation_matrix,
            transition_matrices_confidence_bounds=transition_mat_confidence_bounds,
            observation_matrix_confidence_bound=observation_mat_confidence_bound,
            min_transition_value=min_transition_value,
        )

        # COMPUTE HERE OBSERVATION ACTION OBSERVATION MATRIX
        sampled_obs_action_obs_matrix = compute_observation_action_observation_matrix(
            num_states=num_states,
            num_actions=num_actions,
            num_obs=num_obs,
            estimated_state_observation_mat=sampled_observation_model,
            estimated_state_action_transition_mat=sampled_transition_model,
        )

        # compute optimistic mdp
        (_, optimistic_memoryless_policy,) = compute_stochastic_memoryless_policy(
            num_states=num_obs,
            num_actions=num_actions,
            min_action_prob=min_action_prob,
            state_action_transition_matrix=sampled_obs_action_obs_matrix,
            ext_v_i_stopping_cond=ext_v_i_stopping_cond,
            state_reward=observation_reward_mapping,
            confidence_bound=0,
            min_transition_value=np.min(sampled_obs_action_obs_matrix),
        )

        # generate sampled POMDP
        sampled_pomdp = POMDP(
            num_states=num_states,
            num_actions=num_actions,
            num_observations=num_obs,
            state_action_transition_matrix=sampled_transition_model,
            observation_matrix=sampled_observation_model,
            possible_rewards=observation_reward_mapping,
        )

        # simulate each policy on its own sampled MDP to get the average reward
        average_reward = simulate_memoryless_policy(
            observation_action_mapping=optimistic_memoryless_policy,
            pomdp=sampled_pomdp,
        )

        sampled_transition_models.append(sampled_transition_model)
        sampled_observation_models.append(sampled_observation_model)
        optimistic_memoryless_policies.append(optimistic_memoryless_policy)
        optimistic_average_rewards[sampled_MDP] = average_reward

    best_model_index = np.argmax(optimistic_average_rewards)

    return (
        sampled_transition_models[best_model_index],
        sampled_observation_models[best_model_index],
        optimistic_memoryless_policies[best_model_index],
    )


def compute_estimated_state_reward(
    num_states, associated_rewards, estimated_observation_matrix
):
    state_reward = np.zeros(shape=num_states)
    for state in range(num_states):
        mean_reward = np.sum(associated_rewards * estimated_observation_matrix[state])
        state_reward[state] = mean_reward

    return state_reward


def compute_observation_action_observation_matrix(
    num_states,
    num_actions,
    num_obs,
    estimated_state_observation_mat,
    estimated_state_action_transition_mat,
):
    observation_action_observation_matrix = np.zeros(
        shape=(num_obs, num_actions, num_obs)
    )
    for obs in range(num_obs):
        for action in range(num_actions):
            current_transition_matrix = estimated_state_action_transition_mat[
                :, action, :
            ]
            underlying_state_prob = estimated_state_observation_mat[:, obs]
            if underlying_state_prob.sum() == 0:
                underlying_state_prob = np.ones(shape=num_states) / num_states
            else:
                underlying_state_prob = (
                    underlying_state_prob / underlying_state_prob.sum()
                )

            next_state_prob = current_transition_matrix @ underlying_state_prob
            next_obs_prob = next_state_prob @ estimated_state_observation_mat
            observation_action_observation_matrix[obs, action, :] = next_obs_prob

    return observation_action_observation_matrix


def simulate_belief_based_policy(
    discretized_beliefs, belief_based_policy, pomdp: POMDP
):

    policy = DiscretizedBeliefBasedPolicy(
        num_states=pomdp.num_states,
        num_actions=pomdp.num_actions,
        num_obs=pomdp.num_obs,
        discretized_beliefs=discretized_beliefs,
        estimated_state_action_transition_matrix=pomdp.state_action_transition_matrix,
        estimated_state_observation_matrix=pomdp.state_observation_matrix,
        belief_action_mapping=belief_based_policy,
    )

    # interact with the sampled env
    interaction_horizon = 20000
    cumulative_reward = 0

    state = np.random.choice(pomdp.num_states)
    chosen_action = None

    for t in range(interaction_horizon):
        obs = pomdp.get_observation(state)
        cumulative_reward += pomdp.possible_rewards[obs]

        if chosen_action is not None:
            policy.discretized_belief_update(
                current_observation=obs, past_action=chosen_action
            )
        else:
            policy.discretized_belief_update_obs(obs)

        chosen_action = policy.choose_action()
        state = pomdp.get_next_state(state, chosen_action)

    average_reward = cumulative_reward / interaction_horizon
    print(f"Average Reward for the computed policy is {average_reward}")

    return average_reward


def simulate_memoryless_policy(observation_action_mapping, pomdp: POMDP):

    policy = StochasticMemorylessPolicy(num_actions=pomdp.num_actions, no_info=True)

    policy.update_policy_info(observation_action_probability=observation_action_mapping)

    # interact with the sampled env
    interaction_horizon = 10000
    cumulative_reward = 0

    state = np.random.choice(pomdp.num_states)

    for t in range(interaction_horizon):
        obs = pomdp.get_observation(state)
        cumulative_reward += pomdp.possible_rewards[obs]

        chosen_action = policy.choose_action(obs)
        state = pomdp.get_next_state(state, chosen_action)

    average_reward = cumulative_reward / interaction_horizon
    print(f"Average Reward for the sampled POMDP is {average_reward}")

    return average_reward


def compute_optimistic_MDP(
    num_states,
    num_actions,
    state_action_transition_matrix,
    ext_v_i_stopping_cond,
    state_reward,
    confidence_bound,
    min_transition_value,
):

    """The extended value iteration which finds an optimistic MDP within the plausible set of MDPs and solves for its near-optimal policy."""

    # Initial values (an optimal 0-step non-stationary policy's values)
    u = np.zeros(num_states)
    new_u = np.zeros(num_states)
    du = np.zeros(num_states)
    du[0], du[-1] = np.inf, -np.inf

    # Optimistic MDP and its epsilon-optimal policy
    p_tilde = np.zeros((num_states, num_actions, num_states))
    best_action_per_state = np.zeros(shape=num_states)

    # print("Computing the optimistic MDP")
    counter = 0
    while not du.max() - du.min() < ext_v_i_stopping_cond:
        if counter % 100 == 0 and counter != 0:
            print(f"Current difference is {du.max() - du.min()}")
            print("CONVERGENCE ISSUES")
            # when a convergence issue is present, the stopping condition is increased
            ext_v_i_stopping_cond = 1 + du.max() - du.min()
        counter += 1
        # Sort the states by their values in descending order
        rank = np.argsort(-u)
        for st in range(num_states):
            q_s_per_action = np.zeros(shape=num_actions)
            for ac in range(num_actions):
                # Optimistic transitions
                p_sa_tilde = inner_maximization(
                    state_action_transition_matrix[st, ac],
                    confidence_bound,
                    rank,
                    min_transition_value,
                )
                q_sa = state_reward[st] + (p_sa_tilde * u).sum()
                q_s_per_action[ac] = q_sa
                p_tilde[st, ac] = p_sa_tilde

            best_action_index = np.argmax(q_s_per_action)

            new_u[st] = q_s_per_action[best_action_index]
            best_action_per_state[st] = best_action_index

        du = new_u - u
        u = new_u
        new_u = np.zeros(num_states)
        # print("u", state_value_hat, du.max() - du.min(), epsilon)
    return p_tilde, best_action_per_state


def sample_belief_MDP(
    num_states,
    num_actions,
    num_obs,
    estimated_state_action_transition_matrix,
    estimated_state_observation_matrix,
    transition_matrices_confidence_bounds,
    observation_matrix_confidence_bound,
    min_transition_value,
):

    sampled_transition_model = np.empty(shape=(num_states, num_actions, num_states))
    sampled_observation_model = np.empty(shape=(num_states, num_obs))

    # sample transition matrix
    for action in range(num_actions):
        current_action_confidence_bound = transition_matrices_confidence_bounds[action]
        current_estimated_state_action_transition_matrix = (
            estimated_state_action_transition_matrix[:, action, :]
        )

        current_estimated_state_action_transition_matrix = (
            current_estimated_state_action_transition_matrix - min_transition_value
        )
        current_estimated_state_action_transition_matrix = np.where(
            current_estimated_state_action_transition_matrix < 0,
            0,
            current_estimated_state_action_transition_matrix,
        )

        modified_state_action_transition_matrix = (
            current_estimated_state_action_transition_matrix
            / (current_estimated_state_action_transition_matrix.sum(axis=1))[:, None]
            * (1 - num_states * min_transition_value)
        )
        current_estimated_state_action_transition_matrix = (
            modified_state_action_transition_matrix + min_transition_value
        )

        sampled_action_trans_mat = None
        sampled_transition_matrix_frob_err = +math.inf
        std = 1.0 * np.min([current_action_confidence_bound, 0.50])
        counter = 1
        while sampled_transition_matrix_frob_err > current_action_confidence_bound:
            perturbation_matrix = np.random.normal(
                loc=0, scale=std, size=num_states * num_states
            ).reshape((num_states, num_states))
            sampled_action_trans_mat = (
                current_estimated_state_action_transition_matrix + perturbation_matrix
            )
            sampled_action_trans_mat = np.where(
                sampled_action_trans_mat < min_transition_value,
                min_transition_value,
                sampled_action_trans_mat,
            )
            sampled_action_trans_mat = (
                sampled_action_trans_mat / sampled_action_trans_mat.sum(axis=1)[:, None]
            )
            sampled_matrix_difference = np.absolute(
                sampled_action_trans_mat
                - current_estimated_state_action_transition_matrix
            )
            sampled_transition_matrix_frob_err = np.linalg.norm(
                sampled_matrix_difference, ord="fro"
            )
            std = std / 2
            counter += 1
        print(f"Counter for action {action} is {counter}")
        sampled_transition_model[:, action, :] = sampled_action_trans_mat

    # sample observation matrix
    sampled_observation_matrix_frob_err = +math.inf
    std = 1.0 * observation_matrix_confidence_bound
    counter = 1
    while sampled_observation_matrix_frob_err > observation_matrix_confidence_bound:
        perturbation_matrix = np.random.normal(
            loc=0, scale=std, size=num_states * num_obs
        ).reshape((num_states, num_obs))
        sampled_observation_model = (
            estimated_state_observation_matrix + perturbation_matrix
        )
        sampled_observation_model = np.where(
            sampled_observation_model < 0, 0, sampled_observation_model
        )
        if np.any(sampled_observation_model.sum(axis=1) == 0):
            sampled_observation_model[sampled_observation_model.sum(axis=1) == 0] = (
                np.ones(shape=num_obs) / num_obs
            )
        sampled_observation_model = (
            sampled_observation_model / sampled_observation_model.sum(axis=1)[:, None]
        )
        sampled_matrix_difference = np.absolute(
            sampled_observation_model - estimated_state_observation_matrix
        )
        sampled_observation_matrix_frob_err = np.linalg.norm(
            sampled_matrix_difference, ord="fro"
        )
        std = std / 2
        counter += 1
    print(f"Counter for observation model is {counter}")

    return sampled_transition_model, sampled_observation_model
