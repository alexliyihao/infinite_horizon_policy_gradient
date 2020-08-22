"""
This implementation are same as the optimization_grad.py,
the output is a little bit different(this one is numpy record of all generated).
"""

from utils import *
from implementations import *
from tqdm.notebook import tqdm

def variance_estimate_benchmark(policy_0, policy_1, env, rolling_out_start_recording, rolling_out_step, n_rolling_out):
    """
    the full wrapper for evaluating the benchmark ver. policy gradient
    input:
      policy_0, policy_1: torch.nn.module, the policy
      env: environment, the environment used
      rolling_out_start_recording: int, the step start recording
      rolling_out_step: int the step recording stops
      n_rolling_out: the number of policy gradient generated
    output:
      policy_gradient_record_np: numpy.array, the array of gradient generated, scaled by 1/time_horizon
    """
    # this two part in the same under in-policy setting, i.e. the second parenthesis are all 1, deterministic
    policy_prob_ratio = get_policy_prob_ratio(policy_1 = policy_1,
                                              policy_0 = policy_0,
                                              state_space = 6)

    # the grad of log_pi w.r.t. theta, deterministic
    grad_log = [[get_log_pi_gradient(policy = policy_1, state = i, action=j) for j in range(2)] for i in range(6)]

    policy_gradient_record = []
    # for each loop, roll out a trajectory in policy_0
    for time_step_ in tqdm(range(n_rolling_out), desc = "benchmark: ", leave=True):
        # rollout policy 0
        triple_pi_0, reward_list = roll_out_procedure(policy = policy_0,
                                                      environment = env,
                                                      record_from = rolling_out_start_recording,
                                                      rolling_length = rolling_out_step,
                                                      ihp = False)

        gradient = compute_policy_gradient_benchmark(triple_pi_0,
                                                     policy_prob_ratio,
                                                     grad_log,
                                                     env)

        policy_gradient_record.append(gradient)

    # save the gradient as numpy array from
    policy_gradient_record_np = np.vstack([i.cpu().numpy() for i in policy_gradient_record])

    # divide by time step
    return policy_gradient_record_np/(rolling_out_step - rolling_out_start_recording)

def variance_estimate_causality(policy_0, policy_1, env, rolling_out_start_recording, rolling_out_step, n_rolling_out):
    """
    the full wrapper for evaluating the causality ver. policy gradient
    input:
      policy_0, policy_1: torch.nn.module, the policy
      env: environment, the environment used
      rolling_out_start_recording: int, the step start recording
      rolling_out_step: int the step recording stops
      n_rolling_out: the number of policy gradient generated
    output:
      policy_gradient_record_np: numpy.array, the array of gradient generated, scaled by 1/time_horizon
    """
    # this probability ratio in the same under in-policy setting, i.e. all 1, deterministic
    policy_prob_ratio = get_policy_prob_ratio(policy_1 = policy_1,
                                              policy_0 = policy_0,
                                              state_space = 6)

    # the grad of log_pi w.r.t. theta, deterministic
    grad_log = [[get_log_pi_gradient(policy = policy_1, state = i, action=j) for j in range(2)] for i in range(6)]

    #the trajectory part, random
    policy_gradient_record = []
    # for each loop
    for time_step_ in tqdm(range(n_rolling_out), desc = "causality: ", leave=True):
        # rollout policy 0
        triple_pi_0, reward_list = roll_out_procedure(policy = policy_0,
                                                      environment = env,
                                                      record_from = rolling_out_start_recording,
                                                      rolling_length = rolling_out_step,
                                                      ihp = False)

        gradient = compute_policy_gradient_causality(triple_pi_0,
                                                     policy_prob_ratio,
                                                     grad_log,
                                                     env)
        # record the gradient generated
        policy_gradient_record.append(gradient)

    # save the gradient as numpy array from
    policy_gradient_record_np = np.vstack([i.cpu().numpy() for i in policy_gradient_record])

    # divide by time step
    return policy_gradient_record_np/(rolling_out_step - rolling_out_start_recording)

def variance_estimate_baseline(policy_0, policy_1, env, rolling_out_start_recording, rolling_out_step, n_rolling_out):
    """
    the full wrapper for evaluating the baseline ver. policy gradient
    input:
      policy_0, policy_1: torch.nn.module, the policy
      env: environment, the environment used
      rolling_out_start_recording: int, the step start recording
      rolling_out_step: int the step recording stops
      n_rolling_out: the number of policy gradient generated
    output:
      policy_gradient_record_np: numpy.array, the array of gradient generated, scaled by 1/time_horizon
    """
    # this two part in the same under in-policy setting, i.e. the second parenthesis are all 1, deterministic
    policy_prob_ratio = get_policy_prob_ratio(policy_1 = policy_1,
                                              policy_0 = policy_0,
                                              state_space = 6)

    # the grad of log_pi w.r.t. theta, deterministic
    grad_log = [[get_log_pi_gradient(policy = policy_1, state = i, action=j) for j in range(2)] for i in range(6)]

    #the trajectory part, random
    reward_mean_record = []
    triple_pi_0_list = []
    # for each loop, roll out a trajectory in policy_0
    for time_step_ in tqdm(range(n_rolling_out), desc = "Baseline: Rolling out...", leave=True):
        # rollout policy 0
        triple_pi_0, reward_list = roll_out_procedure(policy = policy_0,
                                                      environment = env,
                                                      record_from = rolling_out_start_recording,
                                                      rolling_length = rolling_out_step,
                                                      ihp = False)
        #save the (s,a,s') triple
        triple_pi_0_list.append(triple_pi_0)

        #save the reward of this traj
        reward_mean_record.append(np.sum(reward_list))

    #calculate the actual baseline
    baseline_b = np.mean(reward_mean_record)

    #calculate the grad
    policy_gradient_record = []
    for time_step_ in tqdm(range(n_rolling_out), desc = "Baseline: Calculating gradient...", leave=True):
        gradient = compute_policy_gradient_baseline(triple_pi_0_list[time_step_],
                                                    policy_prob_ratio,
                                                    grad_log,
                                                    env,
                                                    baseline_b)
        policy_gradient_record.append(gradient)

    # save the gradient as numpy array from
    policy_gradient_record_np = np.vstack([i.cpu().numpy() for i in policy_gradient_record])

    # divide by time step
    return policy_gradient_record_np/(rolling_out_step - rolling_out_start_recording)

def variance_estimate_ihp1(policy_0, policy_1, env, rolling_out_start_recording, rolling_out_step, n_rolling_out):
    """
    the full wrapper for evaluating the ihp1 ver. policy gradient
    input:
      policy_0, policy_1: torch.nn.module, the policy
      env: environment, the environment used
      rolling_out_start_recording: int, the step start recording
      rolling_out_step: int the step recording stops
      n_rolling_out: the number of policy gradient generated
    output:
      policy_gradient_record_np: numpy.array, the array of gradient generated
    """
    # this two part in the same under in-policy setting, i.e. the second parenthesis are all 1, deterministic
    policy_prob_ratio = get_policy_prob_ratio(policy_1 = policy_1,
                                              policy_0 = policy_0,
                                              state_space = 6)

    # the grad of log_pi w.r.t. theta, deterministic
    grad_log = [[get_log_pi_gradient(policy = policy_1, state = i, action=j).detach().cuda() for j in range(2)] for i in range(6)]

    # all the random part
    policy_gradient_record = []

    # for each loop, roll out a trajectory in policy_0
    for time_step_ in tqdm(range(n_rolling_out), desc = "ihp1", leave = True):
        # rollout policy 0
        _ , _ , triple_pi_count_0 , state_f_0 , state_list_0 = roll_out_procedure(policy = policy_0,
                                                                                  environment = env,
                                                                                  record_from = rolling_out_start_recording,
                                                                                  rolling_length = rolling_out_step,
                                                                                  ihp = True)

        #rollout policy 1
        #_ , _ , triple_pi_count_1, state_f_1, state_list_1 = roll_out_procedure(policy = policy_1,
        #                                                                        environment = env,
        #                                                                        record_from = rolling_out_start_recording,
        #                                                                        rolling_length = rolling_out_step,
        #                                                                        ihp = True)

        # in-policy setting
        triple_pi_count_1, state_f_1, state_list_1 = triple_pi_count_0, state_f_0 , state_list_0

        # approximate w for policy 1
        w_approxi = direct_solve_least_square(policy_1,
                                              triple_pi_count_1,
                                              grad_log,
                                              state_list_1
                                              )

        # combine every part together, get the gradient
        gradient = compute_policy_gradient_ihp1(triple_pi_count_0,
                                                policy_prob_ratio,
                                                state_f_0,
                                                state_f_1,
                                                grad_log,
                                                w_approxi,
                                                env)

        #gradient = F.normalize(gradient, p = 2, dim = 0)
        policy_gradient_record.append(gradient)

    # I used the trace of cov matrix here as the estimator
    policy_gradient_record_np = np.vstack([i.cpu().numpy() for i in policy_gradient_record])
    return policy_gradient_record_np

def variance_estimate_aggregate(policy_0, policy_1, env, rolling_out_start_recording, rolling_out_step, n_rolling_out):
    """
    the full wrapper for evaluating all the gradient with same rollout history (Common Random number)
    input:
      policy_0, policy_1: torch.nn.module, the policy
      env: environment, the environment used
      rolling_out_start_recording: int, the step start recording
      rolling_out_step: int the step recording stops
      n_rolling_out: the number of policy gradient generated

    output:
      policy_gradient_record_benchmark: numpy.array, the array of gradient generated with benchmark method,
                                        axis 0 as the order(n_rolling_out), scaled by the time horizon

      policy_gradient_record_causality, numpy.array, the array of gradient generated with causality method,
                                        axis 0 as the order(n_rolling_out), scaled by the time horizon

      policy_gradient_record_baseline, numpy.array, the array of gradient generated with baseline method,
                                        axis 0 as the order(n_rolling_out), scaled by the time horizon

      policy_gradient_record_ihp, numpy.array, the array of gradient generated with ihp1 method,
                                        axis 0 as the order(n_rolling_out), scaled by the time horizon
    """
    # this probability ratio in the same under in-policy setting, i.e. all 1, deterministic
    policy_prob_ratio = get_policy_prob_ratio(policy_1 = policy_1,
                                              policy_0 = policy_0,
                                              state_space = 6)

    # the grad of log_pi w.r.t. theta, deterministic
    grad_log = [[get_log_pi_gradient(policy = policy_1, state = i, action=j) for j in range(2)] for i in range(6)]

    #the trajectory part, random
    policy_gradient_record_benchmark = []
    policy_gradient_record_causality = []
    policy_gradient_record_ihp = []
    reward_mean_record = []
    triple_pi_0_list = []
    # for each loop
    for time_step_ in tqdm(range(n_rolling_out), desc = "rolling out", leave=True):
        # rollout policy 0
        triple_pi_0, reward_list, triple_pi_count_0, state_f_0, state_list_0 = roll_out_procedure(policy = policy_0,
                                                                                                  environment = env,
                                                                                                  record_from = rolling_out_start_recording,
                                                                                                  rolling_length = rolling_out_step,
                                                                                                  ihp = True)
        # baseline part
        # save the (s,a,s') triple
        triple_pi_0_list.append(triple_pi_0)
        #save the reward of this traj
        reward_mean_record.append(np.sum(reward_list))

        #benchmark part
        gradient_benchmark = compute_policy_gradient_benchmark(triple_pi_0,
                                                               policy_prob_ratio,
                                                               grad_log,
                                                               env)
        policy_gradient_record_benchmark.append(gradient_benchmark)

        #causality part
        gradient_causality = compute_policy_gradient_causality(triple_pi_0,
                                                     policy_prob_ratio,
                                                     grad_log,
                                                     env)
        # record the gradient generated
        policy_gradient_record_causality.append(gradient_causality)

        # IHP part, using in-policy setting
        triple_pi_count_1, state_f_1, state_list_1 = triple_pi_count_0, state_f_0 , state_list_0

        # approximate w for policy 1
        w_approxi = direct_solve_least_square(policy_1,
                                              triple_pi_count_1,
                                              grad_log,
                                              state_list_1
                                              )

        # combine every part together, get the gradient
        gradient_ihp = compute_policy_gradient_ihp1(triple_pi_count_0,
                                                policy_prob_ratio,
                                                state_f_0,
                                                state_f_1,
                                                grad_log,
                                                w_approxi,
                                                env)

        #gradient = F.normalize(gradient, p = 2, dim = 0)
        policy_gradient_record_ihp.append(gradient_ihp)


    #calculate the actual baseline
    baseline_b = np.mean(reward_mean_record)

    #calculate the grad
    policy_gradient_record_baseline = []
    for time_step_ in tqdm(range(n_rolling_out), desc = "Calculating gradient...", leave=True):

        gradient_baseline = compute_policy_gradient_baseline(triple_pi_0_list[time_step_],
                                                        policy_prob_ratio,
                                                        grad_log,
                                                        env,
                                                        baseline_b)
        policy_gradient_record_baseline.append(gradient_baseline)


    # save the gradient as numpy array from, divided by the time step
    policy_gradient_record_benchmark = np.vstack([i.cpu().numpy() for i in policy_gradient_record_benchmark])/(rolling_out_step - rolling_out_start_recording)
    policy_gradient_record_causality = np.vstack([i.cpu().numpy() for i in policy_gradient_record_causality])/(rolling_out_step - rolling_out_start_recording)
    policy_gradient_record_baseline = np.vstack([i.cpu().numpy() for i in policy_gradient_record_baseline])/(rolling_out_step - rolling_out_start_recording)

    policy_gradient_record_ihp = np.vstack([i.cpu().numpy() for i in policy_gradient_record_ihp])

    return policy_gradient_record_benchmark, policy_gradient_record_causality,policy_gradient_record_baseline, policy_gradient_record_ihp
