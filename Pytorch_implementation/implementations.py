from .utils import *
import torch
import numpy as np

def compute_policy_gradient_benchmark(triple_pi_0, policy_prob_ratio, grad_log, reward_list):
    """
    combine every part together, get the gradient, not considering anything from causality or baseline
    input:
      triple_pi_0, triple_pi_0_sas_f: the name and distribution of (s,a,s') on policy 0
      policy_prob_ratio: pi_theta(a|s)/ pi_0(a|s)
      grad_log: grad_log_pi_theta(a|s)
      env: environment, for r(s,a,s')
    return:
      final_gradient: torch.FloatTensor in 1-D, the policy gradient
    """
    with torch.no_grad():
        # list the policy prob ratio
        policy_prob_ratio_list = np.fromiter((policy_prob_ratio[triple_pi_0[i][0]][triple_pi_0[i][1]] for i in range(triple_pi_0.shape[0])), float)

        # prod it
        policy_prob_prod = np.product(policy_prob_ratio_list)

        # compute the trajectory's reward
        reward = np.sum(reward_list)

        # along the trajectory, apply each step
        policy_gradient = torch.sum(torch.stack([(grad_log[triple_pi_0[i][0]][triple_pi_0[i][1]]) for i in range(triple_pi_0.shape[0])]),dim = 0) * policy_prob_prod * reward

        return policy_gradient
def compute_policy_gradient_causality(triple_pi_0, policy_prob_ratio, grad_log, reward_list):
    """
    combine every part together, get the gradient in causality ver. formula
    input:
      triple_pi_0: the (s,a,s') history on policy 0
      policy_prob_ratio: pi_theta(a|s)/pi_0(a|s)
      grad_log: grad_log_pi_theta(a|s)
      reward_list: r(s,a,s')
    return:
      final_gradient: torch.FloatTensor in 1-D, the policy gradient
    """
    with torch.no_grad():
        policy_prob_ratio_list = np.fromiter((policy_prob_ratio[triple_pi_0[i][0]][triple_pi_0[i][1]] for i in range(triple_pi_0.shape[0])), float)

        # cumprod it
        policy_prob_cumprod = np.cumprod(policy_prob_ratio_list)

        # cumsum it, then reverse the order
        reward_reverse_cumsum = np.cumsum(reward_list[::-1])[::-1]

        # along the trajectory, apply each step
        policy_gradient = torch.sum(
                          torch.stack(
                          [(grad_log[triple_pi_0[i][0]][triple_pi_0[i][1]]*\
                            policy_prob_cumprod[i]*\
                            reward_reverse_cumsum[i])
                          for i in range(triple_pi_0.shape[0])]
                          ),
                          dim = 0)

        return policy_gradient

def compute_policy_gradient_baseline(triple_pi_0, policy_prob_ratio, grad_log, reward_sum, baseline_b):
    """
    combine every part together, get the gradient, considering baseline
    input:
      triple_pi_0, triple_pi_0_sas_f: the name and distribution of (s,a,s') on policy 0
      policy_prob_0: pi_0(a|s)
      policy_prob_1: pi_theta(a|s)
      grad_log: grad_log_pi_theta(a|s)
      env: environment, for r(s,a,s')
      baseline_b: the baseline specified, which is the average return of each trajectory

    return:
      final_gradient: torch.FloatTensor in 1-D, the policy gradient
    """
    with torch.no_grad():
        # list the policy prob ratio
        policy_prob_ratio_list = np.fromiter((policy_prob_ratio[triple_pi_0[i][0]][triple_pi_0[i][1]] for i in range(triple_pi_0.shape[0])), float)

        # calculate the product
        policy_prob_prod = np.product(policy_prob_ratio_list)

        # along the trajectory, apply each step
        policy_gradient = torch.sum(
                          torch.stack(
                          [(grad_log[triple_pi_0[i][0]][triple_pi_0[i][1]]) for i in range(triple_pi_0.shape[0])]
                          ),
                          dim = 0)*\
                          policy_prob_prod * (reward_sum-baseline_b)

        return policy_gradient

def direct_solve_least_square(policy, triple_pi_count, grad_log, state_list, tb_illustration):
    """
    use gradient descent to directly solve the w
    input:
      policy: nn.module, the policy specified
      triple_pi_count: pair of np.array, the (s,a,s') history of this policy
      grad_log: list of torch.FloatTensor, the gradient of log pi
      state_list: list of float, the rollout list of state
      tb_illustration: bool, decide if the output is send to tensorboard
    output:
      w_approxi: list of torch.FloatTensor, the approxmate w
    """
    w_approxi = [torch.randn(count_parameters(policy), device= "cuda", requires_grad = True) for i in range(6)]
    lr = 1e-4
    if tb_illustration:
        sum_record = []
    for time_step in range(50):
        if time_step % 25 == 0:
          lr /= 5
        sum = torch.tensor(0., device= "cuda")
        for i in range(triple_pi_count[1].shape[0]):
            s_previous = triple_pi_count[0][i][0]
            action = triple_pi_count[0][i][1]
            s_later = triple_pi_count[0][i][2]
            n = triple_pi_count[1][i]
            sum = sum.add(n * torch.norm(w_approxi[s_later] - w_approxi[s_previous] - grad_log[s_previous][action])**2)
        sum.backward()
        if tb_illustration:
           sum_record.append(sum.cpu().item())
        w_approxi = [(j - lr * j.grad).detach().requires_grad_() for j in w_approxi]
    if tb_illustration:
        plt.plot(sum_record)
        tensorboard.add_figure("least square", plt.gcf())
    with torch.no_grad():
        #w_approxi here is correct only on difference between each other, but its value is not fixed
        w_approxi_r = [(i - w_approxi[0]) for i in w_approxi]

        # get the distribution from trajectory, sort is for keeping the order
        state_, count_ = np.unique(np.sort(state_list), return_counts= True)
        state_count = np.zeros(6)
        for state, count in zip(state_,count_):
            state_count[state] = count
        state_distribution = state_count / np.sum(state_count)

        w_approxi_0 = - torch.sum(torch.stack([w_approxi_r[i] * state_distribution[i] for i in range(6)]), dim = 0)
        w_approxi_result = [(i + w_approxi_0).cuda() for i in w_approxi_r]

        return w_approxi_result

def compute_policy_gradient_ihp1(triple_pi_count_0,
                                 policy_prob_ratio,
                                 state_d_0,
                                 state_d_1,
                                 grad_log,
                                 w_approxi,
                                 env):
    """
    combine every part together, get the gradient
    input:
      triple_pi_count_0: the trajectory count record of (s,a,s') on policy 0
      policy_prob_ratio: pi_theta(a|s)/pi_0(a|s)
      state_d_0: d_pi_0(s)
      state_d_1: d_pi_theta(s)
      grad_log: grad_log_pi_theta(a|s)
      w_approxi: the approximation of d_pi_theta(s)
      env: environment, for r(s,a,s')
    return:
      final_gradient: torch.FloatTensor in 1-D, the policy gradient
    """
    with torch.no_grad():
        # the sampled sas name and its normalized frequency
        triple_pi_0_sas_triple = triple_pi_count_0[0]
        triple_pi_0_sas_f = triple_pi_count_0[1] / np.sum(triple_pi_count_0[1])
        d_pi_ratio = np.divide(state_d_1, state_d_0)

        for i in range(triple_pi_0_sas_f.shape[0]):
            state = triple_pi_0_sas_triple[i][0]
            action = triple_pi_0_sas_triple[i][1]
            next_state = triple_pi_0_sas_triple[i][2]

            pi_ratio = policy_prob_ratio[state][action]
            d_ratio = d_pi_ratio[state]
            grad_log_pi = grad_log[state][action]
            grad_log_d = w_approxi[state]

            r = env.get_reward(state, action, next_state)

            expectation_weight = triple_pi_0_sas_f[i]

            if i == 0:
                policy_gradient = pi_ratio * d_ratio * (grad_log_pi + grad_log_d ) * r * expectation_weight
            else:
                policy_gradient = policy_gradient + pi_ratio * d_ratio * (grad_log_pi + grad_log_d ) * r * expectation_weight

        return policy_gradient
