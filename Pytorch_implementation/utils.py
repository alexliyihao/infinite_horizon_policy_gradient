import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns

def get_one_hot(i, state_space):
    """
    convert integer i to one-hot encoding
    input:
      i: int, the numerical encoding of state space
      state_space: int, the number of state_space available
    output:
      state: torch.cuda.FloatTensor: the one-hot-encoding of state
    """

    state = np.zeros(state_space)
    np.put(state, i, 1)
    state = torch.from_numpy(state).type(torch.FloatTensor).cuda()
    return state


def get_policy_prob_ratio(policy_1, policy_0, state_space: int, in_policy: bool):
    """
    for an existed policy, get the probability ratio pi_1(a|s)/pi_0(a|s) of selecting actions in the whole state space
    input:
      policy_1: nn.module, the tested policy specified
      policy_0: nn.module, the baseline policy specified
      state_space: int, the cardinality of the state space
      in_policy: bool, the indicator if it's using in policy updating
    return:
      action_prob_ratio: torch.FloatTensor, a vector of probability selecting each action
    """
    with torch.no_grad():
        # when under in-policy setting, the ratio will always be 1, it also helps dealing with the "0/0" problem
        if in_policy:
            action_prob_ratio = [torch.ones(2, device = "cuda") for i in range(state_space)]
        else:
            action_prob_ratio = []
            for i in range(state_space):
                policy_1_action_1_prob = policy_1(get_one_hot(i, 6))
                policy_1_prob = torch.tensor([1-policy_1_action_1_prob, policy_1_action_1_prob], device= "cuda")
                policy_0_action_1_prob = policy_0(get_one_hot(i, 6))
                policy_0_prob = torch.tensor([1-policy_0_action_1_prob, policy_0_action_1_prob], device = "cuda")
                ratio = torch.true_divide(policy_1_prob, policy_0_prob)
                action_prob_ratio.append(ratio)
        return action_prob_ratio

def get_log_pi_gradient(policy, action, state):
    """
    caluculate the gradient of log probability of running a certain action
    on specified state under a specific policy

    input:
      policy: nn.module, the policy specified
      action: int, the action specified
      state: int, the state specified
    return:
      grad_log_pi Torch.cuda.FloatTensor in 1D, the gradient of each variables in the policy
    """

    # clean the grad
    policy.zero_grad()

    #convert state to one-hot
    state = get_one_hot(state, 6)

    # forward pass
    probs = policy(state)

    # get the distribution
    m = Bernoulli(probs)

    # get the log prob of a specific action
    loss = m.log_prob(action)

    # calculate the gradient
    loss.backward()

    # get the gradient in vector:
    grad_log_pi = torch.cat([torch.flatten(grads) for grads in [value.grad for name, value in policy.named_parameters()]]).detach()

    return grad_log_pi.cuda()

def count_parameters(model):
    """
    count the number of parameters in a neural network
    input:
      model: torch.nn.module, the policy specified
    output:
      int, the number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update_parameter(policy, policy_gradient, learning_rate, normalize = False):
    """
    update the parameter, this weird update is because final_gradient is in 1-D vector form
    input:
      policy: nn.module, the policy need update
      policy_gradient: torch.FloatTensor in 1-D, the policy gradient obtained
      learning_rate: float, the learning rate
      normalize: bool, define if we need a normalized policy gradient
    """
    if normalize:
        policy_gradient = F.normalize(policy_gradient, p = 2, dim = 0)
    weight_vector = torch.nn.utils.parameters_to_vector(policy.parameters()).cuda().add(policy_gradient, alpha = learning_rate)
    torch.nn.utils.vector_to_parameters(weight_vector, policy.parameters())

def get_optimization_result(reward_array, in_policy = True, lr = 0.05, model = "param"):
    reward_array_np = np.array(reward_array)
    df = pd.DataFrame(reward_array_np, columns= ["Benchmark", "Causality", "Baseline", "IHP"])
    sns.lineplot(data = df, dashes=False)
    if in_policy:
        plt.title(f"Optimization: In-policy setting, {model} model, lr = {lr}")
    else:
        plt.title(f"Optimization: Off-policy setting, {model} model, lr = {lr}")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward Per Step")
    return plt.gcf()
