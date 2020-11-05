from utils import *
import torch
from torch.distributions import Categorical, Bernoulli
import numpy as np

def select_action(policy, state: int = 0, mode = "param"):
    """
    Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    input:
      policy: nn.module, the policy specified
      state: int, the numerical encode of the state
    return:
      action: int, the action (0 or 1) selected
    """
    assert mode in ["param", "NN"]
    with torch.no_grad():

        # convert state to one hot
        state = get_one_hot(state, 6)

        # from policy create a probablistic distribution
        action_prob = policy(state)
        if mode == "param":
            # by the probablity obtained, create a categorical distribution
            action_prob = torch.clamp(action_prob, min = 0.0, max = 1.0)
            c = Bernoulli(action_prob)
        else:
            # by the probablity obtained, create a categorical distribution
            c = Categorical(action)
        # sample from this distribution
        action = c.sample()
        return action.item()

def rolling_out(policy,
                environment,
                record_from: int = 1000,
                rolling_length: int = 15000,
                discount_constant: float = 1.0,
                mode = "param"):
    """
    given a policy and an environment, roll out the policy and record the trajectory
    input:
      policy: nn.module, the policy specified
      environment: environment, the environment spacified
      record_from: int, the starting point recording
      rolling_length: int, the total roll-out length
      discount_constant: float in [0,1], the discount constant specified

    output:
      state_list: np.array of int: the state record
      action_list: np.array of int: the action record
      reward_list: np.array of int: the reward record, discounted if discount_constant < 1
    """

    assert record_from <= rolling_length
    assert 0 <= discount_constant <= 1
    assert mode in ["param", "NN"]
    #reset this environment, get its original space
    state = environment.reset()

    state_list = []
    action_list = []
    reward_list = []
    with torch.no_grad():

        for time_step in range(rolling_length):
            #record the action selected by the policy
            action = select_action(policy, state, mode = mode)

            if time_step >= record_from:
                state_list.append(state)
                action_list.append(action)

            # record the state and reward from the environment
            state, reward = environment(action)

            if time_step >= record_from:
                reward_list.append(reward)

        # an additional state here, for the last (s,a,s')
        state_list.append(state)

        # discount the reward, if needed
        if discount_constant == 1.0:
          reward_list = np.array(reward_list)
        else:
          reward_list = np.fromiter((reward_list[i]*discount_constant**(len(reward_list)-1-i) for i in range(len(reward_list))), float)

    return np.array(state_list, dtype=int), np.array(action_list, dtype=int), reward_list

def get_triple(state_list, action_list):
    """
    convert state record and action record into (s,a,s') triples
    input:
      state_list: list of int, the state record
      action_list: list of int: the action record
    return:
      state_action_state_triple: np.array, the np.array of (s,a,s') record
    """
    state_action_state_triple = np.array([[state_list[i], action_list[i], state_list[i+1]] for i in range(len(action_list))])

    return state_action_state_triple

def get_triple_count(state_action_state_triple):
    """
    convert (s,a,s') triples into (s,a,s') counts,
    works for the infinite horizon paper 1.
    input:
      state_action_state_triple: np.array, the np.array of (s,a,s') record
      discount_constant: the discount_constant used
      mode: the mode
    return:
      state_action_state_triple_count: list of np.array, the np.array of (s,a,s') triple and their counts
    """

    sas_list  = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 1, 1],
                          [1, 0, 0],
                          [1, 1, 0],
                          [1, 1, 1],
                          [1, 1, 2],
                          [2, 0, 1],
                          [2, 1, 1],
                          [2, 1, 2],
                          [2, 1, 3],
                          [3, 0, 2],
                          [3, 1, 2],
                          [3, 1, 3],
                          [3, 1, 4],
                          [4, 0, 3],
                          [4, 1, 3],
                          [4, 1, 4],
                          [4, 1, 5],
                          [5, 0, 4],
                          [5, 1, 4],
                          [5, 1, 5]])

    #for always get a valid, "sorted" count
    state_action_state_triple = np.vstack((sas_list, state_action_state_triple))

    # count the (s,a,s'), and convert it into a list
    state_action_state_triple_count = list(np.unique(state_action_state_triple, return_counts= True, axis = 0))

    # minues the additional one, pair cannot operate this step
    state_action_state_triple_count[1] = state_action_state_triple_count[1] - 1

    return state_action_state_triple_count

def single_discounted_state_distribution(item, state_list, discount_constant: float = 1.0):
    #count the backward index in the trajectory:
    backward_index = state_list.shape[0] - 1 - np.where((state_list == item))[0]
    #compute the discounted count:
    weight = np.sum(np.fromiter((discount_constant**i for i in backward_index), float))
    return weight

def get_state_distribution(state_list, discount_constant: float = 1.0):
   """
   The count of state s from state history, discounted by discount_constant if != 1, works for ihp method
   input:
      state_list: np.array, the complete s history
      discount_constant: float, the gamma as discount constant
   """
   assert 0 <= discount_constant <= 1

   if discount_constant == 1.0:
      state_f = np.unique(state_list, return_counts= True)
      state_count = np.zeros(6)
      for state, count in zip(state_f[0],state_f[1]):
          state_count[state] = count
      distribution = state_count / np.sum(state_count)
   else:
      distribution = np.fromiter((single_discounted_state_distribution(item = i,
                                                                        state_list = state_list,
                                                                        discount_constant = discount_constant)
                                  for i in range(6)),
                                  float)
   distribution = distribution/np.sum(distribution)
   return distribution

def roll_out_evaluate_average(policy,
                              environment,
                              start: int = 1000,
                              rolling_length: int = 15000,
                              discount_constant: float = 1.0,
                              mode = "param"):
    """
    generate the average return over step
    input:
      policy: nn.module, the policy specified
      environment: environment, the environment specified
      start: int, start recording step
      step: int, roll out step
    output:
      average_return: float the average return over step
    """

    _, _, reward_list = rolling_out(policy=policy,
                                    environment = environment,
                                    record_from = start,
                                    rolling_length = rolling_length,
                                    discount_constant = discount_constant,
                                    mode = mode)
    return np.average(reward_list)

def roll_out_evaluate_sum(policy,
                          environment,
                          start: int = 1000,
                          rolling_length: int = 15000,
                          discount_constant: float = 1.0,
                          mode = "param"):
    """
    generate the total return over step
    input:
      policy: nn.module, the policy specified
      environment: environment, the environment specified
      step: int, roll out step
    output:
      total_return: float the total return over step
    """

    _, _ , reward_list = rolling_out(policy=policy,
                                     environment = environment,
                                     record_from = start,
                                     rolling_length = rolling_length,
                                     discount_constant = discount_constant,
                                     mode = mode)
    return np.sum(reward_list)

def roll_out_procedure(policy,
                       environment,
                       record_from: int = 1000,
                       rolling_length: int = 15000,
                       discount_constant: float = 1,
                       mode: str = "baseline",
                       model: str = "param"):
    """
    the wrapper of rolling out, taking care of not existed (s,a,s') triple

    input:
      policy: nn.module, the policy specified
      environment: environment, the environment specified
      record_from: int, the starting point recording
      rolling_length: int, the total roll-out length
      discount_constant: float in [0,1], the gamma in markov procedure
      mode: string, the output mode, in "baseline" and "ihp"
    return:
      under "baseline" mode:
        triple_pi: pair of np.array, the np.array of (s,a,s') triple and their counts
        reward_list: the stepwise reward record, discounted if discount constant != 1
      under "ihp" mode:
        triple_pi: pair of np.array, the np.array of (s,a,s') triple and their counts
        reward_list: the stepwise reward record, discounted if discount constant != 1
        triple_pi_count: the count of each (s,a,s') triple, discounted if discount constant != 1
    """
    assert mode in ["baseline", "ihp"]
    assert 0 <= discount_constant <= 1
    # roll out the policy in the environment
    state_list, action_list, reward_list = rolling_out(policy=policy,
                                                       environment = environment,
                                                       record_from = record_from,
                                                       rolling_length = rolling_length,
                                                       discount_constant = discount_constant,
                                                       mode = model
                                                       )

    # convert the state_list and action list into (s,a,s') triple
    triple_pi_record = get_triple(state_list, action_list)

    if mode == 'baseline':
        return triple_pi_record, reward_list
    #for ihp1
    else:
        # get the (discounted, if gamma < 1) count of (s,a,s')
        triple_pi_count= get_triple_count(state_action_state_triple = triple_pi_record)

        state_f = get_state_distribution(state_list=state_list,
                                         discount_constant = discount_constant)

        return triple_pi_record, reward_list, triple_pi_count, state_f, state_list
