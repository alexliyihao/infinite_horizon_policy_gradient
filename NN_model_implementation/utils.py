import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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

def select_action(policy, state):
    """
    Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    input:
      policy: nn.module, the policy specified
      state: int, the numerical encode of the state
    return:
      action: int, the action (0 or 1) selected
    """
    # convert state to one hot
    state = get_one_hot(state, 6)

    # from policy create a probablistic distribution
    action = policy(state)

    # by the probablity obtained, create a categorical distribution
    c = Categorical(action)

    # sample from this distribution
    action = c.sample()

    return action.item()


def rolling_out(policy, environment, record_from, rolling_length):
    """
    given a policy and an environment, roll out the policy and record the trajectory
    input:
      policy: nn.module, the policy specified
      environment: environment, the environment spacified
      record_from: int, the starting point recording
      rolling_length: int, the total roll-out length

    output:
      state_list: list of int, the state record
      action_list: list of int: the action record
      reward_list: list of int: the reward record
    """

    assert record_from <= rolling_length

    #reset this environment, get its original space
    state = environment.reset()

    state_list = []
    action_list = []
    reward_list = []
    with torch.no_grad():

        for time_step in range(rolling_length):
            #record the action selected by the policy
            action = select_action(policy, state)

            if time_step >= record_from:
                state_list.append(state)
                action_list.append(action)

            # record the state and reward from the environment
            state, reward = environment(action)

            if time_step >= record_from:
                reward_list.append(reward)

        # an additional state here, for the last (s,a,s')
        state_list.append(state)

    return state_list, action_list, reward_list

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
    return:
      state_action_state_triple_count: list of np.array, the np.array of (s,a,s') triple and their counts
    """
    state_action_state_triple = np.vstack((np.array([[0, 0, 0],
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
                                                     [5, 1, 5]]),
                                           state_action_state_triple))#for always get a valid count

    # count the (s,a,s'), and convert it into a list
    state_action_state_triple_count = list(np.unique(state_action_state_triple, return_counts= True, axis = 0))

    # minues the additional one, pair cannot operate this step
    state_action_state_triple_count[1] = state_action_state_triple_count[1] - 1

    # check if we have went through all of that
    if np.count_nonzero(state_action_state_triple_count[1]) != 22:
        print("some of the (s,a,s') triple not found, but will work on")
    return state_action_state_triple_count


def get_policy_prob_ratio(policy_1, policy_0, state_space):
    """
    for an existed policy, get the probability ratio pi_1(a|s)/pi_0(a|s) of selecting actions in the whole state space
    input:
      policy_1: nn.module, the tested policy specified
      policy_0: nn.module, the baseline policy specified
      state_space: int, the cardinality of the state space
    return:
      action_prob_ratio: torch.FloatTensor, a vector of probability selecting each action
    """
    with torch.no_grad():
        action_prob_ratio = [torch.true_divide(policy_1(get_one_hot(i, 6)), policy_0(get_one_hot(i, 6))).detach()
                              for i in range(state_space)]
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
    m = Categorical(probs)

    # get the log prob of a specific action
    loss = m.log_prob(torch.tensor(action).cuda())

    # calculate the gradient
    loss.backward()

    # get the gradient in vector:
    grad_log_pi = torch.cat([torch.flatten(grads) for grads in [value.grad for name, value in policy.named_parameters()]]).detach()

    return grad_log_pi.cuda()

def roll_out_evaluate_average(policy, environment, start, step):
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

    state_list, action_list, reward_list = rolling_out(policy=policy,
                                                       environment = environment,
                                                       record_from = start,
                                                       rolling_length = step)
    return np.average(reward_list)

def roll_out_evaluate_sum(policy, environment,step):
    """
    generate the total return over step
    input:
      policy: nn.module, the policy specified
      environment: environment, the environment specified
      step: int, roll out step
    output:
      total_return: float the total return over step
    """

    state_list, action_list, reward_list = rolling_out(policy=policy,
                                                       environment = environment,
                                                       record_from = 0,
                                                       rolling_length = step)
    return np.sum(reward_list)

def roll_out_procedure(policy, environment, record_from, rolling_length, ihp = False):
    """
    the wrapper of rolling out, taking care of not existed (s,a,s') triple

    input:
      policy: nn.module, the policy specified
      environment: environment, the environment specified
      record_from: int, the starting point recording
      rolling_length: int, the total roll-out length
      ihp: bool, the indicator of considering ihp method output in the roll out
    return:
      triple_pi: pair of np.array, the np.array of (s,a,s') triple and their counts
      policy_prob: list of torch.FloatTensor, list of vectors of probability selecting each action
    """
    # roll out the policy in the environment
    state_list, action_list, reward_list = rolling_out(policy=policy,
                                                        environment = environment,
                                                        record_from = record_from,
                                                        rolling_length = rolling_length)

    # convert the state_list and action list into (s,a,s') triple
    triple_pi_record = get_triple(state_list, action_list)

    #for ihp1
    if ihp:
        # get the count of (s, a, s')
        triple_pi_count= get_triple_count(triple_pi_record)

        #get the stationary distribution of s
        state_f = np.unique(np.sort(state_list), return_counts= True)[1]
        state_f = state_f/np.sum(state_f)

        return triple_pi_record, reward_list, triple_pi_count, state_f, state_list

    return triple_pi_record, reward_list

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

class environment():
    def __init__(self):
        self.current = None
        self.state_space = []
        self.action_space = []

    def env_return(self, action):
        pass

    def __call__(self, action):
        current_state, reward = self.env_return(action)
        return current_state, reward

    def get_current(self):
        return self.current

    def set_current(self, state):
        self.current = state

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

class riverswim(environment):
    def __init__(self):
        self.state_space = np.array([0,1,2,3,4,5])
        self.action_space = np.array([0,1])
        self.current = np.random.choice([0,1],p = [0.5,0.5])

    def reset(self):
        self.current = np.random.choice([0,1],p = [0.5,0.5])
        return self.current

    def env_return(self, action):
        assert action in self.action_space
        assert self.current in self.state_space

        if self.current == 0:
            if action == 0:
                self.current = 0
                reward = 0.0005
            if action == 1:
                self.current += np.random.choice([0,1], p=[0.7, 0.3])
                reward = 0
        elif self.current in [1,2,3,4]:
            if action == 0:
                self.current -= 1
                reward = 0
            if action == 1:
                self.current += np.random.choice([-1,0,1], p=[0.1, 0.3, 0.6])#[0.1, 0.6, 0.3])
                reward = 0
        elif self.current == 5:
            if action == 0:
                self.current -= 1
                reward = 0
            if action == 1:
                self.current += np.random.choice([-1,0], p = [0.7, 0.3])
                if self.current == 5:
                    reward = 1
                else:
                    reward = 0
        return self.current, reward

    def get_reward(self, state, action, next_state):
        if state == 0:
            if action == 0:
                reward = 0.0005
            if action == 1:
                reward = 0
        elif state in [1,2,3,4]:
              reward = 0
        elif state == 5:
            if action == 1 and next_state == 5:
                reward = 1
            else:
                reward = 0
        return reward

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 6#env.get_state_space().shape[0]
        self.action_space = 2#env.get_action_space().shape[0]

        self.l1 = nn.Linear(self.state_space, 2)#have it smaller
        self.l2 = nn.Linear(2, self.action_space)

    def forward(self, x):
        x = self.l1(x)
        x = nn.ReLU()(x)
        x = self.l2(x)
        x = nn.Softmax(dim=-1)(x)
        return x

    def num_param(self):
        return count_parameters(self)
