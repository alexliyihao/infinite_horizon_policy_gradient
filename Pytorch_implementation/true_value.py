import numpy as np
import torch

def transition_probability(previous_state, later_state):
    if previous_state == 0:
       if later_state == 0:
          return 0.7
       else:
          return 0.3
    elif previous_state in [1,2,3,4]:
       if later_state == previous_state - 1:
         return 0.1
       elif later_state == previous_state:
         return 0.3
       else:
         return 0.6
    else:
       if later_state == 5:
         return 0.3
       else:
         return 0.7

def get_theoretical_gradient(policy, environment):
    s_a_s_list = [[0, 0, 0],
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
                [5, 1, 5]]
    Prob = policy.l1.weight.data.clone().requires_grad_()
    transition_matrix = torch.stack([torch.stack([(previous[2] == later[0])*
                                                  ((later[1] == 1) * Prob[later[0]] * transition_probability(later[0], later[2]) +
                                                   (later[1] == 0) * (1 - Prob[later[0]]))
                                      for later in s_a_s_list])
                                      for previous in s_a_s_list])
    A = torch.eye(22, device= "cuda") - transition_matrix.transpose(0,1)
    A[20] = torch.ones(1,22)
    B = torch.zeros(22,1, device= "cuda")
    B[20] = 1
    stationary_distribution_sas, _ = torch.solve(input = B, A = A)
    reward_list= torch.tensor([env.get_reward(i[0], i[1], i[2]) for i in s_a_s_list], device = "cuda")
    expected_reward = torch.dot(stationary_distribution_sas.flatten(),reward_list)
    expected_reward.backward()
    gradient = Prob.grad
    return gradient, stationary_distribution_sas
