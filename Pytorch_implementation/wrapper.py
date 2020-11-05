import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
from .optimization_grad import gradient_estimate_benchmark, gradient_estimate_causality, gradient_estimate_baseline, gradient_estimate_ihp1, gradient_estimate_aggregate
from .variance_grad import variance_estimate_aggregate
from .environment import riverswim
from .policies import Policy_NN, Policy_Param
from .analysis import analysis
from .true_value import get_theoretical_gradient
from .rollout import roll_out_evaluate_average
from .utils import update_parameter, get_optimization_result
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm 

def in_policy_optimization(rolling_out_start_recording: int = 1000,
                           rolling_out_step: int = 15000,
                           n_rolling_out: int = 5,
                           optimization_step: int = 50,
                           lr: float = 0.05,
                           normalized_grad_ascent: bool = True,
                           discount_constant: float = 1.0,
                           model: str = "param",
                           tensorboard: bool = True):
    """
    have 4 methods shares the same initialization, in policy
    """
    assert rolling_out_start_recording <= rolling_out_step
    assert 0<=discount_constant<=1
    assert model in ["param", "NN"]

    if model == "param":
        original_policy = Policy_Param()
    else:
        original_policy = Policy_NN()
    environment = riverswim()

    policy_benchmark = copy.deepcopy(original_policy).cuda()
    policy_causality = copy.deepcopy(policy_benchmark).cuda()
    policy_baseline  = copy.deepcopy(policy_benchmark).cuda()
    policy_ihp1      = copy.deepcopy(policy_benchmark).cuda()

    lr_benchmark = lr
    lr_causality = lr
    lr_baseline = lr
    lr_ihp1 = lr

    reward_list_in_policy = []
    if tensorboard:
        tb = SummaryWriter()
    for time_step in tqdm(range(optimization_step), desc = "Optimization step", leave = True):
        policy_grad_benchmark, reward_benchmark = gradient_estimate_benchmark(policy_0 = policy_benchmark,
                                                                      policy_1 = policy_benchmark,
                                                                      env = environment,
                                                                      rolling_out_start_recording = rolling_out_start_recording,
                                                                      rolling_out_step = rolling_out_step,
                                                                      n_rolling_out = n_rolling_out,
                                                                      discount_constant = discount_constant,
                                                                      in_policy = True,
                                                                      model = model)

        benchmark_grad = update_parameter(policy = policy_benchmark,
                                  policy_gradient = policy_grad_benchmark,
                                  learning_rate= lr_benchmark,
                                  normalize = normalized_grad_ascent)

        policy_grad_causality, reward_causality = gradient_estimate_causality(policy_0 = policy_causality,
                                                                      policy_1 = policy_causality,
                                                                      env = environment,
                                                                      rolling_out_start_recording = rolling_out_start_recording,
                                                                      rolling_out_step = rolling_out_step,
                                                                      n_rolling_out = n_rolling_out,
                                                                      discount_constant = discount_constant,
                                                                      in_policy = True,
                                                                      model = model)
        causality_grad= update_parameter(policy = policy_causality,
                policy_gradient = policy_grad_causality,
                learning_rate= lr_causality,
                normalize = normalized_grad_ascent)

        policy_grad_baseline, reward_baseline = gradient_estimate_baseline(policy_0 = policy_baseline,
                                                                          policy_1 = policy_baseline,
                                                                          env = environment,
                                                                          rolling_out_start_recording = rolling_out_start_recording,
                                                                          rolling_out_step = rolling_out_step,
                                                                          n_rolling_out = n_rolling_out,
                                                                          discount_constant = discount_constant,
                                                                          in_policy = True,
                                                                          model = model)
        baseline_grad = update_parameter(policy = policy_baseline,
                policy_gradient = policy_grad_baseline,
                learning_rate= lr_baseline,
                normalize = normalized_grad_ascent)

        policy_grad_ihp1, reward_ihp1 = gradient_estimate_ihp1(policy_0 = policy_ihp1,
                                                      policy_1 = policy_ihp1,
                                                      env = environment,
                                                      rolling_out_start_recording = rolling_out_start_recording,
                                                      rolling_out_step = rolling_out_step,
                                                      n_rolling_out = n_rolling_out,
                                                      discount_constant = discount_constant,
                                                      in_policy = True,
                                                      model = model)

        ihp1_grad = update_parameter(policy = policy_ihp1,
                                policy_gradient = policy_grad_ihp1,
                                learning_rate= lr_ihp1,
                                normalize = normalized_grad_ascent)

        tb.add_scalars(main_tag=f"Average Reward Per Step, In-Policy, lr = {lr}",
                        tag_scalar_dict={"Benchmark": reward_benchmark,
                                        "Causality": reward_causality,
                                        "Baseline": reward_baseline,
                                        "ihp1": reward_ihp1},
                        global_step = time_step)
        reward_list_in_policy.append([reward_benchmark, reward_causality, reward_baseline, reward_ihp1])
        gc.collect()
    if tensorboard:
        tb.close()
    return get_optimization_result(reward_list_in_policy, in_policy = True, lr = lr, model = model)

def off_policy_optimization(rolling_out_start_recording: int = 1000,
                           rolling_out_step: int = 15000,
                           n_rolling_out: int = 5,
                           optimization_step: int = 50,
                           lr: float = 0.05,
                           normalized_grad_ascent: bool = True,
                           discount_constant: float = 1.0,
                           model: str = "param",
                           tensorboard: bool = True):

    lr_benchmark = lr
    lr_causality = lr
    lr_baseline = lr
    lr_ihp1 = lr

    assert rolling_out_start_recording <= rolling_out_step
    assert 0<=discount_constant<=1
    assert model in ["param", "NN"]

    if model == "param":
        policy_0 = Policy_Param()
    else:
        policy_0 = Policy_NN()
    environment = riverswim()

    policy_benchmark = copy.deepcopy(policy_0).cuda()
    policy_causality = copy.deepcopy(policy_0).cuda()
    policy_baseline  = copy.deepcopy(policy_0).cuda()
    policy_ihp1      = copy.deepcopy(policy_0).cuda()
    policy_0 = policy_0.cuda()

    reward_list_off_policy = []

    #visualization
    if tensorboard:
        tb = SummaryWriter()

    for time_step in tqdm(range(optimization_step), desc = "Optimization step", leave = True):


        reward_benchmark = roll_out_evaluate_average(policy = policy_benchmark,
                                                     environment = environment,
                                                     start = rolling_out_start_recording,
                                                     rolling_length = rolling_out_step,
                                                     discount_constant = discount_constant,
                                                     mode = model)
        reward_causality = roll_out_evaluate_average(policy = policy_causality,
                                                     environment = environment,
                                                     start = rolling_out_start_recording,
                                                     rolling_length = rolling_out_step,
                                                     discount_constant = discount_constant,
                                                     mode = model)
        reward_baseline = roll_out_evaluate_average(policy = policy_baseline,
                                                     environment = environment,
                                                     start = rolling_out_start_recording,
                                                     rolling_length = rolling_out_step,
                                                     discount_constant = discount_constant,
                                                     mode = model)
        reward_ihp1 = roll_out_evaluate_average(policy = policy_ihp1,
                                                environment = environment,
                                                start = rolling_out_start_recording,
                                                rolling_length = rolling_out_step,
                                                discount_constant = discount_constant,
                                                mode = model)

        tb.add_scalars(main_tag=f"Average reward per step: off policy, lr = {lr}",
                                tag_scalar_dict={"Benchmark": reward_benchmark,
                                                 "Causality": reward_causality,
                                                 "Baseline": reward_baseline,
                                                 "ihp1": reward_ihp1},
                                global_step = time_step)


        policy_grad_benchmark, policy_grad_causality,policy_grad_baseline, policy_grad_ihp1 = gradient_estimate_aggregate(policy_0 = policy_0,
                                                                                                                          policy_1_benchmark = policy_benchmark,
                                                                                                                          policy_1_causality = policy_causality,
                                                                                                                          policy_1_baseline = policy_baseline,
                                                                                                                          policy_1_ihp1 = policy_ihp1,
                                                                                                                          env = environment,
                                                                                                                          rolling_out_start_recording = rolling_out_start_recording,
                                                                                                                          rolling_out_step = rolling_out_step,
                                                                                                                          n_rolling_out = n_rolling_out,
                                                                                                                          discount_constant = discount_constant,
                                                                                                                          in_policy = False,
                                                                                                                          model = model)
        update_parameter(policy = policy_benchmark,
                         policy_gradient = policy_grad_benchmark,
                         learning_rate= lr_benchmark,
                         normalize = normalized_grad_ascent)

        update_parameter(policy = policy_causality,
                         policy_gradient = policy_grad_causality,
                         learning_rate= lr_causality,
                         normalize = normalized_grad_ascent)

        update_parameter(policy = policy_baseline,
                         policy_gradient = policy_grad_baseline,
                         learning_rate= lr_baseline,
                         normalize = normalized_grad_ascent)

        update_parameter(policy = policy_ihp1,
                         policy_gradient = policy_grad_ihp1,
                         learning_rate= lr_ihp1,
                         normalize = normalized_grad_ascent)


        reward_list_off_policy.append([reward_benchmark, reward_causality, reward_baseline, reward_ihp1])
        gc.collect()
    if tensorboard:
        tb.close()
    return get_optimization_result(reward_list_off_policy, in_policy = False, lr = lr, model = model)

def variance_illustration(rolling_out_start_recording: int = 1000,
                          rolling_out_step: int = 15000,
                          n_rolling_out: int = 5,
                          discount_constant: float = 1.0,
                          model: str = "param",
                          tensorboard: bool = True):

    assert model in ["param", "NN"]

    if model == "param":
        policy_0 = Policy_Param().cuda()
    else:
        policy_0 = Policy_NN().cuda()
    environment = riverswim()

    policy_1 = copy.deepcopy(policy_0).cuda()

    benchmark_record, causality_record, baseline_record, ihp1_record = variance_estimate_aggregate(policy_0 = policy_0,
                                                                                                   policy_1 = policy_1,
                                                                                                   env = environment,
                                                                                                   rolling_out_start_recording = rolling_out_start_recording,
                                                                                                   rolling_out_step = rolling_out_step,
                                                                                                   n_rolling_out = n_rolling_out,
                                                                                                   discount_constant = discount_constant,
                                                                                                   in_policy = True,
                                                                                                   ls_tb_illustration = False,
                                                                                                   model = model
                                                                                                  )
    if model == "param":
        theoretical_value = get_theoretical_gradient(policy = policy_0, environment=environment)[0].cpu().numpy()
        return analysis(benchmark_record = benchmark_record,
                        causality_record = causality_record,
                        baseline_record = baseline_record,
                        ihp1_record = ihp1_record,
                        true_value = theoretical_value,
                        mode = model)
    else:
        return analysis(benchmark_record = benchmark_record,
                        causality_record = causality_record,
                        baseline_record = baseline_record,
                        ihp1_record = ihp1_record,
                        true_value = None,
                        mode = model)
