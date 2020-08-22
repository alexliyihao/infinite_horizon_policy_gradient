"""
All the analysis method used
"""
import numpy as np
import torch.nn.functional as F
import pandas as pd
pd.set_option('display.float_format', '{:.4e}'.format)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

def individual_analysis(record, dim):
    """
    the variance analysis on an gradient record
    input:
      record: np.array, the ndarray form of gradient, in n_grad * num_param shape
      dim: list of int, length 3, the dimension sampled
    return:
      analysis record: np.array, the 1. trace of covariance matrix
                                     2. mean of norm
                                     3. std. dev of norm
                                     4. trace of covariance matrix of normed gradient
                                     5. three std. devs of dimensions sampled from record
    """
    grad_tr = np.trace(np.cov(record.T))
    norm_list = np.linalg.norm(record, axis = 1)
    grad_normed = [F.normalize(torch.from_numpy(i), p=2, dim = 0) for i in record]
    grad_normed_np = np.array([i.numpy() for i in grad_normed])
    normed_tr = np.trace(np.cov(grad_normed_np.T))
    std_0 = np.std(record[:,dim[0]])
    std_1 = np.std(record[:,dim[1]])
    std_2 = np.std(record[:,dim[2]])
    return np.array([grad_tr, np.mean(norm_list), np.std(norm_list),normed_tr, std_0, std_1, std_2])

def variance_analysis(records, names):
    """
    wrapper of the variance analysis, apply individual analysis to all records inputed
    input:
      records: list of np.array, a list of all the ndarray form of gradient in n_grad * num_param shape
      name: the list of string, each individual method's name
    return:
      df: pandas.DataFrame, for showing all the result in a table
    """
    dim = np.random.choice(records[0].shape[1], 3, replace = False)
    dim = np.sort(dim)
    df = pd.DataFrame(np.vstack([individual_analysis(i, dim) for i in records]),
                      index = names,
                      columns = ["tr(cov)",
                                 "mean(l2-norm)",
                                 "std(l2-norm)",
                                 "tr(cov(normed))",
                                 f"std(dim_{dim[0]})",
                                 f"std(dim_{dim[1]})",
                                 f"std(dim_{dim[2]})"]
                      )
    return df

def normalize(record):
    """
    converting a record of gradient into normed form
    input:
      records: np.array, the ndarray form of gradient in n_grad * num_param shape
    return:
      grad_normed_np: np.array, the ndarray form of normed gradient in n_grad * num_param shape
    """
    grad_normed = [F.normalize(torch.from_numpy(i), p=2, dim = 0) for i in record]
    grad_normed_np = np.array([i.numpy() for i in grad_normed])
    return grad_normed_np

def cosine_similarity(a,b):
    """
    compute the cosine similarity between two vectors
    input:
      a,b: np.array, vectors needed
    output:
      cosine_similarity: float, the cosine similarity
    """
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def direction_analysis(benchmark_record, baseline_record, causality_record, ihp1_record):
    """
    wrapper of the direction analysis, compute the cosine similarity to average of each method.
    input:
      benchmark_record, baseline_record, causality_record, ihp1_record: list of np.array, a list of all the ndarray form of gradient in n_grad * num_param shape
    """
    print("benchmark vs baseline whole traj: ", cosine_similarity(np.mean(benchmark_record, axis = 0), np.mean(baseline_record, axis = 0)))
    print("benchmark vs causality: ", cosine_similarity(np.mean(benchmark_record, axis = 0), np.mean(causality_record, axis = 0)))
    print("baseline whole traj vs causality: ", cosine_similarity(np.mean(baseline_record, axis = 0), np.mean(causality_record, axis = 0)))
    print("ihp1 vs benchmark: ", cosine_similarity(np.mean(benchmark_record, axis = 0), np.mean(ihp1_record, axis = 0)))
    print("ihp1 vs baseline: ", cosine_similarity(np.mean(baseline_record, axis = 0), np.mean(ihp1_record, axis = 0)))
    print("ihp1 vs causality: ", cosine_similarity(np.mean(causality_record, axis = 0), np.mean(ihp1_record, axis = 0)))

def analysis(benchmark_record, causality_record, baseline_record, ihp1_record):
    """
    wrapper of the variance and direction analysis
    input:
      benchmark_record, baseline_record, causality_record, ihp1_record: list of np.array, a list of all the ndarray form of gradient in n_grad * num_param shape
    """
    direction_analysis(benchmark_record, causality_record, baseline_record, ihp1_record)
    return variance_analysis(records = [benchmark_record, causality_record, baseline_record, ihp1_record],
                             names = ["benchmark",  "causality", "baseline", "ihp1"])

def boxplot_analysis(benchmark_record,causality_record,baseline_record,ihp1_record, dim):
    """
    comparing distribution of different gradient estimating methods' result in one dimention
    input:
      benchmark_record, baseline_record, causality_record, ihp1_record: list of np.array, a list of all the ndarray form of gradient in n_grad * num_param shape
    """
    df = pd.DataFrame(np.vstack([benchmark_record[:,dim],causality_record[:,dim],baseline_record[:,dim],ihp1_record[:,dim]]))
    df['method'] = ["benchmark", "causality", "baseline", "ihp1"]
    df = pd.melt(df, id_vars = 'method')
    sns.boxplot(data = df, x = "method", y = "value")
