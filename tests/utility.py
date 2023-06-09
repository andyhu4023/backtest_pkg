import numpy as np
from math import sqrt


def cal_std(data):
    if len(data) <= 1:
        return np.nan
    data_mean = sum(data) / len(data)
    data_var = sum((i - data_mean) ** 2 for i in data) / (len(data) - 1)
    return sqrt(data_var)


def cal_mean(data):
    return sum(data) / len(data)
