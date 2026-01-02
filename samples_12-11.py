# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 16:51:39 2025

@author: zaish
"""

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

samples = np.random.normal(loc = 15, scale = 7, size = 100)

mean = np.mean(samples)
std = np.std(samples, ddof = 1)
n = len(samples)

alpha = 0.32   # 例如 99% CI
t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

half_width = t_crit * std / np.sqrt(n)
lower = mean - half_width
upper = mean + half_width

print(f"95% CI = ({lower:.3f}, {upper:.3f})")
print(mean, std)