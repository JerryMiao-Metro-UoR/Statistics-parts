# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:44:48 2025

@author: zaish
"""

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

bin1 = stats.binom.rvs(n=10, p=0.5, size=1000)

plt.hist(bin1, bins=np.arange(-0.5, 11.5, 1), edgecolor='black')
plt.xlabel("k")
plt.ylabel("Frequency")
plt.title("Histogram of Binomial Samples n=10, p=0.5")
plt.show()