# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:57:04 2025

@author: zaish
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n = 10
p = 0.5

x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

plt.stem(x, pmf, basefmt=" ")
plt.xlabel("k")
plt.ylabel("P(X=k)")
plt.title("Binomial PMF (n=10, p=0.5)")
plt.show()
