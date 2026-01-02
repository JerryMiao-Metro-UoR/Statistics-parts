# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:23:04 2025

@author: zaish
"""

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 11)
poi3 = stats.poisson.pmf(x, mu = 5)

plt.stem(x, poi3)
plt.xlabel("k")
plt.ylabel("P(X=k)")
plt.title("Poisson PMF (mu=25)")
plt.show()
print (poi3)