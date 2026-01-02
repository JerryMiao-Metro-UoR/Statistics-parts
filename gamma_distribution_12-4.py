# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:32:21 2025

@author: zaish
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 

x = np.linspace(0, 10, 500)

alpha = 1.1
beta = 0.1

gammapdf = stats.gamma.pdf(x, a=alpha, scale=1/beta)

plt.plot(x, gammapdf)