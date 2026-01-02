# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:27:39 2025

@author: zaish
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 

x = np.linspace(-5, 5, 100)

normpdf = stats.norm.pdf(x, loc=0, scale=1)

plt.plot(x, normpdf)