# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 12:38:04 2025

@author: zaish
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

data = np.genfromtxt('NAO_SSTi.txt', delimiter=',', skip_header=1)
year = data[:, 0]
nao  = data[:, 1]
ssti = data[:, 2]

lag = 5

X = np.column_stack([nao[:-lag], ssti[:-lag]])
X = sm.add_constant(X)
y = nao[lag:]

results = sm.OLS(y, X).fit()
print(results.summary())
