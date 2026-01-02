# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 11:59:54 2025

@author: zaish
"""

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('NAO_SSTi.txt', delimiter=',', skip_header=1)
year = data[:, 0]
nao  = data[:, 1]
ssti = data[:, 2]

from statsmodels.graphics import tsaplots
tsaplots.plot_acf(nao)

inst_corr = np.corrcoef(nao, ssti)

lag1_corr = np.corrcoef(nao[5:], nao[:-5])[0, 1]
print(lag1_corr)
print(inst_corr)