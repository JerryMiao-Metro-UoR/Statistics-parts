# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 12:24:40 2025

@author: zaish
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = np.genfromtxt('NAO_SSTi.txt', delimiter=',', skip_header=1)
year = data[:, 0]
nao  = data[:, 1]
ssti = data[:, 2]

lag = 5

y = nao[lag:]
X = np.column_stack([
    nao[:-lag],               
    ssti[:-lag]                
])

model = LinearRegression()
model.fit(X, y)

bxx, bxy = model.coef_
intercept = model.intercept_
lag5 = bxx + bxy* np. corrcoef(nao, ssti)

plt.plot(bxx,)
plt.show

print(lag5)
print(bxx, bxy, intercept)