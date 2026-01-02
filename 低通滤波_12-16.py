# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 11:24:13 2025

@author: zaish
"""

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('MaunaLoa.txt', delimiter=',', skip_header=1)
year = data[:, 0]
co2  = data[:, 1]

data_smooth = np.full_like(co2, np.nan)

half_width = 6

for i in range(half_width, len(co2) - half_width):
    i0 = i - half_width
    i1 = i + half_width
    data_smooth[i] = np.mean(co2[i0:i1])
    
# plt.plot(year, co2, label="raw")
plt.plot(year, data_smooth, label="low-pass (moving avg)")
plt.legend()
plt.show()
