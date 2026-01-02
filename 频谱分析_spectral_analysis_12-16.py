# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 11:53:52 2025

@author: zaish
"""

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('MaunaLoa.txt', delimiter=',', skip_header=1)
year = data[:, 0]
co2  = data[:, 1]

plt.figure()

plt.magnitude_spectrum(co2,Fs = 12)
plt.yscale('log')                    

plt.xlabel('Frequency (cycles per year)')
plt.ylabel('Magnitude')
plt.title('Power spectrum of Mauna Loa CO$_2$')
