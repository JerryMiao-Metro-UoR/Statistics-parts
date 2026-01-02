# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 16:20:24 2025

@author: zaish
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('winds2015.txt', delimiter=',', skip_header=1)
day = data[:,0]
wind = data[:,1]

mean_wind = np.mean(wind)
std_wind = np.std(wind, ddof = 1)
n = len(wind)

half = 1.96 * std_wind / np.sqrt(n)
upper = mean_wind + half
lower = mean_wind - half

plt.figure(figsize=(8,5))
plt.hist(wind, bins=20, density=False, alpha=0.7, edgecolor='black')
plt.axvline(mean_wind, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_wind:.2f}")

plt.title("Wind Speed Distribution (2015)")
plt.xlabel("Wind speed")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

mean_wind = np.mean(wind)
std_wind = np.std(wind, ddof=1)

print("Mean =", mean_wind)
print("Std dev =", std_wind)
print(f"95% CI = ({lower:.3f}, {upper:.3f})")

# plt.figure(figsize=(10,5))
# plt.scatter(day, wind, s=10, alpha=0.6, label="Wind")
# plt.axhline(mean_wind, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_wind:.2f}")

# plt.title("Daily Wind Speed (2015)")
# plt.xlabel("Day of Year")
# plt.ylabel("Wind speed")
# plt.grid(alpha=0.3)
# plt.legend()
# plt.show()
