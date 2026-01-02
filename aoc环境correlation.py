import numpy as np
import statsmodels.api as sm

data = np.loadtxt("data-lec4.txt", skiprows= 1)

x1 = data[:, 0]
y1 = data[:, 1]
x2 = data[:, 2]
y2 = data[:, 3]
x3 = data[:, 4]
y3 = data[:, 5]
x4 = data[:, 6]
y4 = data[:, 7]

X1 = sm.add_constant(x1)
fit = sm.OLS(y1, X1).fit()

print (fit.summary())

corr1 = np.corrcoef(x1, y1)[0, 1]
corr2 = np.corrcoef(x2, y2)[0, 1]
corr3 = np.corrcoef(x3, y3)[0, 1]
corr4 = np.corrcoef(x4, y4)[0, 1]

print("corr(x1, y1) =", corr1)
print("corr(x2, y2) =", corr2)
print("corr(x3, y3) =", corr3)
print("corr(x4, y4) =", corr4)