import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = np.loadtxt("cet_and_nao_DJF.txt",delimiter=",", skiprows= 1)

year = data[:,0]
cet = data[:,1]
nao = data[:,2]

x = sm.add_constant(year)
model = sm.OLS(nao, x).fit()

year_fit = np.linspace(year.min(), year.max(), 200)
X_fit = sm.add_constant(year_fit)
cet_fit = model.predict(X_fit)

plt.figure(figsize=(10, 6))
plt.plot(year, nao, label = 'CET')
plt.plot(year_fit, cet_fit, label = 'linear trend', linestyle = 'dashdot')
plt.ylim()
plt.xlabel("year")
plt.ylabel("central Eng Temp")
plt.title("Nao-year")
plt.show()
