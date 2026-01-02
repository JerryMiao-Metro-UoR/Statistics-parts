import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.loadtxt("cet_and_nao_DJF.txt", delimiter=",", skiprows=1)

year = data[:, 0]
cet  = data[:, 1]

model = LinearRegression()
model.fit(year.reshape(-1,1), cet)

year_fit = np.linspace(year.min(), year.max(), 200)
cet_fit  = model.predict(year_fit.reshape(-1,1))

plt.figure(figsize=(10,6))
plt.plot(year, cet, label="CET", linewidth=1)
plt.plot(year_fit, cet_fit, label="Trend", linestyle="--")

plt.ylim(-1,8)
plt.xlabel("Year")
plt.ylabel("Central Eng Temp")
plt.title("CET vs Year with Linear Trend")
plt.legend()
plt.show()

print("slope =", model.coef_[0])
print("intercept =", model.intercept_)
