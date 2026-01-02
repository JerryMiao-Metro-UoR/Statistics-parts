import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("cet_and_nao_DJF.txt",delimiter=",", skiprows= 1)

year = data[:,0]
cet = data[:,1]
nao = data[:,2]

fit_mlr = sm.OLS(cet, sm.add_constant(np.stack([year, nao], axis=1))).fit()
fit_mlr.summary()

# 生成 year 和 nao 的网格
year_grid = np.linspace(year.min(), year.max(), 30)
nao_grid  = np.linspace(nao.min(), nao.max(), 30)
Yg, Ng    = np.meshgrid(year_grid, nao_grid)  # Yg: year 面, Ng: nao 面

# 把网格展开成 (Ngrid, 2) -> [year, nao]
X_grid = np.column_stack([Yg.ravel(), Ng.ravel()])
X_grid = sm.add_constant(X_grid)

# 用回归模型在网格上预测 CET
cet_grid = fit_mlr.predict(X_grid)
Cet_surf = cet_grid.reshape(Yg.shape)   # 再 reshape 回 2D，方便画面

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 原始数据的 3D 散点
ax.scatter(year, nao, cet, s=20, label='Observations')

# 拟合平面
ax.plot_surface(Yg, Ng, Cet_surf, alpha=0.4, linewidth=0, antialiased=True)

Z_fit = fit_mlr.predict(sm.add_constant(np.column_stack([year, nao])))

# 添加垂直线（从平面 --> 散点，残差）
for xi, yi, zi_obs, zi_fit in zip(year, nao, cet, Z_fit):
    ax.plot([xi, xi], [yi, yi], [zi_fit, zi_obs], 
            color='black', linewidth=0.8, alpha=0.6)

ax.set_xlabel("Year")
ax.set_ylabel("NAO index (DJF)")
ax.set_zlabel("CET (°C)")
ax.set_title("Multiple Linear Regression: CET ~ Year + NAO")
plt.legend()
plt.tight_layout()
plt.show()