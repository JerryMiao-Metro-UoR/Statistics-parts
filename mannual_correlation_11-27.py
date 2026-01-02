import numpy as np
# import statsmodels.api as sm

data = np.loadtxt("data.txt", skiprows= 1)

x1 = data[:, 0]
y1 = data[:, 1]
x2 = data[:, 2]
y2 = data[:, 3]
x3 = data[:, 4]
y3 = data[:, 5]
x4 = data[:, 6]
y4 = data[:, 7]

# X1 = sm.add_constant(x1)
# fit = sm.OLS(y1, X1).fit()

def pearson_r(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    xm = x - x.mean()
    ym = y - y.mean()
    return np.sum(xm * ym) / np.sqrt(np.sum(xm**2) * np.sum(ym**2))

def ols(x, y):
    """
    对一元线性回归 y ~ beta0 + beta1 * x
    返回: beta0, beta1, R2
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    # 斜率 beta1
    num = np.sum((x - x_mean) * (y - y_mean))        # 分子：协方差的未除以 n 形式
    den = np.sum((x - x_mean) ** 2)                  # 分母：x 的方差的未除以 n 形式
    beta1 = num / den

    # 截距 beta0
    beta0 = y_mean - beta1 * x_mean

    # 预测值
    y_hat = beta0 + beta1 * x

    # R^2
    sst = np.sum((y - y_mean) ** 2)                  # total sum of squares
    sse = np.sum((y - y_hat) ** 2)                   # sum of squared errors
    r2 = 1 - sse / sst

    return beta0, beta1, r2



for i in range(4):
    x = data[:, 2*i]      # x1, x2, x3, x4
    y = data[:, 2*i + 1]  # y1, y2, y3, y4

    beta0, beta1, r2 = ols(x, y)
    print(f"Pair {i+1}: (x{i+1}, y{i+1})")
    print(f"  beta0 = {beta0:.4f}, beta1 = {beta1:.4f}, R^2 = {r2:.4f}")


# r1 = pearson_r(x1, y1)

# print (fit.summary())

# corr1 = np.corrcoef(x1, y1)[0, 1]
# corr2 = np.corrcoef(x2, y2)[0, 1]
# corr3 = np.corrcoef(x3, y3)[0, 1]
# corr4 = np.corrcoef(x4, y4)[0, 1]

# print("corr(x1, y1) =", corr1)
# print("corr(x2, y2) =", corr2)
# print("corr(x3, y3) =", corr3)
# print("corr(x4, y4) =", corr4)
