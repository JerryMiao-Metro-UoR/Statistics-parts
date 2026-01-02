import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def accuracy(A24, B24, C24, D24):
    ((A24 + D24) / (A24 + B24 + C24 + D24))
    ACC24 = accuracy(A24, B24, C24, D24)
    return ACC24

def verification_scores(A, B, C, D):
    """输入 2x2 列联表四个格子，返回 POD, POFD, FAR, CSI"""
    A = float(A); B = float(B); C = float(C); D = float(D)

    # 为防止除以 0，这里用 np.nan 作为结果
    def safe_div(num, den):
        return np.nan if den == 0 else num / den

    POD  = safe_div(A, A + C)      # Probability of Detection
    POFD = safe_div(B, B + D)      # Probability of False Detection
    FAR  = safe_div(B, A + B)      # False Alarm Ratio
    CSI  = safe_div(A, A + B + C)  # Critical Success Index

    return POD, POFD, FAR, CSI

def fit_trend(year, value):
    """给定 year 和 value（可包含 NaN），返回：
       x_fit（用于绘图的平滑年份轴）
       y_fit（回归预测值）
       model（sklearn 模型）
    """
    # 去掉 NaN
    mask = ~np.isnan(year) & ~np.isnan(value)
    x = year[mask].reshape(-1,1)
    y = value[mask]

    # 拟合线性趋势
    model = LinearRegression()
    model.fit(x, y)

    # 生成平滑趋势线
    x_fit = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
    y_fit = model.predict(x_fit)

    return x_fit, y_fit, model

data24 = np.genfromtxt('temps2012_24hr.txt', delimiter = ',', skip_header=1)
data72 = np.genfromtxt('temps2012_72hr.txt', delimiter = ',', skip_header=1)

data24_badrows = np.any(np.isnan(data24), axis=1)
data72_badrows = np.any(np.isnan(data72), axis=1)
all_badrows = data24_badrows | data72_badrows
data24 = data24[~all_badrows]
data72 = data72[~all_badrows]

fcst24 = data24[:,1]
obs24 = data24[:,2]

fcst72 = data72[:,1]
obs72 = data72[:,2]

bias24 = fcst24 - obs24
bias72 = fcst72 - obs72

core24 = np.corrcoef(obs24, fcst24)
core72 = np.corrcoef(obs72, fcst72)

bias_mean72 = np.mean(bias72)
bias_mean24 = np.mean(bias24) 

rmse24 = np.sqrt(np.mean((fcst24 - obs24)**2))
rmse72 = np.sqrt(np.mean((fcst72 - obs72)**2))

thres = 19.5

obs24_warm = (data24[:, 2] > thres)
fcst24_warm = (data24[:, 1] > thres)

obs72_warm = (data72[:,2] > thres)
fcst72_warm = (data72[:,1] > thres)

A24 = np.sum(obs24_warm & fcst24_warm)
B24 = np.sum(~obs24_warm & fcst24_warm)
C24 = np.sum(obs24_warm & ~fcst24_warm)
D24 = np.sum(~obs24_warm & ~fcst24_warm)

A72 = np.sum(obs72_warm & fcst72_warm)
B72 = np.sum(~obs72_warm & fcst72_warm)
C72 = np.sum(obs72_warm & ~fcst72_warm)
D72 = np.sum(~obs72_warm & ~fcst72_warm)

POD24, POFD24, FAR24, CSI24 = verification_scores(A24, B24, C24, D24)
print("24h POD =", POD24)
print("24h POFD =", POFD24)
print("24h FAR =", FAR24)
print("24h CSI =", CSI24)

POD72, POFD72, FAR72, CSI72 = verification_scores(A72, B72, C72, D72)
print("72h POD =", POD72)
print("72h POFD =", POFD72)
print("72h FAR =", FAR72)
print("72h CSI =", CSI72)


print ("bias24=",bias_mean24)
print ("bias72=",bias_mean72)
print ("correlation24=",core24)
print ("correlation72=",core72)
print ("24hRMSE=",rmse24)
print ("72hRMSE=",rmse72)
print("24h contingency:")
print("A (hit)             =", A24)
print("B (false alarm)     =", B24)
print("C (miss)            =", C24)
print("D (correct negative)=", D24)
print("ACC=", )
