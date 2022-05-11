import math
import random

import numpy
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


# 用HMC采样
def HMC():
    # 势能
    def U(xx: np.ndarray):
        res = np.matmul(xx, np.linalg.inv(np.array([[1, 0.8], [0.8, 1]])))
        res = np.matmul(res, np.transpose(xx))
        return res

    # 势能的导数
    def dU(xx: np.ndarray):
        res = np.matmul(xx, np.linalg.inv(np.array([[1, 0.8], [0.8, 1]])))
        return res

    # 动能
    def K(p: np.ndarray):
        res = np.matmul(p, np.transpose(p))
        return res

    delta = 0.3  # leap frog的步长
    nSample = 50000  # 需要采样的样本数量
    L = 20  # leap frog的步数，步数越多，收敛得越快

    # 初始化
    x = np.zeros((nSample, 2))
    x0 = np.array([0, 6])

    x[0, :] = x0
    t = 0

    while t < nSample - 1:
        t = t + 1
        p0 = np.random.rand(1, 2)  # 随机采样一个动量
        # leap frog方法
        pStar = p0 - delta / 2 * dU(x[t - 1, :])  # 动量移动半步
        xStar = x[t - 1, :] + delta * pStar  # 位置移动一步，这里pStar是U对P的偏导
        for jL in range(1, L - 1):
            pStar = pStar - delta * dU(xStar)
            xStar = xStar + delta * pStar
        pStar = pStar - delta / 2 * dU(xStar)

        U0 = U(x[t - 1, :])
        UStar = U(xStar)[0, 0]

        K0 = K(p0)
        KStar = K(pStar)[0, 0]

        # 计算是否接受这个样本
        alpha = min(1., math.exp((U0 + K0) - (UStar + KStar)))
        u = random.random()
        if u < alpha:
            x[t, :] = xStar
        else:
            x[t, :] = x[t - 1, :]

    fig, ax = plt.subplots()
    line, = plt.plot(x[:, 0], x[:, 1], 'o', linewidth=0.5, markersize=1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # def update(i):
    #     global repeatCount
    #     global validCount
    #     line.set_xdata(x[0:i, 0])
    #     line.set_ydata(x[0:i, 1])
    #     if i > 0:
    #         if (x[i, :] == x[i - 1, :]).all():
    #             repeatCount = repeatCount + 1
    #         else:
    #             validCount = validCount + 1
    #     fig.suptitle("%d个有效点，%d个重复点" % (validCount, repeatCount))
    #     return line, fig
    #
    # ani = FuncAnimation(fig, update, interval=100)

    plt.show()
    plt.pause(0)


# 用metropolis采样方法
def MH():
    # 二维正态分布求点xx处的概率
    def TargetDis(xx):
        sigma = np.array([[1, 0.8], [0.8, 1]])
        c = 1 / (math.sqrt(2 * math.pi * np.linalg.det(sigma)))
        m = np.matmul(xx, np.linalg.inv(sigma))
        m = np.matmul(m, np.transpose(xx))
        res = c * math.exp(-1 / 2 * m)
        return res

    # 初始化
    nSample = 50000  # 需要采样的样本数量
    x = np.zeros((nSample, 2))
    x0 = np.array([0, 6])
    x[0, :] = x0
    alpha = 0
    currentP = TargetDis(x0)
    t = 0

    while t < nSample - 1:
        t = t + 1
        point = np.random.rand(1, 2) * 20 - 10  # 在[-5,5][-5,5]范围内均匀采样一个点
        p = TargetDis(point)
        alpha = min(1., p / currentP)
        u = random.random()
        if u < alpha:
            x[t, :] = point
            currentP = p
        else:
            x[t, :] = x[t - 1, :]

    fig, ax = plt.subplots()
    # line, = plt.plot(x[0, 0], x[0, 1], 'o', linewidth=1, markersize=1)
    line, = plt.plot(x[:, 0], x[:, 1], 'o', linewidth=1, markersize=1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    def update(i):
        global repeatCount
        global validCount
        line.set_xdata(x[0:i, 0])
        line.set_ydata(x[0:i, 1])
        if i > 0:
            if (x[i, :] == x[i - 1, :]).all():
                repeatCount = repeatCount + 1
            else:
                validCount = validCount + 1
        fig.suptitle("%d个有效点，%d个重复点" % (validCount, repeatCount))
        return line, fig

    ani = FuncAnimation(fig, update, interval=0.01)

    plt.show()
    plt.pause(0)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    np.random.seed(1234)

    repeatCount = 0
    validCount = 0
    HMC()
    # MH()
