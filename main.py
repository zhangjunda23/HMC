import math
import random

import matplotlib.pyplot as plt
import numpy as np


# 势能
from matplotlib.animation import FuncAnimation


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


if __name__ == '__main__':
    np.random.seed(1234)
    delta = 0.3  # leap frog的步长
    nSample = 200  # 需要采样的样本数量
    L = 20  # leap frog的步数

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
            print(xStar)
        else:
            x[t, :] = x[t - 1, :]

    fig, ax = plt.subplots()
    line, = plt.plot(x[:, 0], x[:, 1], 'ob-')


    def update(i):
        line.set_xdata(x[0:i, 0])
        line.set_ydata(x[0:i, 1])
        return line


    ani = FuncAnimation(fig, update, interval=100)

    plt.show()
