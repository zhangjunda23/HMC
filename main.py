import math
import random
import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


# 用HMC采样
def HMC():
    # 势能，常数被忽略
    def U(xx: np.ndarray):
        xx = xx.reshape((1, 2))
        res = (xx[0, 0]) ** 2 + (xx[0, 1]) ** 2
        return res

    # 势能的导数，常数被忽略。这里是求的偏导向量，即[\frac{\partial U}{x_1},\frac{\partial U}{x_2}]。太跳跃了，因为\Sigma是对称矩阵
    # 所以其逆矩阵也是对称矩阵，计算出来就有了dU
    def dU(xx: np.ndarray):
        xx = xx.reshape((1, 2))
        res = xx
        return res

    # 动能
    def K(p: np.ndarray):
        res = np.matmul(p, np.transpose(p))
        return res

    delta = 0.3  # leap frog的步长
    nSample = 1000  # 需要采样的样本数量
    L = 20  # leap frog的步数

    # 初始化
    x = np.zeros((nSample, 2))
    x0 = np.array([12, 5])

    x[0, :] = x0
    t = 0
    vPoint = 0
    rPoint = 0
    vPoint = vPoint + 1
    with tqdm.tqdm(total=nSample) as pbar:
        pbar.set_description('采样进度')
        pbar.update(1)
        while t < nSample - 1:
            t = t + 1
            # p0 = np.random.rand(1, 2)  # 随机采样一个动量，需要服从动量的分布，而不是均匀采样
            p0 = np.random.randn(1, 2)
            # leap frog方法
            pStar = p0 - delta / 2 * dU(x[t - 1, :])  # 动量移动半步
            xStar = x[t - 1, :] + delta * pStar  # 位置移动一步，这里pStar是U对P的偏导
            for jL in range(1, L - 1):
                pStar = pStar - delta * dU(xStar)
                xStar = xStar + delta * pStar
            pStar = pStar - delta / 2 * dU(xStar)

            U0 = U(x[t - 1, :])
            UStar = U(xStar)

            K0 = K(p0)
            KStar = K(pStar)[0, 0]

            # 计算是否接受这个样本
            alpha = min(1., math.exp((U0 + K0) - (UStar + KStar)))
            u = random.random()
            if u < alpha:
                x[t, :] = xStar
                vPoint = vPoint + 1
            else:
                x[t, :] = x[t - 1, :]
                rPoint = rPoint + 1
            pbar.update(1)

    # 画图
    fig, ax = plt.subplots()
    line, = plt.plot(x[:, 0], x[:, 1], 'o', linewidth=0.5, markersize=1)
    fig.suptitle("%d个有效点，%d个重复点" % (vPoint, rPoint))
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)

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
        sigma = np.array([[1, 0], [0, 1]])  # 两个都是标准正态分布，只不过各自向两个方向移动了一定的距离
        c = 1 / (math.sqrt(2 * math.pi * np.linalg.det(sigma)))  # 正态分布exp前面的系数
        m1 = np.matmul(xx - np.array([5, 5]), np.linalg.inv(sigma))
        m1 = np.matmul(m1, np.transpose(xx - np.array([5, 5])))
        m2 = np.matmul(xx + np.array([5, 5]), np.linalg.inv(sigma))
        m2 = np.matmul(m2, np.transpose(xx + np.array([5, 5])))
        res = 1 / 2 * (c * math.exp(-1 / 2 * m1) + c * math.exp(-1 / 2 * m2))  # 两个概率密度相加
        return res

    # 初始化
    nSample = 1000  # 需要采样的样本数量
    x = np.zeros((nSample, 2))
    x0 = np.array([0, 6])
    x[0, :] = x0
    alpha = 0
    currentP = TargetDis(x0)
    t = 0
    rPoint = 0  # 重复点数
    vPoint = 0  # 有效点数
    vPoint = vPoint + 1
    with tqdm.tqdm(total=nSample) as pbar:
        pbar.set_description('采样进度:')
        pbar.update(1)
        while t < nSample - 1:
            t = t + 1
            point = np.random.rand(1, 2) * 20 - 10  # 在[-5,5][-5,5]范围内均匀采样一个点
            p = TargetDis(point)
            alpha = min(1., p / currentP)
            u = random.random()
            if u < alpha:
                x[t, :] = point
                currentP = p
                vPoint = vPoint + 1
            else:
                x[t, :] = x[t - 1, :]
                rPoint = rPoint + 1
            pbar.update(1)

    fig, ax = plt.subplots()
    # line, = plt.plot(x[0, 0], x[0, 1], 'o', linewidth=1, markersize=1)
    line, = plt.plot(x[:, 0], x[:, 1], 'o', linewidth=1, markersize=1)
    fig.suptitle("%d个有效点，%d个重复点" % (vPoint, rPoint))
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
    # ani = FuncAnimation(fig, update, interval=0.01)

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
