import math
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from scipy.misc import derivative
from time import perf_counter
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32


@cuda.jit(device=True)
# 目标分布pdf，不归一化
def TargetPDF(xin, yin, zin):
    res = math.exp(-1 / 2 * ((xin - 3) ** 2 + (yin - 3) ** 2 + (zin - 3) ** 2)) + 1 / 2 * math.exp(
        -1 / 4 * ((xin + 3) ** 2 + (yin + 3) ** 2 + (zin + 3) ** 2))
    return res


# 能量
@cuda.jit(device=True)
def TargetPDFU(xin, yin, zin):
    return -math.log(TargetPDF(xin, yin, zin))


# 计算偏导，固定其他维度，在要求点的左右各0.0001范围内近似求取
@cuda.jit(device=True)
def partial_derivative(var: int, point: np.ndarray):
    left = point[var] - 0.0001
    right = point[var] + 0.0001
    if var == 0:
        return (TargetPDFU(right, point[1], point[2]) - TargetPDFU(left, point[1], point[2])) / 0.0002
    elif var == 1:
        return (TargetPDFU(point[0], right, point[2]) - TargetPDFU(point[0], left, point[2])) / 0.0002
    else:
        return (TargetPDFU(point[0], point[1], right) - TargetPDFU(point[0], point[1], left)) / 0.0002


# U、dU、K、dK需要准确计算，不能随意舍掉前面的系数，梯度的减小等价于leap frog方法里步长的减小
# 势能，常数被忽略
@cuda.jit(device=True)
def U(xx: np.ndarray):
    # global interpolationCount
    # xx = xx.reshape((1, 3))
    res = TargetPDFU(xx[0], xx[1], xx[2])
    # res_interpolationCount = res_interpolationCount + 1
    return res


# 势能的导数
@cuda.jit(device=True)
def dU(xx: np.ndarray, res: np.ndarray):
    # global interpolationCount
    # res_interpolationCount = res_interpolationCount + 1

    res[0] = partial_derivative(0, xx)
    res[1] = partial_derivative(1, xx)
    res[2] = partial_derivative(2, xx)


# 动能
@cuda.jit(device=True)
def K(p: np.ndarray):
    return p[0] ** 2 + p[1] ** 2 + p[2] ** 2


# 动能的导数
@cuda.jit(device=True)
def dK(p: np.ndarray, res: np.ndarray):
    res[0] = 2 * p[0]
    res[1] = 2 * p[1]
    res[2] = 2 * p[2]


# 用HMC采样
@cuda.jit
def HMC_CUDA(n,
             in_initPoint,
             in_p0,
             pStar,
             xStar,
             res_dUArray,
             res_dKArray,
             res_x,
             res_validCount,
             in_rng):
    threadID = cuda.grid(1)  # 获取线程的绝对id
    thread_Sum = cuda.gridsize(1)  # 线程总数
    # print(thread_Sum)

    # print(cuda.grid(1))

    delta = 0.1  # leap frog的步长
    L = 20  # leap frog的步数

    # 初始化
    x0 = in_initPoint[threadID]  # 获取和线程对应的那个初始位置
    # res_x[0, :] = x0
    res_x[n * threadID, 0] = x0[0]
    res_x[n * threadID, 1] = x0[1]
    res_x[n * threadID, 2] = x0[2]
    t = 0
    # res_validCount = res_validCount + 1
    t = t + 1

    while res_validCount < n:
        p0 = in_p0
        p0[0] = xoroshiro128p_normal_float32(in_rng, threadID)
        p0[1] = xoroshiro128p_normal_float32(in_rng, threadID)
        p0[2] = xoroshiro128p_normal_float32(in_rng, threadID)

        # leap frog方法
        # 动量移动半步
        dU(res_x[n * threadID + t - 1, :], res_dUArray)  # 计算动量，结果保存在res_dUArray
        pStar[0] = p0[0] - delta / 2 * res_dUArray[0]
        pStar[1] = p0[1] - delta / 2 * res_dUArray[1]
        pStar[2] = p0[2] - delta / 2 * res_dUArray[2]

        # 位置移动一步
        dK(pStar, res_dKArray)
        xStar[0] = delta * res_dKArray[0] + res_x[n * threadID + t - 1, 0]
        xStar[1] = delta * res_dKArray[1] + res_x[n * threadID + t - 1, 1]
        xStar[2] = delta * res_dKArray[2] + res_x[n * threadID + t - 1, 2]

        for jL in range(1, L - 1):
            dU(xStar, res_dUArray)
            pStar[0] = pStar[0] - delta * res_dUArray[0]
            pStar[1] = pStar[1] - delta * res_dUArray[1]
            pStar[2] = pStar[2] - delta * res_dUArray[2]
            dK(pStar, res_dKArray)
            xStar[0] = xStar[0] + delta * res_dKArray[0]
            xStar[1] = xStar[1] + delta * res_dKArray[1]
            xStar[2] = xStar[2] + delta * res_dKArray[2]

        dU(xStar, res_dUArray)
        pStar[0] = pStar[0] - delta / 2 * res_dUArray[0]
        pStar[1] = pStar[1] - delta / 2 * res_dUArray[1]
        pStar[2] = pStar[2] - delta / 2 * res_dUArray[2]
        # pStar = pStar - delta / 2 * dU(xStar, res_dUArray)  # 动量移动半步

        U0 = U(res_x[n * threadID + t - 1, :])
        UStar = U(xStar)

        K0 = K(p0)
        KStar = K(pStar)

        # print(threadID, 'OK')
        # 计算是否接受这个样本
        alpha = min(1., math.exp((U0 + K0) - (UStar + KStar)))
        # u = random.random()
        u = xoroshiro128p_uniform_float32(in_rng, threadID)
        if u < alpha:
            # res_x[t, :] = xStar
            res_x[n * threadID + t, 0] = xStar[0]
            res_x[n * threadID + t, 1] = xStar[1]
            res_x[n * threadID + t, 2] = xStar[2]
            res_validCount = res_validCount + 1
            t = t + 1

        else:
            # x[t, :] = x[t - 1, :]
            # res_repeatCount = res_repeatCount + 1
            pass

    # # res_interpolationCount = interpolationCount
    # print(res_interpolationCount, res_validCount, res_repeatCount)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    np.random.seed(123)
    # repeatCount = 0  # 重复点的个数
    validCount = 0  # 有效点的个数
    sampleCount = 10000  # 每个线程采样的点数
    deviation = 3  # 三维正态分布均值偏离量
    # interpolationCount = 0  # 重构次数的计数
    initpStar = np.zeros((3,))
    initxStar = np.zeros((3,))
    initP0 = np.zeros(3, )  # 动量初始位置
    dUArray = np.zeros((3,))
    dKArray = np.zeros((3,))
    threads_per_block = 32  # 每个block包含的线程数
    blocks_per_Grid = 32  # grid包含的block个数
    threadSum = threads_per_block * blocks_per_Grid  # 线程总数
    # 以下参数需要针对每个线程
    # initPoint = np.array([12, 5, 5])  # 初始的采样位置
    initPoint = np.random.randint(20, size=(threadSum, 3)) - 10
    # x = np.zeros((sampleCount, 3))  # 保存结果的数组
    x = np.zeros((sampleCount * threadSum, 3))  # 保存结果的数组
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_Grid, seed=1)  # GPU生成随机数需要的states
    print('Thread Sum:', threadSum)
    print('Point Sum:', sampleCount * threadSum)
    t1 = perf_counter()
    HMC_CUDA[blocks_per_Grid, threads_per_block](sampleCount,
                                                 initPoint,
                                                 initP0,
                                                 initpStar,
                                                 initxStar,
                                                 dUArray,
                                                 dKArray,
                                                 x,
                                                 validCount,
                                                 rng_states)

    cuda.synchronize()
    t2 = perf_counter()
    t_consume = t2 - t1
    print('耗时%.3f秒' % t_consume)
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # fig, ax = plt.subplots()
    line, = plt.plot(x[:, 0], x[:, 1], x[:, 2], 'o', linewidth=0.5, markersize=0.1)
    # fig.suptitle("HMC算法 %d个有效点，%d个重复点 耗时%.3f秒 %d次重构" % (vPoint, rPoint, t2 - t1, interpolationCount))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    plt.show()
