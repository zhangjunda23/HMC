import math
import random

import numpy
import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
import time
import matplotlib.ticker as ticker


# 用HMC采样
def HMC(target_count: int):
    global interpolationCount, sampleCount, deviation

    # 用scipy求偏导，参数为（目标函数，求偏导得维度，求偏导的点）
    def partial_derivative(func, var=0, point=[]):
        args = point[:]

        def wraps(xx):
            args[var] = xx
            return func(*args)

        return derivative(wraps, point[var], dx=1e-6)

    # 目标分布pdf，不归一化
    def TargetPDF(xin, yin, zin):
        res = 0
        res = math.exp(-1 / 2 * ((xin - deviation) ** 2 + (yin - deviation) ** 2 + (zin - deviation) ** 2)) + \
              1 / 2 * math.exp(-1 / 4 * ((xin + deviation) ** 2 + (yin + deviation) ** 2 + (zin + deviation) ** 2))

        return res

    # 能量
    def TargetPDFU(xin, yin, zin):
        return -math.log(TargetPDF(xin, yin, zin))

    # U、dU、K、dK需要准确计算，不能随意舍掉前面的系数，梯度的减小等价于leap frog方法里步长的减小
    # 势能，常数被忽略
    def U(xx: np.ndarray):
        global interpolationCount
        xx = xx.reshape((1, 3))
        res = TargetPDFU(xx[0, 0], xx[0, 1], xx[0, 2])
        interpolationCount = interpolationCount + 1
        return res

    # 势能的导数
    def dU(xx: np.ndarray):
        global interpolationCount
        xx = xx.reshape((1, 3))
        interpolationCount = interpolationCount + 1
        res = np.array(
            [partial_derivative(TargetPDFU, 0, xx.tolist()[0]),
             partial_derivative(TargetPDFU, 1, xx.tolist()[0]),
             partial_derivative(TargetPDFU, 2, xx.tolist()[0])])
        return res

    # 动能
    def K(p: np.ndarray):
        res = np.matmul(p, np.transpose(p))
        return res

    # 动能的导数
    def dK(p: np.ndarray):
        res = 2 * p
        return res

    delta = 0.1  # leap frog的步长
    nSample = target_count  # 需要采样的样本数量
    L = 20  # leap frog的步数

    # 初始化
    x = np.zeros((nSample, 3))
    x0 = np.array([12, 5, 5])

    x[0, :] = x0
    t = 0
    vPoint = 0
    rPoint = 0
    vPoint = vPoint + 1

    t = t + 1
    t1 = time.perf_counter()
    with tqdm.tqdm(total=nSample) as pbar:
        pbar.set_description('采样进度')
        pbar.update(1)
        while vPoint < nSample:
            p0 = np.random.randn(1, 3)
            # leap frog方法
            pStar = p0 - delta / 2 * dU(x[t - 1, :])  # 动量移动半步
            xStar = x[t - 1, :] + delta * dK(pStar)  # 位置移动一步
            for jL in range(1, L - 1):
                pStar = pStar - delta * dU(xStar)
                xStar = xStar + delta * dK(pStar)
            pStar = pStar - delta / 2 * dU(xStar)  # 动量移动半步

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
                t = t + 1
            else:
                # x[t, :] = x[t - 1, :]
                rPoint = rPoint + 1
                pass
            pbar.update(1)
    t2 = time.perf_counter()
    # 画图
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # fig, ax = plt.subplots()
    # line, = plt.plot(x[:, 0], x[:, 1], x[:, 2], 'o', linewidth=0.5, markersize=0.1)
    # fig.suptitle("HMC算法 %d个有效点，%d个重复点 耗时%.3f秒 %d次重构" % (vPoint, rPoint, t2 - t1, interpolationCount))
    # print('hmc生成耗时：{}\nhmc每粒子耗时：{}'.format((t2 - t1), (t2 - t1) / vPoint))
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    # ax.set_zlim(-10, 10)

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
    #
    # plt.show()
    # plt.pause(0)

    return x, t2 - t1


# 用metropolis采样方法
def MH(target_count: int):
    # 三维正态分布求点xx处的概率
    global interpolationCount, sampleCount, deviation

    def TargetDis(xx):
        sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 两个都是标准正态分布，只不过各自向两个方向移动了一定的距离
        # c = 1 / (math.sqrt(2 * math.pi * np.linalg.det(sigma)))  # 正态分布exp前面的系数
        c = 1  # 系数就不算了，反正不影响采样
        m1 = np.matmul(xx - np.array([deviation, deviation, deviation]), np.linalg.inv(sigma))
        m1 = np.matmul(m1, np.transpose(xx - np.array([deviation, deviation, deviation])))
        m2 = np.matmul(xx + np.array([deviation, deviation, deviation]), np.linalg.inv(sigma))
        m2 = np.matmul(m2, np.transpose(xx + np.array([deviation, deviation, deviation])))
        res = 1 / 2 * (c * math.exp(-1 / 2 * m1) + 1 / 2 * c * math.exp(-1 / 4 * m2))  # 两个概率密度相加
        return res

    # 初始化
    interpolationCount = 0
    nSample = target_count  # 需要采样的样本数量
    x = np.zeros((nSample, 3))
    x0 = np.array([12, 5, 5])
    x[0, :] = x0
    alpha = 0
    currentP = TargetDis(x0)
    t = 0
    rPoint = 0  # 重复点数
    vPoint = 0  # 有效点数
    vPoint = vPoint + 1
    t = t + 1
    t1 = time.perf_counter()
    with tqdm.tqdm(total=nSample) as pbar:
        pbar.set_description('采样进度')
        pbar.update(1)
        while vPoint < nSample:
            point = np.random.rand(1, 3) * 20 - 10  # 在一定范围内均匀采样
            # point = np.random.randn(1, 3) + x[t - 1, :]  # 以上一个点为中心的标准正态分布采样下一个点
            p = TargetDis(point)
            interpolationCount = interpolationCount + 1
            alpha = min(1., p / currentP)
            u = random.random()
            if u < alpha:
                x[t, :] = point
                currentP = p
                vPoint = vPoint + 1
                t = t + 1
            else:
                # x[t, :] = x[t - 1, :]
                rPoint = rPoint + 1
                pass
            pbar.update(1)

    t2 = time.perf_counter()
    # figMH = plt.figure()
    # axMH = figMH.add_subplot(projection='3d')
    # # fig, ax = plt.subplots(projection='3d')
    # # line, = plt.plot(x[0, 0], x[0, 1], 'o', linewidth=1, markersize=1)
    # lineMH, = plt.plot(x[:, 0], x[:, 1], x[:, 2], 'o', linewidth=1, markersize=0.1)
    # figMH.suptitle("MH算法 %d个有效点，%d个重复点 耗时%.3f秒 %d次重构" % (vPoint, rPoint, t2 - t1, interpolationCount))
    print('mh生成耗时：{}\nmh每粒子耗时：{}'.format((t2 - t1), (t2 - t1) / vPoint))
    # axMH.set_xlabel('X')
    # axMH.set_ylabel('Y')
    # axMH.set_zlabel('Z')
    # axMH.set_xlim(-10, 10)
    # axMH.set_ylim(-10, 10)
    # axMH.set_zlim(-10, 10)

    # def update(i):
    #     global repeatCount
    #     global validCount
    #     lineMH.set_xdata(x[0:i, 0])
    #     lineMH.set_ydata(x[0:i, 1])
    #     if i > 0:
    #         if (x[i, :] == x[i - 1, :]).all():
    #             repeatCount = repeatCount + 1
    #         else:
    #             validCount = validCount + 1
    #     figMH.suptitle("%d个有效点，%d个重复点" % (validCount, repeatCount))
    #     return lineMH, figMH
    #
    # ani = FuncAnimation(figMH, update, interval=0.01)

    return x


# 生成负粒子
def NegParticles(x, delete_count: int):
    # x = MH(target_count)  # 获得用MH方法生成的粒子
    global interpolationCount, sampleCount, deviation
    x_in = x.copy()

    # 原分布
    def TargetDis_Ori(xx):
        sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 两个都是标准正态分布，只不过各自向两个方向移动了一定的距离
        # c = 1 / (math.sqrt(2 * math.pi * np.linalg.det(sigma)))  # 正态分布exp前面的系数
        c = 1  # 系数就不算了，反正不影响采样
        m1 = np.matmul(xx - np.array([deviation, deviation, deviation]), np.linalg.inv(sigma))
        m1 = np.matmul(m1, np.transpose(xx - np.array([deviation, deviation, deviation])))
        m2 = np.matmul(xx + np.array([deviation, deviation, deviation]), np.linalg.inv(sigma))
        m2 = np.matmul(m2, np.transpose(xx + np.array([deviation, deviation, deviation])))
        res = 1 / 2 * (c * math.exp(-1 / 2 * m1) + 1 / 2 * c * math.exp(-1 / 4 * m2))  # 两个概率密度相加
        return res

    # 负粒子的分布函数
    def TargetDis(xx):
        sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 两个都是标准正态分布，只不过各自向两个方向移动了一定的距离
        # c = 1 / (math.sqrt(2 * math.pi * np.linalg.det(sigma)))  # 正态分布exp前面的系数
        c = 1  # 系数就不算了，反正不影响采样
        m1 = np.matmul(xx - np.array([deviation, deviation, deviation]), np.linalg.inv(sigma))
        m1 = np.matmul(m1, np.transpose(xx - np.array([deviation, deviation, deviation])))
        m2 = np.matmul(xx + np.array([deviation, deviation, deviation]), np.linalg.inv(sigma))
        m2 = np.matmul(m2, np.transpose(xx + np.array([deviation, deviation, deviation])))
        res = 1 / 2 * (c * math.exp(-1 / 2 * m1) + 1 / 2 * c * math.exp(-1 / 4 * m2))  # 两个概率密度相加
        return res

    # 初始化
    interpolationCount = 0
    nSample = delete_count  # 需要减掉的样本数量
    # nSample = sampleCount
    y = np.zeros((nSample, 3))
    # y0 = np.array([12, 5, 5])
    y0 = x_in[random.randint(0, x_in.shape[0]) - 1, :]  # 在x中随机选择一个点作为初始点
    y[0, :] = y0
    alpha = 0
    currentP = TargetDis(y0)
    t = 0
    rPoint = 0  # 重复点数
    vPoint = 0  # 有效点数
    ySize = 0  # 不重复的y点的个数
    vPoint = vPoint + 1
    t = t + 1
    t1 = time.perf_counter()
    with tqdm.tqdm(total=nSample) as pbar:
        pbar.set_description('采样进度')
        pbar.update(1)

        while vPoint < nSample:
            # point = np.random.rand(1, 3) * 20 - 10  # 在一定范围内均匀采样
            selectRow = random.randint(1, x_in.shape[0]) - 1
            point = x_in[selectRow, :]  # 有可能取到相同的点
            # point = np.random.randn(1, 3) + x[t - 1, :]  # 以上一个点为中心的标准正态分布采样下一个点
            p = TargetDis(point)
            interpolationCount = interpolationCount + 1
            alpha = min(1., p / currentP * TargetDis_Ori(point) / TargetDis_Ori(currentP))  # MH算法
            u = random.random()
            if u < alpha:
                # isContain = np.any(y == point)  # 判断是否已经包含了这个点
                '''
                这种排除y中重复点的做法，当需要删除的点数特别接近原点数时，会让y中元素的数量很难到达nSample，
                因为最后少数不在y中的点满足不了MH算法，导致收敛很慢。
                '''
                # if not isContain:
                y[t, :] = point
                x_in = np.delete(x_in, selectRow, axis=0)  # 在原点集中删除这个点
                currentP = p
                vPoint = vPoint + 1
                t = t + 1
                # else:
                #     rPoint = rPoint + 1
            else:
                # x[t, :] = x[t - 1, :]
                rPoint = rPoint + 1

            pbar.update(1)

    t2 = time.perf_counter()
    # figNeg = plt.figure()
    # axNeg = figNeg.add_subplot(projection='3d')
    # # fig, ax = plt.subplots(projection='3d')
    # # line, = plt.plot(x[0, 0], x[0, 1], 'o', linewidth=1, markersize=1)
    # lineNeg, = plt.plot(y[:, 0], y[:, 1], y[:, 2], 'or', linewidth=1, markersize=0.1)
    # figNeg.suptitle(
    #     "负粒子算法 %d个有效点，%d个重复点 耗时%.3f秒 %d次重构" % (vPoint, rPoint, t2 - t1, interpolationCount))
    # print('删除耗时{}\nnegative粒子每粒子耗时：{}'.format((t2 - t1), (t2 - t1) / vPoint))
    # axNeg.set_xlabel('X')
    # axNeg.set_ylabel('Y')
    # axNeg.set_zlabel('Z')
    # axNeg.set_xlim(-10, 10)
    # axNeg.set_ylim(-10, 10)
    # axNeg.set_zlim(-10, 10)

    # 在原样本中删除样本
    # x = x[x[:, 0].argsort()]  # 按第一列进行排序
    # y = y[y[:, 0].argsort()]
    #
    # assert np.equal(x, y).all(), 'x和y不相等'

    return t2 - t1  # 返回删除总耗时


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    np.random.seed(123)
    repeatCount = 0
    validCount = 0
    sampleCount = 10000
    deviation = 3
    interpolationCount = 0  # 重构次数的计数
    # HMC(sampleCount)
    # MH(sampleCount)
    delete_ratio = 0.7  # 删除粒子占原粒子数的比例

    # NegParticles(sampleCount, int(sampleCount * delete_ratio))  # 用删除粒子的方法
    # HMC(int(sampleCount - sampleCount * delete_ratio))  # 用直接生成的方法
    # MH(int(sampleCount - sampleCount * delete_ratio))  # 用直接生成的方法

    # x,_ = HMC(sampleCount)  # 用直接生成的方法
    # np.save(str(sampleCount), x)

    x = np.load(str(sampleCount) + '.npy')
    num = 10  # 把粒子数分成num份
    repeat = 1
    delTime = np.zeros((num * repeat, 2))  # 删除的结果
    genTime = np.zeros((num * repeat, 2))  # 生成的结果
    for i in range(num):
        # t_average = 0
        for u in range(repeat):
            # 删除
            print('删除{}%的粒子'.format(1 / num * i * 100))
            if i != 0:  # 排除删除0个粒子
                t = NegParticles(x, int(1 / num * i * sampleCount))  # 用删除粒子的方法
                delTime[i * repeat + u, 0] = int(1 / num * i * sampleCount) / sampleCount * 100
                delTime[i * repeat + u, 1] = t

            _, t_gen = HMC(int(sampleCount - 1 / num * i * sampleCount))
            genTime[i * repeat + u, 0] = 100 - delTime[i * repeat + u, 0]
            genTime[i * repeat + u, 1] = t_gen / 9 * 13

    # 对最右侧的点右侧进行更细的分割
    rightlimit = 1 / num * (num - 1)  # 对rightlimit到1之间的区域进行分割
    rightParticleCount = sampleCount * rightlimit
    num_more = 5
    step_more = (1 - rightlimit) / num_more
    delTime_more = np.zeros((num_more * repeat - repeat, 2))
    genTime_more = np.zeros((num_more * repeat - repeat, 2))
    for i in range(1, num_more):
        # t_average = 0
        for u in range(repeat):
            # 删除
            print('删除{}%的粒子'.format((step_more * i + rightlimit) * 100))
            t = NegParticles(x, int((step_more * i + rightlimit) * sampleCount))  # 用删除粒子的方法
            delTime_more[(i - 1) * repeat + u, 0] = (step_more * i + rightlimit) * 100
            delTime_more[(i - 1) * repeat + u, 1] = t

            _, t_gen = HMC(int(sampleCount - (step_more * i + rightlimit) * sampleCount))
            genTime_more[(i - 1) * repeat + u, 0] = 100 - delTime_more[(i - 1) * repeat + u, 0]
            genTime_more[(i - 1) * repeat + u, 1] = t_gen / 9 * 13

    # 手动添加最接近100%的点
    delTime99 = np.zeros((1, 2))
    # genTime99 = np.zeros((1, 2))
    for i in range(1):
        t = NegParticles(x, int(0.995 * sampleCount))
        delTime99[i, 0] = 0.995 * 100
        delTime99[i, 1] = t

    # _, t_gen = HMC(int(0.99 * sampleCount))
    # genTime99[0, 0] = 100 - delTime99[0, 0]
    # genTime99[0, 1] = t_gen

    # 组合不同分割密度的结果
    delTime_all = np.vstack((delTime, delTime_more, delTime99))
    genTime_all = np.vstack((genTime, genTime_more))
    # 多项式拟合
    delete_fit = np.polyfit(delTime_all[:, 0], delTime_all[:, 1], 8)
    generate_fit = np.polyfit(genTime_all[:, 0], genTime_all[:, 1], 2)
    # fit_delete = np.polyval(np.polyfit(delTime_all[:, 0], delTime_all[:, 1], 5), delTime_all[:, 0])
    # fit_generate = np.polyval(np.polyfit(genTime_all[:, 0], genTime_all[:, 1], 3), genTime_all[:, 0])
    fit_delete = np.polyval(delete_fit, np.arange(1, 100, .1))
    fit_generate = np.polyval(generate_fit, np.arange(1, 100, .1))

    # 计算两条回归线的大概交点
    move_step = 0.01
    move_point = 90
    gen_move = np.polyval(generate_fit, 100 - move_point)
    del_move = np.polyval(delete_fit, move_point)
    while gen_move > del_move:
        move_point = move_point + move_step
        gen_move = np.polyval(generate_fit, 100 - move_point)
        del_move = np.polyval(delete_fit, move_point)
    cross_point_y = np.polyval(delete_fit, move_point)
    cross_point_x = move_point
    print('两条回归线交于({},{})'.format(cross_point_x, cross_point_y))

    fig, ax = plt.subplots(figsize=(5, 3.09))
    ax2 = ax.twiny()
    ax2.invert_xaxis()
    dot1, = ax.plot(delTime_all[:, 0], delTime_all[:, 1], 'o', markersize=5, color='#f11616',
                    label='Particle deleting')
    ax.plot(np.arange(1, 100, 0.1), fit_delete, '-', linewidth=2, color='#f1161655')
    # ax.plot(delTime_all[:, 0], delTime_all[:, 1], '-', linewidth=2, color='#f1161655')
    dot2, = ax2.plot(genTime_all[:, 0], genTime_all[:, 1], 'o', markersize=5, color='#2d2c56',
                     label='Particle generating')
    ax2.plot(np.arange(1, 100, 0.1), fit_generate, '-', linewidth=2, color='#2d2c5655')
    ax.plot([cross_point_x, cross_point_x], [cross_point_y, -1], '--', linewidth=1, color='#11111166')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlabel('Percentage of particles to be deleted(%)')
    ax2.set_xlabel('Percentage of particles to be generated(%)')
    ax.set_ylabel('Time(ms)')
    ax.legend(handles=[dot1, dot2])
    # ax.set_xlim(60, 100)
    # ax2.set_xlim(40, 0)
    ax.set_ylim(-1, 14)
    ax2.set_ylim(-1, 14)

    fig.savefig('del_gen_particles.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()
