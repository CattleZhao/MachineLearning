# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

if __name__ == "__main__":
    # a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
    # print(a)
    # L = [1,2,3,4]
    # a = np.array(L)
    # print(a.dtype)


    # a = np.arange(1, 10, 0.5)
    # print(a)
    #
    # b = np.linspace(1, 10, 10)
    # print(b)
    #
    # c = np.logspace(1, 2, 9, endpoint=True)
    # print(c)
    # t = 1000
    # a = np.zeros(10000)
    # for i in range(t):
    #     a += np.random.uniform(-5, 5, 10000)
    # a /= t
    # plt.hist(a, bins=30, color='g', alpha=0.5, normed=True, label='sa')
    # plt.legend(loc='upper left')
    # plt.grid()
    # plt.show()

    lamda = 10
    p = stats.poisson(lamda)
    y = p.rvs(size=1000)
    print(y)
    mx = 30
    r = (0, mx)
    bins = r[1] - r[0]
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.subplot(121)
    plt.hist(y, bins=bins, range=r, color='g', alpha=0.8, normed=True)
    t = np.arange(0, mx + 1)
    plt.plot(t, p.pmf(t), 'ro-', lw=2)
    plt.grid()
    plt.show()
