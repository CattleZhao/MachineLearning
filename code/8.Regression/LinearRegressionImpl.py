# coding=utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def regression(d_x, d_y, alpha):  # d代表数据，alpha代表学习率
    n = len(d_x)
    theta = np.zeros(3)
    for times in range(time_cal):
        for i in range(n):
            g = np.dot(theta, d_x[i]) - d_y[i]
            theta = theta - alpha * g * d_x[i]
        print(times, theta)
        lostFunction.append(fw(theta, d_x, d_y))
    return theta


def r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum()

    return (1 - (numerator / denominator))


def fw(theta, d_x, d_y):
    num = 0
    for i in range(len(d_x)):
        num += (np.dot(theta, d_x[i]) - d_y[i]) ** 2
    return num/2


if __name__ == '__main__':
    data = pd.read_csv('Advertising.csv', header=0)
    data['intercept'] = 1
    x = data[['TV', 'Radio', 'intercept']]
    y = data['Sales']
    lostFunction = []
    # print(x.ix[0])
    # print(np.dot([1, 1, 1], x.ix[0]))

    alpha = 0.00001  # 学习率
    time_cal = 2000 #迭代次数
    # print(x)
    # print(y)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]  # 将y_test的数据按照从小到大排序
    x_test = x_test.values[order, :]  # 同理
    theta = regression(x_train.values, y_train.values, alpha)

    # theta = [0.04619357, 0.18170718, 2.9082168]
    y_hat = []
    for i in range(len(x_test)):
        y_hat.append(np.dot(theta, x_test[i]))

    mse = np.average((y_hat - y_test) ** 2)
    rmse = np.sqrt(mse)
    print('MSE = ', mse)
    print('RMSE = ', rmse)

    print('R2 = ', r2_score(y_test, y_hat))

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid()
    plt.show()

    #绘制损失函数
    plt.figure(facecolor='w')
    t = np.arange(time_cal)
    print(len(t))
    plt.plot(t, lostFunction, 'r-', linewidth=2)
    plt.title(u'损失函数', fontsize=18)
    plt.grid()
    plt.show()
