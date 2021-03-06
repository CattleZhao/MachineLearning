# coding=utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    data = pd.read_csv('Advertising.csv', header=0)
    x = data[['TV', 'Radio']]
    y = data['Sales']
    # print(x)
    # print(y)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # # 绘制1
    # plt.figure(facecolor='w')
    # plt.plot(data['TV'], y, 'ro', label='TV')
    # plt.plot(data['Radio'], y, 'g^', label='Radio')
    # plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    # plt.legend(loc='lower right')
    # plt.xlabel(u'广告费', fontsize=16)
    # plt.xlabel(u'销售额', fontsize=16)
    # plt.title(u'广告花费与销售额对比数据', fontsize=16)
    # plt.grid()
    # plt.show()
    #
    # # 绘制2
    # plt.figure(facecolor='w', figsize=(9, 10))
    # plt.subplot(311)
    # plt.plot(data['TV'], y, 'ro')
    # plt.title('TV')
    # plt.grid()
    # plt.subplot(312)
    # plt.plot(data['Radio'], y, 'g^')
    # plt.title('Radio')
    # plt.grid()
    # plt.subplot(313)
    # plt.plot(data['Newspaper'], y, 'b*')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print(linreg.coef_, linreg.intercept_)

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]  # 将y_test的数据按照从小到大排序
    x_test = x_test.values[order, :]  # 同理
    y_hat = linreg.predict(x_test)
    mse = np.average((y_hat - y_test) ** 2)
    rmse = np.sqrt(mse)
    print('MSE = ', mse)
    print('RMSE = ', rmse)
    print('R2 = ', linreg.score(x_train, y_train))
    print('R2 = ', linreg.score(x_test, y_test))

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量',fontsize = 18)
    plt.grid()
    plt.show()
