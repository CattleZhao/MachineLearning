# coding=utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge

if __name__ == '__main__':
    data = pd.read_csv('Advertising.csv')
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # model = Ridge()
    model = Lasso()
    alpha_can = np.logspace(-3, 2, 10)  # base=10的等比数列，一共10个数
    np.set_printoptions(suppress=True)
    result_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    result_model.fit(x_train, y_train)
    print(u'超参数: \n', result_model.best_params_)

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order,:]
    y_hat = result_model.predict(x_test)
    print(result_model.score(x_test,y_test))
    mse = np.average((y_hat - y_test) ** 2)
    rmse = np.sqrt(mse)
    print('MSE = ', mse)
    print('RMSE = ', rmse)

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()