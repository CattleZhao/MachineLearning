# coding=utf-8
import xgboost as xgb
import numpy as np


def g_h(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1 - p)
    return g, h


if __name__ == '__main__':
    data_train = xgb.DMatrix('agaricus_train.txt')
    data_test = xgb.DMatrix('agaricus_test.txt')

    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 7
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=g_h)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('样本总数: \t', len(y_hat))
    print('错误数目: \t', error)
    print('错误率: \t', error_rate)
