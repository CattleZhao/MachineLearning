# coding=utf-8
import pandas as pd

datafile = '../../chapter4/demo/data/discretization_data.xls'

data = pd.read_excel(datafile)
data = data[u'肝气郁结证型系数'].copy()
k = 4

d1 = pd.cut(data, k, labels=range(k))  # 等宽离散化

# 等频率离散化
w = [1.0 * i / k for i in range(k + 1)]
w = data.describe(percentiles=w)[4:4 + k + 1]  # 使用describe函数自动计算分位数
w[0] = w[0] * (1 - 1e-10)
d2 = pd.cut(data, w, labels=range(k))

from sklearn.cluster import KMeans  # 引入KMeans

kmodel = KMeans(n_clusters=k, n_jobs=1)
kmodel.fit(data.reshape((len(data), 1)))  # 训练模型
c = pd.DataFrame(kmodel.cluster_centers_).sort(0)  # 输出聚类中心，并且排序（默认是随机排序的）
w = pd.rolling_mean(c, 2).iloc[1:]  # 相邻两项求中点，作为边界点
w = [0] + list(w[0]) + [data.max()]  # 把首末边界点加上
d3 = pd.cut(data, w, labels=range(k))


def cluster_plot(d, k):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 3))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')

    plt.ylim(-0.5, k - 0.5)
    return plt


cluster_plot(d1, k).show()
cluster_plot(d2, k).show()
cluster_plot(d3, k).show()
