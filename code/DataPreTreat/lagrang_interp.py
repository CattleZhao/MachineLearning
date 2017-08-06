# coding=utf-8
# 拉格朗日插值法实现
import pandas as pd
from scipy.interpolate import lagrange

inputfile = '../../chapter4/demo/data/catering_sale.xls'
outputfile = 'sales.xls'

data = pd.read_excel(inputfile)
data[u'销量'][(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None


# 自定义列向量差值函数
def ployinterp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)


print(data.columns)
for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:  # i在这是一维（列）,data[i]等于把这一列的值取出来了，
            # j在这里是二维（行），i相当于XY轴的X轴，j相当于XY轴的Y轴,data[i].isnull是判断整列的某一个值是nan
            # 那么就返回True,data[i].isnull[j]是逐行逐行的判断是否为nan了，是的话返回True从而进行下面代码块的处理。
            data[i][j] = ployinterp_column(data[i], j)

data.to_excel(outputfile)
