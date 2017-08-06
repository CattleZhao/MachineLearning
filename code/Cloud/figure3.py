# coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    x = np.linspace(0,1,10)
    y_optimize = np.array([0.9864999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y_charm = np.array(
        [0.9753264287206371, 0.9769592994829677, 0.995650471727615, 0.995650471727615, 0.9860305917420893,
         0.9887983750023119, 0.9803449558542459, 0.9806491749197788, 0.980954142590007, 0.9812598045525849])
    plt.figure(facecolor='w')
    plt.plot(x, y_optimize, 'r-', linewidth=2, label=u'本文得到的方案的QoS')
    plt.plot(x, y_charm, 'g-', linewidth=2, label=u'charm得到的方案的QoS')
    plt.xlabel("read frequency")
    plt.ylabel("quality of service")
    plt.yticks(np.arange(0.5,1.1,0.1))
    plt.title(u'对比', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
