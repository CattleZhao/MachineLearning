# -*- coding:utf-8 -*-
import numpy as np
from time import time
import math


def is_prime(x):
    return 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)]


if __name__ == '__main__':
    a = 2
    b = 100000

    t = time()
    p = [p for p in range(a, b) if 0 not in [p % d for d in range(2, int(math.sqrt(p)) + 1)]]
    print(time() - t)
    # print(p)

    t = time()
    p = list(filter(is_prime, range(a, b)))
    print(time() - t)
    # print(p)

    # 方法3：利用filter lambda
    t = time()
    is_prime2 = (lambda x: 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)])
    p = list(filter(is_prime2, range(a, b)))
    print(time() - t)
    # print(p)

    # 定义
    t = time()
    p_list = []
    for i in range(2, b):
        flag = True
        for p in p_list:
            if p > math.sqrt(i):
                break
            if i % p == 0:
                flag = False
                break
        if flag:
            p_list.append(i)
    print(time() - t)
    print(p_list)
