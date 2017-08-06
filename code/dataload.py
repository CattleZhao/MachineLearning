# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series, DataFrame

df = pd.read_csv('../data/adult.csv');

result = pd.read_csv('../data/adult.csv', nrows = 10);

print(result);
