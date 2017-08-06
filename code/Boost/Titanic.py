# coding=utf-8
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler


def load_data(file_name, is_train):
    data = pd.read_csv(file_name)
    pd.set_option('display.width', 200)
    # print(data.describe())

    data['Sex'] = pd.Categorical(data['Sex']).codes

    # 补齐船票价格缺失值
    if len(data.Fare[data.Fare == 0]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data.Fare[data.Pclass == (f + 1)].dropna().median()
        for f in range(0, 3):
            data.loc[(data.Fare == 0) & (data.Pclass == (f + 1)), 'Fare'] = fare[f]

    # 年龄：使用均值代替缺失值
    # data.loc[(data.Age.isnull()),'Age'] = data.Age.dropna().mean()
    if is_train:
        # 年龄：使用随机森林预测缺失值
        print(u'随机森林预测缺失年龄：--start--')
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[data.Age.notnull()]
        age_null = data_for_age.loc[data.Age.isnull()]
        x = age_exist.iloc[:, 1:]
        y = age_exist.iloc[:, 0]
        rfr = RandomForestRegressor(n_estimators=20)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.iloc[:, 1:])
        # x = age_exist.values[:, 1:]
        # y = age_exist.values[:, 0]
        # rfr = RandomForestRegressor(n_estimators=20)
        # rfr.fit(x, y)
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print(u'随机森林预测缺失年龄：--over--')
    else:
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[data.Age.notnull()]
        age_null = data_for_age.loc[data.Age.isnull()]
        x = age_exist.iloc[:, 1:]
        y = age_exist.iloc[:, 0]
        rfr = RandomForestRegressor(n_estimators=20)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.iloc[:, 1:])
        # x = age_exist.values[:, 1:]
        # y = age_exist.values[:, 0]
        # rfr = RandomForestRegressor(n_estimators=20)
        # rfr.fit(x, y)
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
    data['Age'] = pd.cut(data['Age'], bins=6, labels=np.arange(6))

    # 起始城市
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'
    embarked_data = pd.get_dummies(data.Embarked)
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)
    # data.to_csv('New_Date.csv')

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    y = None
    if 'Survived' in data:
        y = data['Survived']

    x = np.array(x)
    y = np.array(y)
    x = np.tile(x, (5, 1))
    y = np.tile(y, (5,))
    if is_train:
        return x, y
    return x, data['PassengerId']


if __name__ == '__main__':
    x, y = load_data('Titanic.train.csv', True)
    # x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    y_hat = lr.predict(x_test)
    lr_acc = accuracy_score(y_test, y_hat)

    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_hat = rfc.predict(x_test)
    rfc_acc = accuracy_score(y_test, y_hat)

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=20, evals=watch_list)
    y_hat = bst.predict(data_test)
    print(bst)
    y_hat[y_hat > 0.5] = 1
    y_hat[~(y_hat > 0.5)] = 0
    xgb_acc = accuracy_score(y_test, y_hat)

    print('Logistic回归：%.3f%%' % (100 * lr_acc))
    print('随机森林：%.3f%%' % (100 * rfc_acc))
    print('XGBoost：%.3f%%' % (100 * xgb_acc))
