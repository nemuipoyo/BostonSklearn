#!/usr/bin/env python 
# -*- coding: utf-8 -

from sklearn.datasets import load_boston
from sklearn import cross_validation
from sklearn import linear_model

boston = load_boston()

# 学習用/テスト用に分割
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    boston.data, boston.target, test_size=0.3, random_state=0)

# 学習
train = linear_model.LinearRegression()
train.fit(X_train, y_train)

# クロスバリデーション
print train.score(X_test, y_test)
