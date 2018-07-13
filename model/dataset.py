#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

URL_BASE = "http://kdd.ics.uci.edu/databases/kddcup99"
KDD_10_PERCENT_URL = URL_BASE + '/' + 'kddcup.data_10_percent.gz'
KDD_COLNAMES_URL = URL_BASE + '/' + 'kddcup.names'


def gen_train_valid_data():
    df_colnames = pd.read_csv(KDD_COLNAMES_URL, skiprows=1, sep=':', names=['f_names', 'f_types'])
    df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']

    df = pd.read_csv(KDD_10_PERCENT_URL, header=None, names=df_colnames['f_names'].values)

    df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]

    # one-hot encoding
    X = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['f_names'][:-1])  # except status

    # normal: 1, abnormal: 0
    y = np.where(df['status'] == 'normal.', 1, 0)

    # generate train/test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

    # only train with abnormal data
    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]

    return X_train, X_test, y_train, y_test
