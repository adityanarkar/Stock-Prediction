import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVR
import math

import Regression as regression

forecast_days = 10

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
])


def fit(df: pd.DataFrame):

    df['label'] = df.apply(createLabel, axis=1)

    X_temp = np.array(df.drop(['label'], 1))

    X = X_temp[:-forecast_days]

    X_lately = X_temp[-forecast_days:]

    y = np.array(df['label'])[:-forecast_days]

    X_train, X_test, y_train, y_test = regression.get_train_test_split(X, y, 0.2)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge"))
    ])

    svm_clf.fit(X_train, y_train)
    print("Svm : " + str(svm_clf.predict(X_lately)))
    print("Score: " + str(svm_clf.score(X_test, y_test)))
    return svm_clf.predict(X_lately)


def createLabel(x):
    # You can create a label based on
    # 1. Max (High price) for next 10 days
    # 2. Average of (High-low) for next 10 days

    if math.isnan(x['label']):
        return np.nan
    elif x['label'] > x['Adj Close']:
        return 1
    else:
        return -1
