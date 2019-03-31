import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import Regression as regression

forecast_days = 10

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
])


def fit(df: pd.DataFrame):
    print(df.tail(20))

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
