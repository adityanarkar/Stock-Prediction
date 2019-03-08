import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import features

forecast_days = 6
moving_avg_window = 10


def addFeatures(df: pd.DataFrame):
    # Selecting features that brings in value
    df = df[['Open', 'Adj Close', 'High', 'Low']]
    openIndex = df.columns.get_loc('Open')
    closeIndex = df.columns.get_loc("Adj Close")
    highIndex = df.columns.get_loc("High")
    lowIndex = df.columns.get_loc("Low")

    df['HL_PCT'] = ((df['High'] - df['Low']) / df['Low']) * 100
    df['PCT_CHNG'] = ((df['Open'] - df['Adj Close']) / df['Adj Close']) * 100

    df['Moving_Avg'] = np.nan
    features.simpleMA(df, moving_avg_window, closeIndex)

    df['Weighted_MA'] = np.nan
    features.weightedMA(df, moving_avg_window, closeIndex)
    df.dropna(inplace=True)

    df['Momentum'] = np.nan
    features.momentum(df, closeIndex)
    df.dropna(inplace=True)

    df['%K'] = np.nan
    features.stochasticK(df, closeIndex, highIndex, lowIndex, 10)
    df.dropna(inplace=True)

    df['%d'] = np.nan
    features.stochasticD(df, df.columns.get_loc('%K'))
    df.dropna(inplace=True)
    # print(df.tail())

    df['RSI'] = np.nan
    features.RSI(df, closeIndex)
    df.dropna(inplace=True)

    features.MACD(df, closeIndex)
    df.dropna(inplace=True)

    df['label'] = df['Adj Close'].shift(-forecast_days)
    df['label'] = df.apply(createLabel, axis=1)

    return df


def linearRegression(df: pd.DataFrame):

    X_temp = np.array(df.drop(['label'], 1))

    X = X_temp[:-forecast_days]

    X_lately = X_temp[-forecast_days:]

    y = np.array(df['label'])[:-forecast_days]

    X_train, X_test, y_train, y_test = get_train_test_split(X, y, 0.8)

    print(len(X_test))
    print(len(X_train))
    print(len(y_test))
    print(len(y_train))
    #
    linearReg = LinearRegression().fit(X_train, y_train)
    print("Linear regression = " + str(linearReg.score(X_test, y_test)))
    #
    forecastLinearReg = linearReg.predict(X_lately)
    print("Linear regression forecast = " + str(forecastLinearReg))

    logisticReg = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, y_train)
    print("Logistic regression = " + str(logisticReg.score(X_test, y_test)))
    #
    forecastLogisticReg = logisticReg.predict(X_lately)
    print("Logistic regression forecast = " + str(forecastLogisticReg))


def createLabel(x):
    if math.isnan(x['label']):
        return np.nan
    elif x['label'] > x['Adj Close']:
        return 1
    else:
        return -1


def get_train_test_split(X, y, size):
    X_train = X[:-math.ceil(len(X) * size)]
    X_test = X[-math.ceil(len(X) * size):]
    y_train = y[:-math.ceil(len(y) * size)]
    y_test = y[-math.ceil(len(y) * size):]
    return X_train, X_test, y_train, y_test
