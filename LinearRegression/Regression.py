import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

import features

forecast_days = 10
moving_avg_window = 10


def addFeatures(df: pd.DataFrame):
    # Selecting features that brings in value
    df = df[['Date', 'Open', 'Adj Close', 'High', 'Low']]
    openIndex = df.columns.get_loc('Open')
    closeIndex = df.columns.get_loc("Adj Close")
    highIndex = df.columns.get_loc("High")
    lowIndex = df.columns.get_loc("Low")

    df['HL_PCT'] = ((df['High'] - df['Low']) / df['Low']) * 100
    df['PCT_CHNG'] = ((df['Open'] - df['Adj Close']) / df['Adj Close']) * 100
    #
    df['Moving_Avg'] = np.nan
    features.simpleMA(df, moving_avg_window, closeIndex)
    #
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

    # print(df.tail(20))
    return df


def generateLabel(df: pd.DataFrame, highIndex):
    df['label'] = np.nan
    for row in range(0, len(df.index) - forecast_days):
        df.iloc[row, -1] = df.iloc[row + 1:row + forecast_days + 1, highIndex].max()
    return df

def linearRegression(df: pd.DataFrame):
    X_temp = np.array(df.drop(['label'], 1))
    print(df)

    X = X_temp[:-forecast_days]

    X_lately = X_temp[-forecast_days:]

    y = np.array(df['label'])[:-forecast_days]

    X_train, X_test, y_train, y_test = get_train_test_split(X, y, 0.2)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    linearReg = LinearRegression().fit(X_train, y_train)
    print("Linear regression = " + str(linearReg.score(X_test, y_test)))

    forecastLinearReg = linearReg.predict(X_lately)


    return forecastLinearReg

def logisticRegression(df: pd.DataFrame):
    df['label'] = df.apply(createLabel, axis=1)

    X_temp = np.array(df.drop(['label'], 1))
    X = X_temp[:-forecast_days]
    X_lately = X_temp[-forecast_days:]
    y = np.array(df['label'])[:-forecast_days]
    print(y)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, 0.2)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    logisticReg = LogisticRegression(solver='liblinear', C=0.1).fit(X_train, y_train)
    print("Logistic regression = " + str(logisticReg.score(X_test, y_test)))

    forecastLogisticReg = logisticReg.predict(X_lately)
    print("Logistic regression forecast = " + str(forecastLogisticReg))

    return forecastLogisticReg


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


def get_train_test_split(X, y, size):
    X_train = X[:-math.ceil(len(X) * size)]
    X_test = X[-math.ceil(len(X) * size):]
    y_train = y[:-math.ceil(len(y) * size)]
    y_test = y[-math.ceil(len(y) * size):]

    return X_train, X_test, y_train, y_test


def plotResults(X_train, y_train, X_test, logisticReg, linearReg):
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X_train[:, 0], y_train[:], color='black', zorder=20)

    # Plotting logistic regression
    plt.plot(X_test, X_test * logisticReg.coef_ + logisticReg.intercept_, color='red', linewidth=3)

    # Preparing Linear Regression
    plt.plot(X_test, linearReg.coef_ * X_test + linearReg.intercept_, linewidth=1)

    plt.ylabel('y')
    plt.xlabel('X')
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
               loc="lower right", fontsize='small')
    plt.tight_layout()

    plt.show()
