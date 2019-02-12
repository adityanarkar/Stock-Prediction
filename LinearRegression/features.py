import numpy as np
import pandas as pd


def simpleMA(df, moving_avg_window, close):
    # TODO:row - moving_avg_window:row-1
    for row in range(moving_avg_window, len(df.index)):
        df.iloc[row, -1] = 1 if (df.iloc[row - moving_avg_window:row, close].mean()) < df.iloc[row, close] else -1


def weightedMA(df, moving_avg_window, close):
    for row in range(moving_avg_window, len(df.index)):
        tempDiv = 0
        total = 0
        for weight in range(moving_avg_window, 0, -1):
            temp = df.iloc[row - (weight), close]
            total = (((moving_avg_window + 1) - weight) * temp) + total
            tempDiv = tempDiv + weight
        df.iloc[row, -1] = 1 if df.iloc[row, close] > (total / tempDiv) else -1


def EMA(df, moving_avg_window, close):
    temp = 0
    columnName = str(moving_avg_window) + '-day-EMA'
    df[columnName] = np.nan
    for row in range(moving_avg_window, moving_avg_window + 1):
        temp = df.iloc[row - moving_avg_window:row - 1, close].mean()

    multiplier = (2 / (moving_avg_window + 1))
    for row in range(moving_avg_window, len(df.index)):
        if row == moving_avg_window:
            df.iloc[row, -1] = (df.iloc[row, close] - temp) * multiplier + temp
        else:
            temp = df.iloc[row - 1, -1]
            df.iloc[row, -1] = (df.iloc[row, close] - temp) * multiplier + temp


def discretizeMomentum(df: pd.DataFrame, row, prev):
    if prev != np.nan:
        if df.iloc[row, -1] < prev:
            prev = df.iloc[row, -1]
            df.iloc[row, -1] = -1
        else:
            prev = df.iloc[row, -1]
            df.iloc[row, -1] = 1
    return prev

def momentum(df, close):
    prev = np.nan
    for row in range(9, len(df.index)):
        df.iloc[row, -1] = df.iloc[row, close] - df.iloc[row - 9, close]
        prev = discretizeMomentum(df, row, prev)

def stochasticK(df, close, high, low, window):
    prev = np.nan
    for row in range(window, len(df.index)):
        currentClose = df.iloc[row, close]
        highestHigh = df.iloc[row - window:row, high].max()
        lowestLow = df.iloc[row - window:row, low].min()
        df.iloc[row, -1] = (currentClose - lowestLow) / (highestHigh - lowestLow)
        prev = discretizeMomentum(df, row, prev)


def stochasticD(df, K):
    prev = np.nan
    for row in range(2, len(df.index)):
        df.iloc[row, -1] = df.iloc[row - 2:row, K].mean()
        prev = discretizeMomentum(df, row, prev)


def RSI(df: pd.DataFrame, close):
    for row in range(15, len(df.index)):
        temp1 = df.iloc[row - 15:row - 2, close].reset_index(drop=True)
        temp2 = df.iloc[row - 14:row - 1, close].reset_index(drop=True)
        temp = temp1 - temp2
        AvgGain = temp[temp > 0].sum() / 14
        AvgLoss = -1 * (temp[temp < 0].sum()) / 14
        RS = AvgGain / AvgLoss
        df.iloc[row, -1] = 100 - (100 / (1 + RS))



def MACD(df: pd.DataFrame, close):
    EMA(df, 9, close)
    EMA(df, 12, close)
    EMA(df, 26, close)
    df.dropna(inplace=True)
    df['MACD'] = np.nan
    MACDLine = df['12-day-EMA'] - df['26-day-EMA']
    df['MACD'] = MACDLine - df['9-day-EMA']
    df['MACD'] = df['MACD'].diff()
    df.dropna(inplace=True)
    df['MACD'] = df['MACD'].apply(checkValue)


def checkValue(value):
    if value >= 0:
        return 1
    else:
        return -1
