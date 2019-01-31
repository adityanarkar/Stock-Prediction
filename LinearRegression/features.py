import pandas as pd
def simpleMA(df, moving_avg_window, close):
    for row in range(moving_avg_window, len(df.index)):
        df.iloc[row, -1] = (df.iloc[row - moving_avg_window:row, close].mean())


def weightedMA(df, moving_avg_window, close):
    for row in range(moving_avg_window, len(df.index)):
        tempDiv = 0
        total = 0
        for weight in range(moving_avg_window, 0, -1):
            temp = df.iloc[row - (weight), close]
            total = (((moving_avg_window + 1) - weight) * temp) + total
            tempDiv = tempDiv + weight
        df.iloc[row, -1] = total / tempDiv


def momentum(df, close):
    for row in range(9, len(df.index)):
        df.iloc[row, -1] = df.iloc[row, close] - df.iloc[row - 9, close]


def stochasticK(df, close, high, low, window):
    for row in range(window, len(df.index)):
        currentClose = df.iloc[row, close]
        highestHigh = df.iloc[row - window:row, high].max()
        lowestLow = df.iloc[row - window:row, low].min()

        df.iloc[row, -1] = (currentClose - lowestLow) / (highestHigh - lowestLow)


def stochasticD(df, K):
    for row in range(2, len(df.index)):
        df.iloc[row, -1] = df.iloc[row - 2:row, K].mean()


def RSI(df: pd.DataFrame, close):
    for row in range(15, len(df.index)):
        temp1 = df.iloc[row - 15:row - 2, close].reset_index(drop=True)
        temp2 = df.iloc[row - 14:row - 1, close].reset_index(drop=True)
        temp = temp1 - temp2
        AvgGain = temp[temp > 0].sum() / 14
        AvgLoss = -1 * (temp[temp < 0].sum()) / 14
        RS = AvgGain / AvgLoss
        df.iloc[row, -1] = 100 - (100 / (1 + RS))
