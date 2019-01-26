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
