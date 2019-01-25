def simpleMA(df, moving_avg_window):
    column = 1

    for row in range(moving_avg_window, len(df.index)):
        df.iloc[row, -1] = (df.iloc[row - moving_avg_window:row, column].mean())


def weightedMA(df, moving_avg_window):
    column = 1

    for row in range(moving_avg_window, len(df.index)):
        tempDiv = 0
        total = 0
        for weight in range(moving_avg_window, 0, -1):
            temp = df.iloc[row - (weight), column]
            total = (((moving_avg_window + 1) - weight) * temp) + total
            tempDiv = tempDiv + weight
        df.iloc[row, -1] = total / tempDiv
