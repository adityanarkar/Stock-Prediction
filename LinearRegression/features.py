def simpleMA(df, moving_avg_window):
    column = 1

    for row in range(moving_avg_window, len(df.index)):
        df.iloc[row, -1] = (df.iloc[row - moving_avg_window:row, column].mean())
