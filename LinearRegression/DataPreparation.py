import os

import pandas as pd

import Regression as regression

LOCAL_FILE_PATH = os.path.join("datasets", "Titan", "TITAN.NS.CSV")
FILE_NAME = "Titan.csv"


def readData():
    df = pd.read_csv("./" + LOCAL_FILE_PATH)
    print(df.tail())
    return df


df = readData()
df = regression.addFeatures(df)
df.drop(columns=['12-day-EMA', '26-day-EMA', '9-day-EMA', 'Open', 'High', 'Low', 'Date'], inplace=True)

regression.linearRegression(df)
