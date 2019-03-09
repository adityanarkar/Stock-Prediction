import os

import matplotlib.pyplot as plt
import pandas as pd

import Regression as regression

LOCAL_FILE_PATH = os.path.join("datasets", "Titan", "TITAN.NS.CSV")
FILE_NAME = "Titan.csv"


def readData():
    df = pd.read_csv("./" + LOCAL_FILE_PATH)
    print(df.tail())
    return df


def visualize(df: pd.DataFrame):
    print(df.head())
    # plt.scatter(df["Adj Close"], df.label)
    plt.legend()
    plt.show()


df = readData()
df = regression.addFeatures(df)
# visualize(df)
df.drop(columns=['12-day-EMA', '26-day-EMA', '9-day-EMA', 'Open', 'High', 'Low', 'Date'], inplace=True)
# print(df.tail())
# print(df.corr())
regression.linearRegression(df)
