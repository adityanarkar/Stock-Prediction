import os

import matplotlib.pyplot as plt
import pandas as pd

import Regression as regression

LOCAL_FILE_PATH = os.path.join("datasets", "TITAN", "TITAN.NS.CSV")
FILE_NAME = "Titan.csv"


def readData():
    df = pd.read_csv("./" + LOCAL_FILE_PATH)
    print(df.head())
    return df


def visualize(df: pd.DataFrame):
    print(df.head())
    plt.scatter(df.Date, df.Close)
    plt.legend()
    plt.show()


df = readData()
# visualize(df)
df = regression.addFeatures(df)
regression.linearRegression(df)
