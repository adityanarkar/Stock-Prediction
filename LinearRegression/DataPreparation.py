import os

import pandas as pd

import Regression as regression
import svm

LOCAL_FILE_PATH = os.path.join("datasets", "M&MFIN", "M&MFIN.NS.CSV")

def readData():
    df = pd.read_csv("./" + LOCAL_FILE_PATH)
    print(df.tail())
    return df


df = readData()
df = regression.addFeatures(df)
df.drop(columns=['12-day-EMA', '26-day-EMA', '9-day-EMA', 'Date'], inplace=True)

regression.linearRegression(df)
svm.fit(df)
