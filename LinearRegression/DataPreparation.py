import os
import pandas as pd
import matplotlib.pyplot as plt
import Regression as regression
import svm

STOCK = "Titan"

LOCAL_FILE_PATH = os.path.join("datasets", STOCK, STOCK+".NS.CSV")

def readData():
    df = pd.read_csv("./" + LOCAL_FILE_PATH)
    print(df.tail())
    return df


df = readData()
df = regression.addFeatures(df)
df.drop(columns=['12-day-EMA', '26-day-EMA', '9-day-EMA', 'Date'], inplace=True)
# df.drop(columns=['Date'], inplace=True)


# corr = df.corr()
# print(corr)
# corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# plt.matshow(corr)
# plt.show()

# regression.linearRegression(df)
regression.logisticRegression(df)
# svm.svr(df)
