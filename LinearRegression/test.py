import os
import pandas as pd

import svm
import Regression as regression
import math
import numpy as np

STOCK = 'Titan'
LOCAL_FILE_PATH = os.path.join("datasets", STOCK, STOCK+".NS.CSV")


def readData():
    df = pd.read_csv("./" + LOCAL_FILE_PATH)
    return df


def getMinMax(df: pd.DataFrame):
    max = df['High'].max()
    min = df['Low'].min()
    return [min, max]

def writeIntoFile(output, file, mode):
    with open(file, mode) as file:
        file.write(output)

def createDataFrame():
    dfComplete = readData()
    dfComplete = regression.addFeatures(dfComplete)
    dfComplete.drop(columns=['12-day-EMA', '26-day-EMA', '9-day-EMA', 'Date'], inplace=True)

    return dfComplete

def testLinearRegression():

    file = STOCK+'LinearRegression-result.txt'

    dfComplete = createDataFrame()

    writeIntoFile(STOCK, file, 'w')

    for i in range(20, 100):
        df = dfComplete[:-i]
        dfRem = dfComplete[-i:(-i+10)]
        minMaxRange = getMinMax(dfRem)
        result = regression.linearRegression(df)
        # resultMinMax = [result.min(), result.max()]
        count = 0
        for res in result:
            if res >= minMaxRange[0] and res <= minMaxRange[1]:
                count += 1
        if count >= 5:
            writeIntoFile('\n1', file, 'a')
        else:
            writeIntoFile('\n0', file,  'a')

def testLogisticRegression():

    file = STOCK + 'LogisticRegression-result.txt'

    dfComplete = createDataFrame()

    writeIntoFile(STOCK, file, 'w')

    for i in range(20, 100):
        df = dfComplete[:-i]
        dfRem = dfComplete[-i:(-i+10)]

        minMaxRange = getMinMax(dfRem)
        # avg = dfRem['Adj Close'].mean()
        result = regression.logisticRegression(df)
        zeros = result.tolist().count(0)
        last = df.tail(1)['Adj Close']

        if zeros < 5:
            if (minMaxRange[1] > last).bool():
                writeIntoFile('\n1', file, 'a')
            else:
                writeIntoFile('\n0', file, 'a')
        else:
            if (minMaxRange[1] < last).bool():
                writeIntoFile('\n1', file, 'a')
            else:
                writeIntoFile('\n0', file, 'a')


def testSVM():

    file = STOCK + 'SVM-result.txt'

    dfComplete = createDataFrame()

    writeIntoFile(STOCK, file, 'w')

    for i in range(20, 100):
        df = dfComplete[:-i]
        dfRem = dfComplete[-i:(-i+10)]

        minMaxRange = getMinMax(dfRem)
        result = svm.fit(df)
        zeros = result.tolist().count(0)
        last = df.tail(1)['Adj Close']

        if zeros < 5:
            if (minMaxRange[1] > last).bool():
                writeIntoFile('\n1', file, 'a')
            else:
                writeIntoFile('\n0', file, 'a')
        else:
            if (minMaxRange[1] < last).bool():
                writeIntoFile('\n1', file, 'a')
            else:
                writeIntoFile('\n0', file, 'a')

def createLabel(x):
    # You can create a label based on
    # 1. Max (High price) for next 10 days
    # 2. Average of (High-low) for next 10 days

    if math.isnan(x['label']):
        return np.nan
    elif x['label'] > x['Adj Close']:
        return 1
    else:
        return 0

def generateLabel(df: pd.DataFrame, highIndex):
    df['label'] = np.nan
    for row in range(0, len(df.index) - 10):
        df.iloc[row, -1] = df.iloc[row + 1:row + 10 + 1, highIndex].max()
    return df


# testLinearRegression()
testLogisticRegression()
# testSVM()