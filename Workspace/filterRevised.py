import pandas as pd
import sklearn.model_selection as skl
import random
import time

def main():
    startTime = time.time()
    testFilter()
    totalTime = time.time()-startTime
    print(f"Runtime: {totalTime}")


def testFilter():

    dataFile = "smartphone_activity.csv"
    reductionValue = 500
    acceptedNumPoints = 100
    increaseVal = 1.1

    proposedFilter(reductionValue,acceptedNumPoints,dataFile,increaseVal)

def proposedFilter(reductionValue,acceptedNumPoints,dataFile,increaseVal):
    acceptedPointCounter = 0

    trainData = loadData(dataFile)

    minValues = [trainData.iat[0,i] for i in range(len(trainData.columns))]
    maxValues = [trainData.iat[0,i] for i in range(len(trainData.columns))]
    rangesReduced = [[trainData.iat[0,i] for i in range(len(trainData.columns))]for j in range(2)]

    #Initializing Min and Max Values
    for i in range(len(trainData.index)):
        for j in range(len(trainData.columns)):
            if trainData.iat[i,j] < minValues[j]:
                minValues[j] = trainData.iat[i,j]
            if trainData.iat[i,j] > maxValues[j]:
                maxValues[j] = trainData.iat[i,j]

    for i in range(len(trainData.columns)):
        rangesReduced[0][i] = int(minValues[i]-(minValues[i]/reductionValue))
        rangesReduced[1][i] = int(maxValues[i] + (maxValues[i]/reductionValue))

    newData,acceptedPointCounter = checkVals(trainData,rangesReduced)

    while acceptedPointCounter<acceptedNumPoints:
        rangesReduced[i][0] = int(rangesReduced[i][0]-(rangesReduced[i][0]*increaseVal))
        rangesReduced[i][1] = int(rangesReduced[i][1]+(rangesReduced[i][1]*increaseVal))

        newData,acceptedPointCounter = checkVals(trainData,rangesReduced)
    
    


    print(f"Accepted Point Counter: {acceptedPointCounter}\n")
    with open("smartphoneList.txt","w+") as myFile:
        for i in range(len(newData)):
            myFile.write(str(newData[i]).strip("[]") + "\n")



def loadData(dataFile):
    #Reading in Data
    data = pd.read_csv(dataFile,header=None, sep=",")
    df = pd.DataFrame(data)
    

    trainData,testData,trainClass,testClass = skl.train_test_split(df,df[561],train_size=0.7,test_size=0.3)
    
    return trainData


def checkVals(trainData,rangesReduced):
    acceptedPointCounter = 0
    newData = [[0 for i in range(len(trainData.columns))]for j in range(len(trainData.index))]
    for i in range(len(trainData.index)):
        for j in range(0,len(trainData.columns)):
            if trainData.iat[i,j] > rangesReduced[0][j] and trainData.iat[i,j] < rangesReduced[1][j]:
                newData[i][j] = trainData.iat[i,j]
                acceptedPointCounter = acceptedPointCounter + 1
    
    return newData,acceptedPointCounter 


if __name__=="__main__":
    main()


