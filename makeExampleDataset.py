import random
import numpy as np
import csv

inst = 300
inputs = 2
outputs = 1
fileName='Q1_300'

data = [0] * inst
for  i in range(inst):
        data[i] = [0] * (inputs + outputs)

def fillData(dataList):
        print("Filling Data...")
        for i in range(len(dataList)):
                dataList[i][0]=random.uniform(-1.0,1.0)                                 #Random between -1 and 1
                dataList[i][1]=random.uniform(-1.0,1.0)                                #Random between -1 and 1
                if dataList[i][0] >= 0 and dataList[i][1] >= 0: dataList[i][2]=1        #Rule, when Instance is 1
        print("Done")

def writeData(dataList):
        print('Writing Data in File: data/'+fileName+'.csv')        
        with open('data/'+fileName+'.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(dataList)
        print("Done")

def getDatalist(name):
        with open('data/'+name+'.csv', newline='') as f:
                reader = csv.reader(f)
                data_list = list(reader)
        return data_list


fillData(data)
writeData(data)
