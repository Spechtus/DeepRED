import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import sys
import os
import scipy.stats
from obj_data_set import DataSet
import load_restore as lr
import evaluation_formulas as ef
import deep_nn_train as dnnt
import BNNloader as bl

dataset_name= 'Q1_500'
split_name= '70b709'

BNNmodel_name = 'nn,4,3,2hidden,tanh,Q1_500,709BNN'

hidden_nodes=[4,3,2]

BNN1ruleX = 0.1586238139
BNN1ruleY = 0.1304165404

BNN2ruleX = 0.1374163373
BNN2ruleY = 0.1304165404

BNN3ruleX = 0.1374163373
BNN3ruleY = 0.1304165404

BNN4ruleX = 0.1374163373
BNN4ruleY = 0.1304165404

BNN5ruleX = 0.1321407816
BNN5ruleY = 0.1304165404

BNN6ruleX = 0.1374163373
BNN6ruleY = 0.11464800003

BNN7ruleX = 0.1374163373
BNN7ruleY = 0.1304165404

#the cut-off boundaries
min_x=-1
max_x=1
min_y=-1
max_y=1

resolution_grid=1000

#DataSet with wich the models were trained on
data = DataSet(dataset_name, hidden_nodes)
train, test = lr.load_indexes(dataset_name, split_name)
vali = lr.load_vali_indexes(dataset_name, split_name)
data.set_split(train, vali, test)

def applyNet(x,y,model_name):
    act_train, _, _, _, _, _ = dnnt.execute_network_plot(x, y, model_name, hidden_nodes)
    zz= np.zeros((len(act_train)))
    c = 0
    for i in act_train:
        #print(np.amax(i))
        temp = np.where(i == (np.amax(i)))
        #print(temp[0])
        zz[c]=temp[0][0]
        c += 1
    return zz 

def applyBinaryNet(x,y,model_name):
    act_train= bl.execute_BNN_plot(x,y,model_name)
    zz= np.zeros((len(act_train)))
    c = 0
    for i in act_train:
        #print(np.amax(i))
        temp = np.where(i == (np.amax(i)))
        #print(temp[0])
        zz[c]=temp[0][0]
        c += 1
    return zz 

def applyRules(x,y,model_name,rx, ry):
    zz=np.arange(len(x[0])*len(x[0])).reshape(len(x[0]),len(x[0]))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i,j] > rx and y[i,j] > ry: zz[i,j] = 1 
            #elif x[i,j] < 0 and y[i,j] < 0: zz[i,j] = 1
            else: zz[i,j] = 0
    return zz

def applyBinaryRules(x,y,model_name,rx, ry):
    zz=np.arange(len(x[0])*len(x[0])).reshape(len(x[0]),len(x[0]))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i,j] > rx and y[i,j] > ry: zz[i,j] = 1 
            #elif x[i,j] < 0 and y[i,j] < 0: zz[i,j] = 1
            else: zz[i,j] = 0
    return zz

def applyFunction(x,y,model_name):
    zz=np.arange(len(x[0])*len(x[0])).reshape(len(x[0]),len(x[0]))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i,j] > 0 and y[i,j] > 0: zz[i,j] = 1 
            #elif x[i,j] < 0 and y[i,j] < 0: zz[i,j] = 1
            else: zz[i,j] = 0
    return zz

################################################## Plot BNN #########################################################

#training data
x, y = data.get_train_x_y()
x_train, y_train = [],[]
for i in range(len(x)):
	x_train.append(x[i][0])
	y_train.append(x[i][1])

#plot training data; pos = red; neg = blue
#for i in range(len(x_train)):
#	if x_train[i] > 0 and y_train[i] > 0:
#		plt.plot(x_train[i],y_train[i],'go')
#	else: 
#		plt.plot(x_train[i],y_train[i],'ko')

#plotting 
grid = np.linspace(min_x, max_x, 1000)
xx,yy = np.meshgrid(grid,grid)

#Funktion anwenden(Grau)
zz= applyFunction(xx,yy,BNNmodel_name) 

#zeichnen
plt.contour(grid,grid,zz,colors='grey',levels=[0,1],linestyles='dotted' )

#generate data
grid = np.linspace(min_x, max_x, resolution_grid)
xx,yy = np.meshgrid(grid,grid)

xx= xx.reshape(-1)
yy= yy.reshape(-1)

#predictions Net(Rot)
zz= applyBinaryNet(xx,yy,BNNmodel_name)

#now, plot the predictions
print(xx.shape)

xx= xx.reshape(resolution_grid,resolution_grid)
yy= yy.reshape(resolution_grid,resolution_grid)
zz= zz.reshape(resolution_grid,resolution_grid)

plt.contour(xx,yy,zz,colors='red',levels=[0,1],linestyles='solid',linewidths=2.5)

#predictions Rules(Blau)
grid = np.linspace(min_x, max_x, resolution_grid)
xx,yy = np.meshgrid(grid,grid)

zz= applyBinaryRules(xx,yy,BNNmodel_name,BNN1ruleX,BNN1ruleY)

#now, plot the rules
print(xx.shape)

plt.contour(xx,yy,zz,colors='orange',levels=[0,1],linestyles='solid',linewidths=2.5)

zz= applyBinaryRules(xx,yy,BNNmodel_name,BNN2ruleX,BNN2ruleY)

#now, plot the rules
print(xx.shape)

plt.contour(xx,yy,zz,colors='blue',levels=[0,1],linestyles='solid',linewidths=2.5)

zz= applyBinaryRules(xx,yy,BNNmodel_name,BNN5ruleX,BNN5ruleY)

#now, plot the rules
print(xx.shape)

plt.contour(xx,yy,zz,colors='brown',levels=[0,1],linestyles='solid',linewidths=2.5)

zz= applyBinaryRules(xx,yy,BNNmodel_name,BNN6ruleX,BNN6ruleY)

#now, plot the rules
print(xx.shape)

plt.contour(xx,yy,zz,colors='cyan',levels=[0,1],linestyles='solid',linewidths=2.5)


plt.title("BNN")
plt.grid(True)
plt.show()
