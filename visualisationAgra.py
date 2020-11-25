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

dataset_name= 'agaricusBinary'
split_name= '70b701'
model_name= 'nn,4,3,2hidden,tanh,Q1_500,7010NN'
BNNmodel_name = 'nn,4,3,2hidden,tanh,Q1_500,7010BNN'

hidden_nodes=[4,3,2]

ruleX = -0.0486436732
ruleY = -0.0276950941

BNNruleX = 0.1100622013
BNNruleY = -0.1253643645

#the cut-off boundaries
min_x=0
max_x=7
min_y=0
max_y=5000

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

plt.axis( [min_x,max_x,min_y,max_y] )
#fig, (ax1, ax2) = plt.subplots(1,2)

##################################################Plot NN#########################################################
#    
#   b 	blue            o   Punkt
#   c 	cyan            s   REchteck
#   g 	green           D   Diamant
#   m 	magenta         ^   Dreieck
#   r 	red             x   x
#   y 	yellow          +   +
#   k 	black
#   w 	white
#

#Legende:
#   x = terms
#   y = rules
#   Form Pruning
#   Farbe CD
plt.plot(x,y,'bo')
#plt.plot(,,'bo')


plt.show()
