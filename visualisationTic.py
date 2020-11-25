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

dataset_name= 'tic-tac-toeBinary'
split_name= '70b'

hidden_nodes = [30,16,2]

#the cut-off boundaries
min_x=0
max_x=500
min_y=0
max_y=3500

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
#   x =     terms
#   y =     rules
#   Farbe = max depth
#   Form =  NN/pol/BNN

#max depth
#c = 4 ; b = 5; g = 6; r = 7

#NN
plt.plot(37,210,'co')
plt.plot(229,1822,'bo')
plt.plot(max_x,max_y,'go')
plt.plot(max_x,max_y,'ro')

#pol
plt.plot(15,68,'c+')
plt.plot(34,184,'b+')
plt.plot(95,743,'g+')
plt.plot(97,768,'r+')

#pol2
plt.plot(6,23,'c+')
plt.plot(16,94,'b+')
plt.plot(19,116,'g+')
plt.plot(29,187,'r+')

#BNN
plt.plot(14,31,'cx')
plt.plot(195,1215,'bx')
plt.plot(250,1882,'gx')
plt.plot(406,3106,'rx')

#BNN2
plt.plot(18,120,'cx')
plt.plot(62,490,'bx')
plt.plot(153,1347,'gx')
plt.plot(406,3106,'rx')

#plt.xscale("log")
#plt.yscale("log")
plt.xlabel("#rules")
plt.ylabel("#terms")
plt.title("Tic-Tac-Toe")
plt.grid(True)
plt.show()
