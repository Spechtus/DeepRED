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

dataset_name= 'Q13_2000'
split_name= '70b701'
model_name= 'nn,4,3,2hidden,tanh,Q13_2000,701NN'
BNNmodel_name = 'nn,4,3,2hidden,tanh,Q13_2000,701NNBNN'

hidden_nodes=[4,3,2]

ruleBiggerX = 0.06750518574399			
ruleSmallerX = 0.06750518574399
ruleBiggerY = -0.18192100880029
ruleSmallerY = 0.0571396272735

BNNruleBiggerX = -0.0312995708362			
BNNruleSmallerX = -0.02206609278794
BNNruleBiggerY = 0.20040251078254
BNNruleSmallerY = -0.06790830722725

#the cut-off boundaries
min_x=-1
max_x=1
min_y=-1
max_y=1

seed=1

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

def applyRules(x,y,model_name,ax, bx,ay,by):
    zz=np.arange(len(x[0])*len(x[0])).reshape(len(x[0]),len(x[0]))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i,j] > ax and y[i,j] > ay: zz[i,j] = 1 
            elif x[i,j] < bx and y[i,j] < by: zz[i,j] = 1
            else: zz[i,j] = 0
    return zz

def applyBinaryRules(x,y,model_name,ax, bx,ay,by):
    zz=np.arange(len(x[0])*len(x[0])).reshape(len(x[0]),len(x[0]))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i,j] > ax and y[i,j] > ay: zz[i,j] = 1 
            elif x[i,j] < bx and y[i,j] < by: zz[i,j] = 1
            else: zz[i,j] = 0
    return zz

def applyFunction(x,y,model_name):
    zz=np.arange(len(x[0])*len(x[0])).reshape(len(x[0]),len(x[0]))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i,j] > 0 and y[i,j] > 0: zz[i,j] = 1 
            elif x[i,j] < 0 and y[i,j] < 0: zz[i,j] = 1
            else: zz[i,j] = 0
    return zz

fig, (ax1, ax2) = plt.subplots(1,2)

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

#training data
x, y = data.get_train_x_y()
x_train, y_train = [],[]
for i in range(len(x)):
	x_train.append(x[i][0])
	y_train.append(x[i][1])

#plot training data; pos = red; neg = blue
for i in range(len(x_train)):
	if x_train[i] > 0 and y_train[i] > 0:
		ax1.plot(x_train[i],y_train[i],'go')
	elif x_train[i] < 0 and y_train[i] < 0:
		ax1.plot(x_train[i],y_train[i],'go')
	else: 
		ax1.plot(x_train[i],y_train[i],'ko')



#plotting 
grid = np.linspace(min_x, max_x, 1000)
xx,yy = np.meshgrid(grid,grid)

#Funktion anwenden(Grau)
zz= applyFunction(xx,yy,model_name) 

#zeichnen
ax1.contour(grid,grid,zz,colors='grey',levels=[0,1],linestyles='dotted' )

#generate grid data set
grid = np.linspace(min_x, max_x, resolution_grid)
xx,yy = np.meshgrid(grid,grid)

xx= xx.reshape(-1)
yy= yy.reshape(-1)

#predictions Net(Rot)
zz= applyNet(xx,yy,model_name)

#now, plot the predictions
print(xx.shape)

xx= xx.reshape(resolution_grid,resolution_grid)
yy= yy.reshape(resolution_grid,resolution_grid)
zz= zz.reshape(resolution_grid,resolution_grid)

ax1.contour(xx,yy,zz,colors='red',levels=[0,1],linestyles='solid',linewidths=3)


#predictions Rules(Blau)
grid = np.linspace(min_x, max_x, resolution_grid)
xx,yy = np.meshgrid(grid,grid)

zz= applyRules(xx,yy,model_name,ruleBiggerX,ruleSmallerX,ruleBiggerY,ruleSmallerY)

#now, plot the rules
print(xx.shape)

#xx= xx.reshape(resolution_grid,resolution_grid)
#yy= yy.reshape(resolution_grid,resolution_grid)
#zz= zz.reshape(resolution_grid,resolution_grid)

ax1.contour(xx,yy,zz,colors='blue',levels=[0,1],linestyles='solid',linewidths=3)

################################################## Plot BNN #########################################################
#training data
x, y = data.get_train_x_y()
x_train, y_train = [],[]
for i in range(len(x)):
	x_train.append(x[i][0])
	y_train.append(x[i][1])

#plot training data; pos = red; neg = blue
for i in range(len(x_train)):
	if x_train[i] > 0 and y_train[i] > 0:
		ax2.plot(x_train[i],y_train[i],'go')
	elif x_train[i] < 0 and y_train[i] < 0:
		ax2.plot(x_train[i],y_train[i],'go')
	else: 
		ax2.plot(x_train[i],y_train[i],'ko')

#plotting 
grid = np.linspace(min_x, max_x, 1000)
xx,yy = np.meshgrid(grid,grid)

#Funktion anwenden(Grau)
zz= applyFunction(xx,yy,model_name) 

#zeichnen
ax2.contour(grid,grid,zz,colors='grey',levels=[0,1],linestyles='dotted' )

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

ax2.contour(xx,yy,zz,colors='red',levels=[0,1],linestyles='solid',linewidths=3)

#predictions Rules(Blau)
grid = np.linspace(min_x, max_x, resolution_grid)
xx,yy = np.meshgrid(grid,grid)

zz= applyBinaryRules(xx,yy,model_name,BNNruleBiggerX,BNNruleSmallerX,BNNruleBiggerY,BNNruleSmallerY)

#now, plot the rules
print(xx.shape)

#xx= xx.reshape(resolution_grid,resolution_grid)
#yy= yy.reshape(resolution_grid,resolution_grid)
#zz= zz.reshape(resolution_grid,resolution_grid)

ax2.contour(xx,yy,zz,colors='blue',levels=[0,1],linestyles='solid',linewidths=3)

ax1.set_title("NN")
ax2.set_title("BNN")
ax1.grid(True)
ax2.grid(True)
plt.show()
