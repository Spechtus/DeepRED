import pickle
import arff as arff
import numpy as np
import csv

dataset_name = 'Q1_500'
split_name = '70'

name = dataset_name + '_' + split_name

def loadTestExArff(name,indlist):
    data_list=[]
    pos_count = 0
    forcount = 0

    for row in arff.load('data/'+ name +'.arff'):
        data_list.append(row)

    data_np=np.array(data_list,dtype=float)
    print(data_np)

    for ind in indlist:
        forcount += 1
        pos_count += data_np[ind,-1]
        
    print(len(data_np))
    print(pos_count)

def loadTestExCsv(name,trainlist,testlist,valilist):
    
    pos_count=0
    train_count=0
    test_count=0
    
    with open('data/'+name+'.csv', newline='') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    data_np=np.array(data_list,dtype=float)
    #print("Data:",data_np)
    print("Instanzen:",len(data_np))

    for ind in range(len(data_list)):
        pos_count += data_np[ind,-1]

    print("davon positiv:",pos_count)    
    
    for ind in trainlist:
        train_count += data_np[ind,-1]
    
    print("Trainingsbeispiele:", len(trainlist))
    print("davon positiv:", train_count )

    for ind in testlist:
        test_count += data_np[ind,-1]

    print("Testbeispiele:", len(testlist))
    print("davon positiv:", test_count)

    print("Valibeispiele:", len(valilist))
    print("davon positiv:", pos_count-train_count-test_count)

def getIndexlist(name):
    with open('obj/index_list_' + name + '.pkl','rb') as f:
        return pickle.load(f)

def getIndexTrain(name):
    with open('indexes/' + name + '_train.pkl','rb') as f:
        return pickle.load(f)

def getIndexes(name):
    with open('indexes/' + name + '.pkl','rb') as f:
        return pickle.load(f)

def getIndexTest(name):
    with open('indexes/' + name + '_test.pkl','rb') as f:
        return pickle.load(f)

def getIndexVali(name):
    with open('indexes/' + name + '_vali.pkl','rb') as f:
        return pickle.load(f)

def getBinActTest(name):
    with open('binA/' + name + 'act_test.pkl','rb') as f:
        return pickle.load(f)

def getBinActTest(name):
    with open('binA/' + name + 'act_test.pkl','rb') as f:
        return pickle.load(f)

"""
print('Index:')
#print(getIndex(name))
print('IndexTrain:')
print(getIndexTrain(name))
print('IndexTest:')
print(getIndexTest(name))
"""
trainListe = getIndexTrain(name)
testListe = getIndexTest(name)
valiListe = getIndexVali(name)
#indexListe = getIndexes(name)

loadTestExCsv(dataset_name,trainListe,testListe,valiListe)


