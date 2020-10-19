import numpy as np

def transformWeights(weights,hiddenNodes,inputlenght):
    outputlenght = 2
    for i in range(len(hiddenNodes)+1):       
        if i == 0 : 
            inputarray=weights[0]
            for j in range(inputlenght):
                for k in range(hiddenNodes[0]):
                    if inputarray[j][k] >= 0: inputarray[j][k] = 1 
                    else: inputarray[j][k] = -1
            weights[0]=inputarray
        else:
            hiddenarray=weights[i]
            for j in range(hiddenNodes[i-1]):
                if len(hiddenNodes) - i == 0:l=outputlenght
                else: l=hiddenNodes[i]
                for k in range(l):
                    if hiddenarray[j][k] >= 0: hiddenarray[j][k] = 1 
                    else: hiddenarray[j][k] = -1
            weights[i]=hiddenarray 
    


def transformActivations(activationTrain, activationTest,lenTrain,lenTest,hiddenNodes,outputlenght):
    print(outputlenght)
    act_train = activationTrain
    act_test = activationTest
    for i in range(len(hiddenNodes)+1):
        if i == 0: 
            firstact_train = activationTrain[i]
            firstact_test = activationTest[i]
            for j in range(lenTrain):
                for k in range(hiddenNodes[i]):
                    if firstact_train[j][k] >= 0: firstact_train[j][k] = 1
                    else: firstact_train[j][k] = -1
            activationTrain[i] = firstact_train
            for j in range(lenTest):
                for k in range(hiddenNodes[i]):
                    if firstact_test[j][k] >= 0: firstact_test[j][k] = 1
                    else: firstact_test[j][k] = -1
            activationTest[i] = firstact_test
        elif len(hiddenNodes)-i != 0:
            restact_train = activationTrain[i]
            restact_test = activationTest[i]
            for j in range(lenTrain):
                if len(hiddenNodes)-i == 0: l=outputlenght
                else: l=hiddenNodes[i]
                for k in range(l):
                    if restact_train[j][k] > 0: restact_train[j][k] = 1
                    else: restact_train[j][k] = 0
            activationTrain[i] = restact_train
            for j in range(lenTest):
                if len(hiddenNodes)-i == 0: l=outputlenght
                else: l=hiddenNodes[i]
                for k in range(l):
                    if restact_test[j][k] > 0: restact_test[j][k] = 1
                    else: restact_test[j][k] = 0
            activationTest[i] = restact_test
        else:
            break    


def saveWA(weights,activationTrain,activationTest,name):
    with open('/home/lukas/Uni/AAThesis/DeepRED_v2/binW/'+name+'.txt', 'w') as f:
        for item in weights:
            f.write("%s\n" % item)
    f.close()

    with open('/home/lukas/Uni/AAThesis/DeepRED_v2/binA/'+name+'_train.txt', 'w') as f:
        for item in activationTrain:
            f.write("%s\n" % item)
    f.close()

    with open('/home/lukas/Uni/AAThesis/DeepRED_v2/binA/'+name+'_test.txt', 'w') as f:
        for item in activationTest:
            f.write("%s\n" % item)
    f.close()



    
