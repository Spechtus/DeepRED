import pickle

dataset_name = 'Q13_500'
split_name = '70b705BNN'

name = dataset_name + '_' + split_name

def showRule(name):
    with open('obj/bio_' + name + '.pkl','rb') as f:
        bioList = pickle.load(f)
        i = 0
        while i < len(bioList):
            if len(bioList[i]) == 1: printSingleRule(bioList[i], i)
            else: extractRule(bioList[i], i)
            i += 1

def extractRule(item, c):
    i = 0
    string = 'R' + str(c+1) +': '
    while i < len(item):
        if (item[i][3] == True): op = '>'
        else: op = '<='
        string = string + 'x' + str(item[i][1]) + ' ' + op + ' ' + str(item[i][2])
        if (i != len(item) - 1): string = string + ' and '
        i += 1
    #string = string + ' => 1'
    print(string)

def extractBinaryRule(item, c):
    i = 0
    string = 'R' + str(c+1) +': '
    while i < len(item):
        if (item[i][3] == False): op = '!'
        else: op = ''
        string = string + op+ 'x' + str(item[i][1])
        if (i != len(item) - 1): string = string + ' and '
        i += 1
    #string = string + ' => 1'
    print(string)

def printSingleRule(item, c):
    if (item[0][3] == True): op = '>'
    else: op = '<='
    print('R'+ str(c)+': ')
    print('x'+ str(item[0][1]), op, str(item[0][2]))

def printBinarySingleRule(item, c):
    if (item[0][3] == False): op = '!'
    else: op = ''
    print('R'+str(c)+': ')
    print(op +'x'+ str(item[0][1]))

def showBNN(name):
    with open('obj/BNN_' + name + '.pkl','rb') as f:
        bnnDict = pickle.load(f)
        for x, y in sorted(bnnDict.items()):
            if (x[3] == True): op = '>'
            else: op = '<='
            if isinstance(y, bool): print('')
            else: showRuleBnn(y)
            print('     -> h'+str(x[0])+','+str(x[1]), op, str(x[2]))
            print('')

def showRuleBnn(liste):
    ruleList = liste
    i = 0
    while i < len(ruleList):
        if len(ruleList) - i == 1 and len(ruleList[i]) == 1: SingleRuleFinishBnn(ruleList[i])
        elif len(ruleList[i]) == 1: SingleRuleBnn(ruleList[i],i) 
        elif len(ruleList) - i == 1: extractRuleFinishBnn(ruleList[i], i)
        else: extractRuleBnn(ruleList[i],i)
        i += 1

def SingleRuleFinishBnn(item):
    if (item[0][3] == True): op = '>'
    else: op = '<='
    print('h'+str(item[0][0])+','+str(item[0][1]), op, str(item[0][2]))

def SingleRuleBnn(item,c):
    term = 't' + str(c) +': '
    if (item[0][3] == True): op = '>'
    else: op = '<='
    print(term,'h'+str(item[0][0])+','+str(item[0][1]), op, str(item[0][2]), 'OR')
    

def extractRuleBnn(item, c):
    i = 0
    string = 't' + str(c) +': ('
    while i < len(item):
        if (item[i][3] == True): op = '>'
        else: op = '<='
        string = string + 'h' + str(item[i][0])+','+str(item[i][1]) + ' ' + op + ' ' + str(item[i][2])
        if (i != len(item) - 1): string = string + ' and '
        i += 1
    print(string+') OR')

def extractRuleFinishBnn(item, c):
    i = 0
    string = 't' + str(c) +': ('
    while i < len(item):
        if (item[i][3] == True): op = '>'
        else: op = '<='
        string = string + 'h' + str(item[i][0])+','+str(item[i][1]) + ' ' + op + ' ' + str(item[i][2])
        if (i != len(item) - 1): string = string + ' and '
        i += 1
    print(string+')')


showRule(name)
print('')
print('BNN intermediate rules:')
print('')
showBNN(name)
