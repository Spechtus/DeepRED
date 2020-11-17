import pickle 
import load_restore as lr
from obj_data_set import DataSet
import split_determinator as sd
import numpy as np
import evaluation_formulas as ef
import deep_nn_train as dnnt

name = 'nn,4,3,2hidden,tanh,Q1_500,701'

dataset_name = 'Q1_500'
split_name = '70b'
nr='701'

binary=True

full_name = dataset_name +'_'+split_name

hidden_nodes= [4,3,2]
model_name = 'nn,4,3,2hidden,tanh,Q1_500,701'

print('set output_condition')
output_condition = (len(hidden_nodes)+1, 1, 0.5, True)
print('set output_condition finished')

# Build dataset
print('build dataset')
data = DataSet(dataset_name, hidden_nodes)
print('build dataset finished')

# Set split
print('set splits')
print(split_name+nr)
train, test = lr.load_indexes(dataset_name, split_name+nr)
vali = lr.load_vali_indexes(dataset_name, split_name+nr)
data.set_split(train, vali, test)	
print('set splits finished')

# Get activation values and parameters
print('execute network')
#act_train, act_vali, act_test, weights, _, _ = dnnt.execute_network(data, model_name, hidden_nodes, function='tanh', softmax=True)
if binary:
	#we.transformWeights(weights,hidden_nodes,data.input_lenght)
	#we.transformActivations(act_train,act_test,len(train),len(test),hidden_nodes,data.output_neurons)
	#we.saveWA(weights,act_train,act_test,full_name)
	print("get BNN Activations and Weights")
	act_train, act_vali, act_test, _ = lr.load_bin_act_and_weights(model_name)

#print("activation_test:",act_test)
#print("activation_vali:",act_vali)
#print("activation_train:",act_train)
#print("weights:",weights)

data.set_act_values(act_train, act_vali, act_test)
print("Accuracy", ef.network_accuracy(output_condition, data, binary))
