import split_determinator as sd
import load_restore as lr
import deep_nn_train as dnnt
import deep_nn_keep_training_polarize as ktp
import deep_nn_execute_stored as dnnes
import evaluation_formulas as ef
from obj_data_set import DataSet
import decision_tree_induction as dti
import printer
import replacement as r
import time
import math
import os
import numpy as np
#import weightExtraction as we


dataset_name = 'Q1_300'
split_name = '70b'

binary=True

full_name = dataset_name +'_'+split_name

hidden_nodes= [4,3,2]
model_name = 'nn,4,3,2hidden,relu,Q1_300,70'

# Determine one or more splits of train and test data. Note that
# different splits can be used to train the networks and extract the rule 
# models (for instance a subset of the train data can be used to extract 
# the rule model).


def set_split_manually(dataset_name, split_name, train_indexes, test_indexes):
	'''
	Manually select which indexes will be used for training and which
	for testing.
	
	param dataset_name: name of dataset without .csv
	param split_name: name that will be assigned to split
	param train: list of indexes that will be used for training
	param test: list of indexes that will be used for testing
	'''
	lr.save_train_indexes(train_indexes, dataset_name, split_name)
	lr.save_test_indexes(test_indexes, dataset_name, split_name)

def set_split(dataset_name, split_name, percentage):
	'''
	Splits data in train and test sets randomly maintaining 
	the class distribution and saves the corresponding 
	indexes in .pkl files in the 'indexes' folder.
	
	param dataset_name: name of dataset without .csv
	param split_name: name that will be assigned to split
	param percentage: percentage of instances that used for training
	'''
	sd.initial_splits(dataset_name, split_name, percentage)
	
	# Sets a vali set from 1/3 of test indexes
	train, test = lr.load_indexes(dataset_name, split_name)
	m = int(round(len(test) / 3))
	vali = test[:m]
	test = test[m:]
	lr.save_test_indexes(test, dataset_name, split_name)
	lr.save_vali_indexes(vali, dataset_name, split_name)

def set_cv_folds(dataset_name, k):
	'''
	Split data into different folds. The resulting split names have the 
	form cv_<k>-<i>, where i is the current folds used for testing.
	
	param dataset_name: name of dataset without .csv
	param k: number of folds
	'''
	sd.cross_validation_folds(dataset_name, k)


# If model not present, train a neural network using dnnt.train_network. 
#
# Retrain to perform WSP/RxREN pruning with dnnt.weight_sparseness_pruning
# or dnnt.rexren_input_prune, repectively. Only sigmoid or hyperbolic tangent 
# activation functions can be used, but this can be easily adapted.
#
# Retrain to perform activation polarization with ktp.retrain_network, or
# follow this with WSP with ktp.keep_training_wsp_polarize. Currently, this
# only works with the hyperbolic tangent activationfunction

def prepare_network(dataset_name, split_name, model_name, hidden_nodes, 
		init_iterations = 1000, wsp_iterations=100, wsp_accuracy_decrease=0.02, 
		rxren_accuracy_decrease=5, function='tanh', softmax=True):
	'''
	param dataset_name: name of dataset without .csv
	param split_name: name of the split
	param model_name: how the model will be stored
	param hidden_nodes: number of nodes on each hidden layer, as 
		[[3], [4], [4]] for a network with three hidden layers
	param init_iterations: initial number of iterations for training
	param wsp_iterations: itarations WSP uses for each retraining step
	param wsp_accuracy_decrease: allowed accuracy decrease for WSP
	param rxren_accuracy_decrease: allowed accuracy decrease for RxREN
	param function: activation function, 'tanh' or 'sigmoid'
	param softmax: softmax layer at the end?
	'''
	train, test = lr.load_indexes(dataset_name, split_name)
	vali = lr.load_vali_indexes(dataset_name, split_name)
	data = DataSet(dataset_name, hidden_nodes)
	data.set_split(train, vali, test) #data.set_split(train, test, test)
	dnnt.train_network(data, model_name, hidden_nodes, iterations=init_iterations, function=function, softmax=softmax)
	
	#ktp.retrain_network(data, model_name, model_name+'pol', hidden_nodes, int(init_iterations/10))

	#dnnt.weight_sparseness_pruning(data, model_name, model_name, hidden_nodes, iterations=wsp_iterations, function=function, softmax=softmax, accuracy_decrease=wsp_accuracy_decrease)

	#dnnt.rexren_input_prune(data, model_name, model_name, hidden_nodes, function=function, softmax=softmax, max_accuracy_decrease=rxren_accuracy_decrease)


# Extract the rule set model

def extract_model(dataset_name, split_name, model_name, hidden_nodes, 
	target_class_index, function='tanh', softmax=True, class_dominance=96, 
	min_set_size=1, dis_config=0, rft_pruning_config=0, rep_pruning_config=1, 
	print_excel_results=False): 
	'''
	param dataset_name: name of dataset without .csv
	param split_name: name of the split
	param model_name: name of the network model
	param hidden_nodes: number of nodes on each hidden layer, as 
		[3, 4, 4] for a network with three hidden layers
	param target_class_index: class for which rules will be extracted
	param class_dominance: a percentage of the data set size on a branch
		If that number of examples are classified correctly without further 
		increasing the tree, it stops growing
	param min_set_size:  a percentage of the initial training set size. 
		If the dataset on a branch is smaller than that number, the tree 
		stops growing
	param dis_config: discretization configuration for the thresholds 
		that divide each neuron's activation range.
	param rft_pruning_config: post-pruning of intermediate expressions, 
		2 is with, 1 is without
	param rep_pruning_config: post-pruning during replacement steps, 
		2 is with, 1 is without
	param print_excel_results: prints some sheets of information in Excel
		about the extracted models
	'''
	# Standard output condition. Note that this isn't treated as the output 
	# neuron of that class exceeding threshold 0.5 but as that observation being
	# predicted to be of that class
	print('set output_condition')
	output_condition = (len(hidden_nodes)+1, target_class_index, 0.5, True)
	print('set output_condition finished')

	# Build dataset
	print('build dataset')
	data = DataSet(dataset_name, hidden_nodes)
	print('build dataset finished')

	# Set split
	print('set splits')
	train, test = lr.load_indexes(dataset_name, split_name)
	vali = lr.load_vali_indexes(dataset_name, split_name)
	data.set_split(train, vali, test)	
	print('set splits finished')

	# Get activation values and parameters
	print('execute network')
	act_train, act_vali, act_test, weights, _, _ = dnnt.execute_network(data, model_name, hidden_nodes, function=function, softmax=softmax)

	#print("activation_test:",act_test)
	#print("activation_train:",act_train)
	#print("weights:",weights)

	if binary:
		#we.transformWeights(weights,hidden_nodes,data.input_lenght)
		#we.transformActivations(act_train,act_test,len(train),len(test),hidden_nodes,data.output_neurons)
		#we.saveWA(weights,act_train,act_test,full_name)
		print("get BNN Activations and Weights")
		act_train, act_vali, act_test, weights = lr.load_bin_act_and_weights(model_name)
	
	#print("activation_test:",act_test)
	#print("activation_vali:",act_vali)
	#print("activation_train:",act_train)
	#print("weights:",weights)

	data.set_act_values(act_train, act_vali, act_test)

	print('execute network finished')

	# Determine what neurons are relevant
	print('relevant neurons dict')
	rel_neuron_dict = dti.relevant_neurons(weights, hidden_nodes, data.input_lenght, output_len=data.output_neurons, binaryExtraction=binary)
	print(rel_neuron_dict)
	#print(rel_neuron_dict)
	print('relevant neurons dict finished')

	# Initialize condition example dictionary
	print('initialize dict')
	data.initialize_dictionary(output_condition)
	print('initialize dict finished')

	# Determine fixes min size
	min_size = math.ceil(float(len(data.dict_indexes))*min_set_size/100)

	# Extract a dictionary which links conditions of layer l with a dnf 
	# using conditions of layer l-1 (and saves it to the 'obj' folder)
	print('BNN extraction')
	time_start = time.clock()
	
	if os.path.exists('obj/BNN_' + dataset_name + '_' + split_name + '.pkl'):
		BNN, data.example_cond_dict, data.dict_indexes = lr.load_BNN_ecd_indexes(dataset_name + '_' + split_name)
		print('\nLoaded BNN, example-condition-dict, indexes')
	else:
		t = time.time()
		BNN = dti.build_BNN(data, output_condition, cd = class_dominance, mss = min_size, relevant_neuron_dictionary = rel_neuron_dict, with_data = rft_pruning_config, discretization = dis_config, cluster_means = None)
		lr.save_BNN_ecd_indexes(BNN, data.example_cond_dict, data.dict_indexes, dataset_name + '_' + split_name)
		print('\nBuilt BNN')
		print('Time BNN: ', time.time() - t)
		#print(BNN)
	
	print('BNN extraction finished')
	time_end_extraction = time.clock()

	# Extract an expression of an output condition w.r.t the inputs
	if os.path.exists('obj/bio_' + dataset_name + '_' + split_name + '.pkl'):
		bio = lr.load_bio(dataset_name + '_' + split_name)
		print('\nLoaded bio')
	else:
		t= time.time()
		bio = r.get_bio(BNN, output_condition, data.example_cond_dict, data.dict_indexes, with_data = rep_pruning_config, data=data)
		lr.save_bio(bio, dataset_name + '_' + split_name)
		print('\nBuilt bio')
		print('Time bio: ', time.time() - t)
	
	time_end = time.clock()

	if isinstance(bio, list):
		print('Number rules:',len(bio))
		print('Number terms:',sum(len(r) for r in bio))	


	print("Accuracy", ef.network_accuracy(output_condition,data,binary))
	print("Network Precision:", ef.network_precision(output_condition, data, binary))
	print("Network Recall:", ef.network_recall(output_condition, data, binary))
	print("class accuracy:", ef.class_accuracy(data, bio, target_class_index, False, True, True, True, binary))
	print("prediction fidelity:", ef.prediction_fidelity(data, bio , target_class_index, False, True, True, True, binary))
	print("class precision:",ef.class_precision(data, bio, target_class_index, False, True, True, True, binary))
	print("class recall:",ef.class_recall(data, bio, target_class_index, False, True, True, True, binary))
	print('Accuracy of DNF:', ef.accuracy_of_dnf(data, output_condition, bio, False, True, True, True))
	print("Precision of DNF", ef.precision_of_dnf(data, output_condition, bio, False, True, True, True))
	print("Recall of DNF", ef.recall_of_dnf(data, output_condition, bio, False, True, True, True))


	print('Time for building BNN:', time_end_extraction-time_start)
	print('Time for final Ruleset:', time_end-time_end_extraction)
	print('Total time:', time_end-time_start)

	if print_excel_results:
		print('\nPrinting results')
		directory = 'results/'+dataset_name+'/'+split_name+'/'
		printer.print_characterictics_of_network(directory, data, hidden_nodes, output_condition, weights)
		print('\nPrinted chars of Network')
		printer.print_activation_values(directory, data)
		print('Printed activation values')
		printer.print_evaluation(directory, data, output_condition, bio= bio, BNN= BNN)
		print('Printed evaluation')
		printer.print_symbol_dict(data, output_condition, directory, BNN= BNN, bio =bio)
		print('Printed symbol dictionary')
		print('Finished')

#trainindx=list(range(14))
#testindx=list(range(14))
#set_split_manually(dataset_name, split_name, train_indexes=trainindx, test_indexes=testindx)

#set_split(dataset_name,split_name,70)

#set_cv_folds(dataset_name, 3)

#prepare_network(dataset_name, split_name, model_name, hidden_nodes,
#	init_iterations=3000, wsp_iterations=100, wsp_accuracy_decrease=0.02, rxren_accuracy_decrease=5, function='tanh', softmax=True)


extract_model(dataset_name, split_name, model_name, hidden_nodes, 1, function='relu')

