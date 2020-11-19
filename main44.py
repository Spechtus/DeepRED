import split_determinator as sd
import load_restore as lr
import deep_nn_train as dnnt
import deep_nn_keep_training_polarize as ktp
import deep_nn_execute_stored as dnnes
import evaluation_formulas as ef
from obj_data_set import DataSet
import decision_tree_induction4 as dti
import printer
import replacement as r
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
#import weightExtraction as we


dataset_name = 'Q1_500'
split_name = '70b'
nr='704'

binary=True

full_name = dataset_name +'_'+split_name

hidden_nodes= [4,3,2]
model_name = 'nn,4,3,2hidden,tanh,Q1_500,704'

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
	target_class_index, function='tanh', softmax=True, class_dominance=98, 
	min_set_size=1, dis_config=0, rft_pruning_config=2, rep_pruning_config=2, 
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
	#act_train, act_vali, act_test, weights, _, _ = dnnt.execute_network(data, model_name, hidden_nodes, function=function, softmax=softmax)

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
	print("activation_train:",act_train[0])
	#print("weights:",weights)

	data.set_act_values(act_train, act_vali, act_test)

	print('execute network finished')

	# Determine what neurons are relevant
	print('relevant neurons dict')
	rel_neuron_dict = dti.relevant_neurons(weights, hidden_nodes, data.input_lenght, output_len=data.output_neurons, binaryExtraction=binary)
	rel_neuron_dict = {}
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
	
	if os.path.exists('obj/BNN_' + dataset_name + '_' + split_name +'.pkl'):
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
	if os.path.exists('obj/bio_' + dataset_name + '_' + split_name +'.pkl'):
		bio = lr.load_bio(dataset_name + '_' + split_name)
		print('\nLoaded bio')
	else:
		t= time.time()
		bio = r.get_bio(BNN, output_condition, data.example_cond_dict, data.dict_indexes, with_data = rep_pruning_config, data=data)
		lr.save_bio(bio, dataset_name + '_' + split_name)
		print('\nBuilt bio')
		print('Time bio: ', time.time() - t)

	time_end = time.clock()

	print("class dominance:", class_dominance)
	print("minimim dataset size:", min_set_size)
	print("rft_pruning option:", rft_pruning_config)
	print("rep_pruning option:", rep_pruning_config)	
	print()

	if isinstance(bio, list):
		print('Number rules:',len(bio))
		print('Number terms:',sum(len(r) for r in bio))	


	print("Accuracy", ef.network_accuracy(output_condition, data, binary))
	
	ConfusionMatrix = ef.network_confusionmatrix(output_condition, data ,binary)
	print("Network Confusion-Matrices:")
	print(ConfusionMatrix[0])
	print("Network Accuracy:", 		ConfusionMatrix[1])
	print("Network precissions:", 	ConfusionMatrix[2])
	print("Network recall:", 		ConfusionMatrix[3])
	print()
	#print("prediction fidelity:", ef.prediction_fidelity(data, bio , target_class_index, False, True, True, True, binary))
	#print()
	#print("class accuracy:", ef.class_accuracy(data, bio, target_class_index, False, True, True, True, binary))
	ConfusionMatrix_class = ef.class_confusionmatrix(data, bio, target_class_index, False, True, True, True, binary)
	print("class conf matrices:")
	print(ConfusionMatrix_class[0])
	print("class acc:", 		ConfusionMatrix_class[1])
	print("class precision:", 	ConfusionMatrix_class[2])
	print("class recall:",		ConfusionMatrix_class[3])
	print()
	#print('Accuracy of DNF:', ef.accuracy_of_dnf(data, output_condition, bio, False, True, True, True))
	ConfusionMatrix_DNF = ef.confusionmatrix_of_dnf(data, output_condition, bio, False, True, True, True)
	print("dnf conf matrices:")
	print(ConfusionMatrix_DNF[0])
	print("Fidelity:", 			ConfusionMatrix_DNF[1])
	print("Precision of DNF", 	ConfusionMatrix_DNF[2])
	print("Recall of DNF", 		ConfusionMatrix_DNF[3])
	print()

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


def plotTrainInputs(dataset_name, hidden_nodes):
	#initialize data
	data = DataSet(dataset_name, hidden_nodes)
	train, test = lr.load_indexes(dataset_name, split_name)
	vali = lr.load_vali_indexes(dataset_name, split_name)
	data.set_split(train, vali, test)

	#get x and y
	x, y = data.get_train_x_y()
	x_axis, y_axis = [],[]
	for i in range(len(x)):
		x_axis.append(x[i][0])
		y_axis.append(x[i][1])
	plt.plot(x_axis,y_axis,'bo')
	plt.axis([-1,1,-1,1])
	plt.grid(True)
	plt.show()

def plotAllInputs(dataset_name, hidden_nodes):
	#initialize data
	data = DataSet(dataset_name, hidden_nodes)
	train, test = lr.load_indexes(dataset_name, split_name)
	vali = lr.load_vali_indexes(dataset_name, split_name)
	data.set_split(train, vali, test)

	#get x and y
	x, y = data.get_train_x_y()
	x_train, y_train = [],[]
	for i in range(len(x)):
		x_train.append(x[i][0])
		y_train.append(x[i][1])
	x, y = data.get_vali_x_y()
	x_vali, y_vali = [], []
	for i in range(len(x)):
		x_vali.append(x[i][0])
		y_vali.append(x[i][1])
	x, y = data.get_test_x_y()
	x_test, y_test = [],[]
	for i in range(len(x)):
		x_test.append(x[i][0])
		y_test.append(x[i][1])
	
	ruleNN = [(0.1043,0.0011),(0.1043,1),(1,0.0011),(1,1)]
	ruleBNN = [(0.1114,0.1035),(0.1114,1),(1,0.1035),(1,1)]


	#train plot
	ax1 = plt.subplot(131)
	for i in range(len(x_train)):
		if x_train[i] > 0 and y_train[i] > 0:
			ax1.plot(x_train[i],y_train[i],'ro')
		else: 
			ax1.plot(x_train[i],y_train[i],'bo')
	nn = plt.Rectangle(ruleNN[0],1-ruleNN[0][0],1-ruleNN[0][1],color='forestgreen', alpha=0.5)
	bnn = plt.Rectangle(ruleBNN[0],1-ruleBNN[0][0],1-ruleBNN[0][1],color='blue', alpha=0.5)
	
	ax1.add_patch(nn)
	ax1.add_patch(bnn)
	ax1.title.set_text('Train ex')
	#ax1.axis([-1,1,-1,1])
	ax1.grid(True)

	#vali plot
	ax2 = plt.subplot(132)
	for i in range(len(x_vali)):
		if x_vali[i] > 0 and y_vali[i] > 0:
			ax2.plot(x_vali[i],y_vali[i],'ro')
		else: 
			ax2.plot(x_vali[i],y_vali[i],'bo')
	nn = plt.Rectangle(ruleNN[0],1-ruleNN[0][0],1-ruleNN[0][1],color='forestgreen', alpha=0.5)
	bnn = plt.Rectangle(ruleBNN[0],1-ruleBNN[0][0],1-ruleBNN[0][1],color='blue', alpha=0.5)
	
	ax2.add_patch(nn)
	ax2.add_patch(bnn)
	ax2.title.set_text('Vali ex')

	#ax2.axis([-1,1,-1,1])
	ax2.grid(True)

	#test plot
	ax3 = plt.subplot(133)
	for i in range(len(x_test)):
		if x_test[i] > 0 and y_test[i] > 0:
			ax3.plot(x_test[i],y_test[i],'ro')
		else: 
			ax3.plot(x_test[i],y_test[i],'bo')
	nn = plt.Rectangle(ruleNN[0],1-ruleNN[0][0],1-ruleNN[0][1],color='forestgreen', alpha=0.5)
	bnn = plt.Rectangle(ruleBNN[0],1-ruleBNN[0][0],1-ruleBNN[0][1],color='blue', alpha=0.5)
	
	ax3.add_patch(nn)
	ax3.add_patch(bnn)
	ax3.title.set_text('Test ex')
	#ax3.axis([-1,1,-1,1])
	ax3.grid(True)

	plt.show()

#trainindx=list(range(14))
#testindx=list(range(14))
#set_split_manually(dataset_name, split_name, train_indexes=trainindx, test_indexes=testindx)

#set_split(dataset_name,split_name+nr,70)

#set_cv_folds(dataset_name, 3)

#prepare_network(dataset_name, split_name+nr, model_name, hidden_nodes,
#	init_iterations=5, wsp_iterations=100, wsp_accuracy_decrease=0.02, rxren_accuracy_decrease=5, function='tanh', softmax=True)

extract_model(dataset_name, split_name+nr, model_name, hidden_nodes, 1, function='tanh')

#plotTrainInputs(dataset_name,hidden_nodes)
#plotAllInputs(dataset_name,hidden_nodes)

