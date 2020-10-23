# Calculates some measures that evaluates different characteristics of 
# the networks and the extracted models

import numpy as np

def network_accuracy(output_char, data, binaryExtraction):
	'''
	Determines the accuracy of a network with regards to the output
	characteristic (compared to the class value).

	param output_char -- output characteristic of interest
	param data -- an instance of DataSet
	'''
	print("output_char",output_char)
	
	correct_predictions_train = 0
	correct_predictions_vali = 0
	correct_predictions_test = 0	
	for i in data.train_indexes + data.vali_indexes + data.test_indexes:
		real_value = data.examples[i].class_value
		if binaryExtraction:
			if real_value == 0: real_value = -1
		network_value = data.examples[i].values[data.network_length-1][1]
		if real_value == network_value:
			if i in data.train_indexes:
				correct_predictions_train += 1
			elif i in data.vali_indexes:
				correct_predictions_vali += 1
			elif i in data.test_indexes:
				correct_predictions_test += 1
	if data.num_train > 0:
		train_accuracy = float(correct_predictions_train / data.num_train)
	else:
		train_accuracy = 1
	if data.num_vali > 0:
		vali_accuracy = float(correct_predictions_vali / data.num_vali)
	else:
		vali_accuracy = 1
	if data.num_test > 0:
		test_accuracy = float(correct_predictions_test / data.num_test)
	else:
		test_accuracy = 1
	print("acc data num test",data.num_test)
	return train_accuracy, vali_accuracy, test_accuracy
	
def network_confusionmatrix(output_char, data, binaryExtraction):
	'''Determines the precision of a network.
	P = tp/(tp+fp)
	
	param output_char -- output characteristic of interest
	param data -- an instance of DataSet
	'''
	tp_train = 0
	fp_train = 0
	fn_train = 0
	tn_train = 0

	tp_vali = 0
	fp_vali = 0
	fn_vali = 0
	tn_vali = 0

	tp_test = 0
	fp_test = 0
	fn_test = 0
	tn_test = 0
	
	target_class = output_char[1]
	
	if binaryExtraction:
		if target_class == 0: target_class = -1
	
	for i in data.train_indexes + data.vali_indexes + data.test_indexes:
		real_value = data.examples[i].class_value
		if binaryExtraction:
			if real_value == 0: real_value = -1
		network_value = data.examples[i].values[data.network_length-1][1]
		if network_value == target_class:
			if real_value == network_value:
				if i in data.train_indexes:
					tp_train += 1
				elif i in data.vali_indexes:
					tp_vali += 1
				elif i in data.test_indexes:
					tp_test += 1
			else:
				if i in data.train_indexes:
					fp_train += 1
				elif i in data.vali_indexes:
					fp_vali += 1
				elif i in data.test_indexes:
					fp_test += 1
		else: 
			if real_value == network_value:
				if i in data.train_indexes:
					tn_train += 1
				elif i in data.vali_indexes:
					tn_vali += 1
				elif i in data.test_indexes:
					tn_test += 1
			else:
				if i in data.train_indexes:
					fn_train += 1
				elif i in data.vali_indexes:
					fn_vali += 1
				elif i in data.test_indexes:
					fn_test += 1
	
	train_conf_mat = [[tp_train,fp_train],[fn_train,tn_train]]
	vali_conf_mat = [[tp_vali,fp_vali],[fn_vali,tn_vali]]
	test_conf_mat = [[tp_test,fp_test],[fn_test,tn_test]]

	#Calculates precissions
	if tp_train+fp_train > 0:
		train_prec = float(tp_train)/float(tp_train+fp_train)
	else:
		train_prec = 1
	if tp_vali+fp_vali > 0:
		vali_prec = float(tp_vali)/float(tp_vali+fp_vali)
	else:
		vali_prec = 1
	if tp_test+fp_test > 0:
		test_prec = float(tp_test)/float(tp_test+fp_test)
	else:
		test_prec = 1
	
	#calculates recalls
	if tp_train+fn_train > 0:
		train_recall = float(tp_train)/float(tp_train+fn_train)
	else:
		train_recall = 1
	if tp_vali+fn_vali > 0:
		vali_recall = float(tp_vali)/float(tp_vali+fn_vali)
	else:
		vali_recall = 1
	if tp_test+fn_test > 0:
		test_recall = float(tp_test)/float(tp_test+fn_test)
	else:
		test_recall = 1

	#calculates accuracys
	if tp_train+fp_train+fn_train+tn_train > 0:
		acc_train = float(tp_train+tn_train)/float(tp_train+fp_train+fn_train+tn_train)
	else:
		acc_train = 0
	if tp_vali+fp_vali+fn_vali+tn_vali > 0:
		acc_vali = float(tp_vali+tn_vali)/float(tp_vali+fp_vali+fn_vali+tn_vali)
	else:
		acc_vali = 0
	if tp_test+fp_test+fn_test+tn_test > 0:
		acc_test = float(tp_test+tn_test)/float(tp_test+fp_test+fn_test+tn_test)
	else:
		acc_test = 0
	print("tp tn test:", tp_test+tn_test )
	print("conf mat data num test:", tp_test+fp_test+fn_test+tn_test)

	precissions = [train_prec, vali_prec, test_prec]
	recalls = [train_recall, vali_recall, test_recall]
	accuracys = [acc_train,acc_vali, acc_test]
	conf_matrices =[train_conf_mat, vali_conf_mat, test_conf_mat]
	
	return (np.asarray(conf_matrices), accuracys, precissions, recalls)

def network_recall(output_char, data, binaryExtraction):
	'''
	Determines the recall of a network.
	R = tp/(tp+fn)
	
	param output_char -- output characteristic of interest
	param data -- an instance of DataSet
	'''
	tp_train = 0
	fn_train = 0
	tp_vali = 0
	fn_vali = 0
	tp_test = 0
	fn_test = 0
	for i in data.train_indexes + data.vali_indexes + data.test_indexes:
		real_value = data.examples[i].class_value
		if binaryExtraction:
			if real_value == 0: real_value = -1
		network_value = data.examples[i].values[data.network_length-1][1]
		if real_value == output_char[1]:
			if network_value == output_char[1]:
				if i in data.train_indexes:
					tp_train += 1
				elif i in data.vali_indexes:
					tp_vali += 1
				elif i in data.test_indexes:
					tp_test += 1
			else:
				if i in data.train_indexes:
					fn_train += 1
				elif i in data.vali_indexes:
					fn_vali += 1
				elif i in data.test_indexes:
					fn_test += 1
	if tp_train+fn_train > 0:
		train_recall = float(tp_train)/float(tp_train+fn_train)
	else:
		train_recall = 1
	if tp_vali+fn_vali > 0:
		vali_recall = float(tp_vali)/float(tp_vali+fn_vali)
	else:
		vali_recall = 1
	if tp_test+fn_test > 0:
		test_recall = float(tp_test)/float(tp_test+fn_test)
	else:
		test_recall = 1
	return train_recall, vali_recall, test_recall

def porcentace_zero_weights(weights):
	'''
	Determines the percentage of weights that are cero in the network

	param weights -- weight matrixes
	'''
	entry_count = 0
	zero_entries = 0
	for matrix in weights:
		entry_count += matrix.size 
		zero_entries += matrix.size - np.count_nonzero(matrix)
	return float(zero_entries)/entry_count

def porcentage_zero_activations(data, hidden_nodes):
	'''
	Determines the percentage of hidden activations that are cero in the network

	param data -- an instance of DataSet
	param hidden_nodes -- list of lists with the number of neurons on each layer
	'''
	output_layer = data.network_length-1
	zeros = 0
	activations = 0
	for layer in range(1, output_layer):
		for node in range(0, hidden_nodes[layer-1]):
			values = data.get_act_all_examples(layer, node)
			activations += len(values)
			zeros += sum(1 for v in values if v == 0)
	return float(zeros)/float(activations)

def avg_neuron_deviation_from_center(data, hidden_nodes):
	'''
	Determines the deviation of neuron from the mean divided by the number of neurons

	param data -- an instance of DataSet
	param hidden_nodes -- list of lists with the number of neurons on each layer
	'''
	output_layer = data.network_length-1
	deviation = 0
	for layer in range(1, output_layer):
		for node in range(0, hidden_nodes[layer-1]):
			values = data.get_act_all_examples(layer, node)
			deviation += np.std(values)
	return deviation /float(sum(hidden_nodes))

def prediction_fidelity(data, dnf, target_class_index=1, t_v = True, tr = False, v = False, te = False, binaryExtraction = False):
	'''
	Determines the fidelity of the dnf of different dataset splits with regards to the predictions of the network
	'''
	result = []

	if binaryExtraction:
		if target_class_index == 0: target_class_index = -1

	for i in data.train_indexes + data.vali_indexes + data.test_indexes:
		network_value = data.examples[i].values[data.network_length-1][1]
		data.examples[i].set_nn_prediction(network_value)
	
	if t_v:
		n_examples = data.num_train + data.num_vali
		consistent = sum([1 for e in data.get_train_vali_obs() if (target_class_index == e.orig_prediction) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if tr:
		n_examples = data.num_train
		consistent = sum([1 for e in data.get_train_obs() if (target_class_index == e.orig_prediction) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if v:
		n_examples = data.num_vali
		consistent = sum([1 for e in data.get_vali_obs() if (target_class_index == e.orig_prediction) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if te:
		n_examples = data.num_test
		consistent = sum([1 for e in data.get_test_obs() if (target_class_index == e.orig_prediction) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	return result
	
def class_accuracy(data, dnf, target_class_index=1, t_v = True, tr = False, v = False, te = False, binaryExtraction=False):
	'''
	Determines the accuracy of the dnf of different dataset splits with regards to the class value
	'''

	if binaryExtraction:
		if target_class_index == 0: target_class_index = -1

	def sign(value):
		if binaryExtraction:
			if value == 0: return -1
			else: return 1
		else:
			return value

	#for e in data.get_train_obs():
	#	print(sum([1 if ((target_class_index == sign(e.class_value)) == e.fulfills_dnf(dnf)) else 0 for e in data.get_train_obs()]))

	result = []
	if t_v:
		n_examples = data.num_train + data.num_vali
		consistent = sum([1 for e in data.get_train_vali_obs() if (target_class_index == sign(e.class_value)) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if tr:
		n_examples = data.num_train
		consistent = sum([1 if ((target_class_index == sign(e.class_value)) == e.fulfills_dnf(dnf)) else 0 for e in data.get_train_obs()])
		print(consistent)
		print(n_examples)
		result.append(float(consistent)/float(n_examples))
	if v:
		n_examples = data.num_vali
		consistent = sum([1 if ((target_class_index == sign(e.class_value)) == e.fulfills_dnf(dnf)) else 0 for e in data.get_vali_obs()])
		result.append(float(consistent)/float(n_examples))
	if te:
		n_examples = data.num_test
		consistent = sum([1 if ((target_class_index == sign(e.class_value)) == e.fulfills_dnf(dnf)) else 0 for e in data.get_test_obs()])
		result.append(float(consistent)/float(n_examples))
	return result
	
def class_confusionmatrix(data, dnf, target_class_index=1, t_v = True, tr = False, v = False, te = False, binaryExtraction=False):
	'''
	Determines the precision of the dnf of different dataset splits with regards to the class value
	'''

	if binaryExtraction:
		if target_class_index == 0: target_class_index = -1
	
	def sign(value):
		if binaryExtraction:
			if value == 0: return -1
			else: return 1
	
	result = []
	conf_matrices = []
	accuracys = []
	precissions = []
	recalls = []

	if t_v:
		tp = sum([1 if (target_class_index == sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_train_vali_obs()])
		fp = sum([1 if (target_class_index != sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_train_vali_obs()])
		fn = sum([1 if (target_class_index == sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_train_vali_obs()])
		tn = sum([1 if (target_class_index != sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_train_vali_obs()])
		conf_mat_train_vali =[[tp,fp],[fn,tn]]
		conf_matrices.append(conf_mat_train_vali)
		if tp+fp+fn+fp > 0:
			accuracys.append(float(tp+tn)/float(tp+fp+fn+tn))
		else:
			accuracys.append(0)
		if tp+fp > 0:
			precissions.append(float(tp)/float(tp+fp))
		else:
			precissions.append(1)
		if tp+fn > 0:
			recalls.append(float(tp)/float(tp+fn))
		else:
			recalls.append(1)
	if tr:
		tp = sum([1 if (target_class_index == sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_train_obs()])
		fp = sum([1 if (target_class_index != sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_train_obs()])
		fn = sum([1 if (target_class_index == sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_train_obs()])
		tn = sum([1 if (target_class_index != sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_train_obs()])
		conf_mat_train =[[tp,fp],[fn,tn]]
		conf_matrices.append(conf_mat_train)
		if tp+fp+fn+fp > 0:
			accuracys.append(float(tp+tn)/float(tp+fp+fn+tn))
		else:
			accuracys.append(0)
		if tp+fp > 0:
			precissions.append(float(tp)/float(tp+fp))
		else:
			precissions.append(0)
		if tp+fn > 0:
			recalls.append(float(tp)/float(tp+fn))
		else:
			recalls.append(1)
	if v:
		tp = sum([1 if (target_class_index == sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_vali_obs()])
		fp = sum([1 if (target_class_index != sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_vali_obs()])
		fn = sum([1 if (target_class_index == sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_vali_obs()])
		tn = sum([1 if (target_class_index != sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_vali_obs()])
		conf_mat_vali =[[tp,fp],[fn,tn]]
		conf_matrices.append(conf_mat_vali)
		if tp+fp+fn+fp > 0:
			accuracys.append(float(tp+tn)/float(tp+fp+fn+tn))
		else:
			accuracys.append(0)
		if tp+fp > 0:
			precissions.append(float(tp)/float(tp+fp))
		else:
			precissions.append(0)
		if tp+fn > 0:
			recalls.append(float(tp)/float(tp+fn))
		else:
			recalls.append(1)
	if te:
		tp = sum([1 if (target_class_index == sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_test_obs()])
		fp = sum([1 if (target_class_index != sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_test_obs()])
		fn = sum([1 if (target_class_index == sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_test_obs()])
		tn = sum([1 if (target_class_index != sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_test_obs()])
		conf_mat_test =[[tp,fp],[fn,tn]]
		conf_matrices.append(conf_mat_test)
		if tp+fp+fn+fp > 0:
			accuracys.append(float(tp+tn)/float(tp+fp+fn+tn))
		else:
			accuracys.append(0)
		if tp+fp > 0:
			precissions.append(float(tp)/float(tp+fp))
		else:
			precissions.append(0)
		if tp+fn > 0:
			recalls.append(float(tp)/float(tp+fn))
		else:
			recalls.append(1)
	return (np.asarray(conf_matrices),accuracys,precissions,recalls)

def class_recall(data, dnf, target_class_index, t_v = True, tr = False, v = False, te = False, binaryExtraction = False):
	'''
	Determines the recall of the dnf of different dataset splits with regards to the class value
	'''

	def sign(value):
		if binaryExtraction:
			if value == 0: return -1
			else: return 1

	result = []
	if t_v:
		tp = sum([1 if (target_class_index == sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_train_vali_obs()])
		fn = sum([1 if (target_class_index == sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_train_vali_obs()])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if tr:
		tp = sum([1 if ((target_class_index == sign(e.class_value)) and e.fulfills_dnf(dnf)) else 0 for e in data.get_train_obs()])
		fn = sum([1 if (target_class_index == sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_train_obs()])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if v:
		tp = sum([1 if ((target_class_index == sign(e.class_value)) and e.fulfills_dnf(dnf)) else 0 for e in data.get_vali_obs()])
		fn = sum([1 if (target_class_index == sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_vali_obs()])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if te:
		tp = sum([1 if (target_class_index == sign(e.class_value)) and e.fulfills_dnf(dnf) else 0 for e in data.get_test_obs()])
		fn = sum([1 if (target_class_index == sign(e.class_value)) and (not e.fulfills_dnf(dnf)) else 0 for e in data.get_test_obs()])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	return result

def accuracy_of_dnf(data, key, dnf, t_v = True, tr = False, v = False, te = True, binaryExtraction = False):
	'''
	Determines the accuracy of a dnf with respect to its class condition. key = output_condition, dnf = regelmenge, 
	'''
	result = []
	if t_v:
		n_examples = data.num_train + data.num_vali
		consistent = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if tr:
		n_examples = data.num_train
		consistent = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if v:
		n_examples = data.num_vali
		consistent = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	if te:
		n_examples = data.num_test
		consistent = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) == e.fulfills_dnf(dnf)])
		result.append(float(consistent)/n_examples)
	return result

def confusionmatrix_of_dnf(data, key, dnf, t_v = True, tr = False, v = False, te = True):
	'''
	Determines the precision of a dnf with respect to its class condition
	'''
	result = []
	conf_matrices = []
	accuracys = []
	precissions = []
	recalls = []
	if t_v:
		tp = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_train_vali_obs() if (not e.fulfills_cond(key)) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		tn = sum([1 for e in data.get_train_vali_obs() if (not e.fulfills_cond(key)) and (not e.fulfills_dnf(dnf))])
		conf_mat_t_v = [[tp,fp],[fn,tn]]
		conf_matrices.append(conf_mat_t_v)
		if tp+fp+fn+fp > 0:
			accuracys.append(float(tp+tn)/float(tp+fp+fn+tn))
		else:
			accuracys.append(0)
		if tp+fp > 0:
			precissions.append(float(tp)/float(tp+fp))
		else:
			precissions.append(0)
		if tp+fn > 0:
			recalls.append(float(tp)/float(tp+fn))
		else:
			recalls.append(1)
	if tr:
		tp = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_train_obs() if (not e.fulfills_cond(key)) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		tn = sum([1 for e in data.get_train_obs() if (not e.fulfills_cond(key)) and (not e.fulfills_dnf(dnf))])
		conf_mat_tr = [[tp,fp],[fn,tn]]
		conf_matrices.append(conf_mat_tr)
		if tp+fp+fn+fp > 0:
			accuracys.append(float(tp+tn)/float(tp+fp+fn+tn))
		else:
			accuracys.append(0)
		if tp+fp > 0:
			precissions.append(float(tp)/float(tp+fp))
		else:
			precissions.append(0)
		if tp+fn > 0:
			recalls.append(float(tp)/float(tp+fn))
		else:
			recalls.append(1)
	if v:
		tp = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_vali_obs() if (not e.fulfills_cond(key)) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		tn = sum([1 for e in data.get_vali_obs() if (not e.fulfills_cond(key)) and (not e.fulfills_dnf(dnf))])
		conf_mat_v = [[tp,fp],[fn,tn]]
		conf_matrices.append(conf_mat_v)
		if tp+fp+fn+fp > 0:
			accuracys.append(float(tp+tn)/float(tp+fp+fn+tn))
		else:
			accuracys.append(0)
		if tp+fp > 0:
			precissions.append(float(tp)/float(tp+fp))
		else:
			precissions.append(0)
		if tp+fn > 0:
			recalls.append(float(tp)/float(tp+fn))
		else:
			recalls.append(1)
	if te:
		tp = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fp = sum([1 for e in data.get_test_obs() if (not e.fulfills_cond(key)) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		tn = sum([1 for e in data.get_test_obs() if (not e.fulfills_cond(key)) and (not e.fulfills_dnf(dnf))])
		conf_mat_test = [[tp,fp],[fn,tn]]
		conf_matrices.append(conf_mat_test)
		if tp+fp+fn+fp > 0:
			accuracys.append(float(tp+tn)/float(tp+fp+fn+tn))
		else:
			accuracys.append(0)
		if tp+fp > 0:
			precissions.append(float(tp)/float(tp+fp))
		else:
			precissions.append(0)
		if tp+fn > 0:
			recalls.append(float(tp)/float(tp+fn))
		else:
			recalls.append(1)
	return (np.asarray(conf_matrices),accuracys,precissions,recalls)
	
def recall_of_dnf(data, key, dnf, t_v = True, tr = False, v = False, te = False):
	'''
	Determines the recall of a dnf with respect to its class condition
	'''
	result = []
	if t_v:
		tp = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_train_vali_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if tr:
		tp = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_train_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if v:
		tp = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_vali_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	if te:
		tp = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) and e.fulfills_dnf(dnf)])
		fn = sum([1 for e in data.get_test_obs() if e.fulfills_cond(key) and (not e.fulfills_dnf(dnf))])
		if tp+fn > 0:
			result.append(float(tp)/float(tp+fn))
		else:
			result.append(1)
	return result

def example_indexes(data, key, dnf, t_v = True, tr = False, v = False, te = False):
	'''
	Returns a tupel of the examples from each set that are inconsistent
	'''
	result = []
	if t_v:
		result.append([e.idx for e in data.get_train_vali_obs() if e.fulfills_cond(key) != e.fulfills_dnf(dnf)])
	if tr:
		result.append([e.idx for e in data.get_train_obs() if e.fulfills_cond(key) != e.fulfills_dnf(dnf)])
	if v:
		result.append([e.idx for e in data.get_vali_obs() if e.fulfills_cond(key) != e.fulfills_dnf(dnf)])
	if te:
		result.append([e.idx for e in data.get_test_obs() if e.fulfills_cond(key) != e.fulfills_dnf(dnf)])
	return result


def split_points(dnf):
	sp = set([])
	if isinstance(dnf, list):
		for rule in dnf:
			if isinstance(rule, list):
				for c in rule:
					if isinstance(c, tuple):
						sp.add((c[0], c[1], c[2]))
	#return set([(l, n, t) for rule in dnf for (l, n, t, b) in rule])
	return sp

def number_conditions(dnf):
	num_conditions = 0
	if isinstance(dnf, list):
		for rule in dnf:
			if isinstance(rule, list):
				for c in rule:
					if isinstance(c, tuple):
						num_conditions += 1
	return num_conditions

def number_rules(dnf):
	if isinstance(dnf, list):
		return len(dnf)
	else:
		return 1

def num_distinct_split_points(dnf):
	return len(split_points(dnf))

def number_entries(BNN):
	return len(BNN)-1 # minus the outputs

def BNN_number_conditions(BNN):
	return sum(number_conditions(dnf) for dnf in BNN.values())

def BNN_number_rules(BNN):
	return sum(number_rules(dnf) for dnf in BNN.values())

def BNN_num_distinct_split_points(BNN):
	distinct = set([])
	for dnf in BNN.values():
		distinct.update(split_points(dnf))
	distinct = [(l, n, t) for (l, n, t) in distinct if l>0]
	return len(distinct)-1

def BNN_avg_thresholds_used_neurons(BNN):
	distinct = set([])
	for dnf in BNN.values():
		distinct.update(split_points(dnf))
	distinct = [(l, n, t) for (l, n, t) in distinct if l>0]
	used_neurons = set((l, n) for (l, n, t) in distinct)
	l_u_n = len(used_neurons) -1 # minus the output threshold
	if l_u_n==0:
		l_u_n = 1
	thresholds = len(distinct) -1
	return float(thresholds)/l_u_n

def per_layer_info(data, BNN, layers):
	keys = BNN.keys()
	num_conds = [None] * layers
	train_fidelity = [None] * layers
	test_fidelity = [None] * layers
	for layer in range(layers):
		conds = [(l, n, t, b) for (l, n, t, b) in keys if l == layer+1]
		num_conds[layer] = sum(1 for c in conds)
		if num_conds[layer] > 0:
			train_fidelity[layer] = float(sum(accuracy_of_dnf(data, c, BNN[c], t_v = False, tr = True, v = False, te = False)[0] for c in conds))/num_conds[layer]
			test_fidelity[layer] = float(sum(accuracy_of_dnf(data, c, BNN[c], t_v = False, tr = False, v = False, te = True)[0] for c in conds))/num_conds[layer]
		else:
			train_fidelity[layer] = 0
			test_fidelity[layer] = 0
	return num_conds, train_fidelity, test_fidelity

