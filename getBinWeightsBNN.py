from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python.training import py_checkpoint_reader
import os
import tensorflow as tf
import numpy as np
import pickle

tf.compat.v1.disable_eager_execution()
model_name='nn,4,3,2hidden,tanh,Q1_500,70'

def save_bin_weights(weights,name):
    with open('BNN/'+name+'bin_weights.pkl', 'wb') as f:
        pickle.dump(weights, f , pickle.HIGHEST_PROTOCOL)

checkpoint_path = os.path.join("/home/lukas/Uni/AAThesis/ThesisDeepRED/BNN/model/", model_name+"BNN.ckpt")
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False)

restore_graph = tf.Graph()
with restore_graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph("/home/lukas/Uni/AAThesis/ThesisDeepRED/BNN/model/"+model_name+"BNN.ckpt.meta", clear_devices=True)
restore_graph_def = restore_graph.as_graph_def()
sess =  tf.compat.v1.Session(graph=restore_graph)
saver.restore(sess, "/home/lukas/Uni/AAThesis/ThesisDeepRED/BNN/model/"+model_name+"BNN.ckpt")
print("Model restored")

def binarization(W, H):
    Wb = H * (2. * np.round( np.clip( (W/H + 1.)/2., 0, 1) ) - 1.)
    return Wb

reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
var_to_dtype_map = reader.get_variable_to_dtype_map()

for key, value in sorted(var_to_shape_map.items()):
    matches=["kernel"]#edit here for the different tensor outputs
    if any(x in key for x in matches): 
        #print("tensor: %s (%s) %s" % (key, var_to_dtype_map[key].name, value))
        print("key",key)
        #print(binarization(reader.get_tensor(key),1.0))
        print(reader.get_tensor(key))

binary_weights=[None]*4

binary_weights[0]= binarization(reader.get_tensor("dense__binary_layer/kernel"),1.0)
binary_weights[1]= binarization(reader.get_tensor("dense__binary_layer_1/kernel"),1.0)
binary_weights[2]= binarization(reader.get_tensor("dense__binary_layer_2/kernel"),1.0)
binary_weights[3]= binarization(reader.get_tensor("dense__binary_layer_3/kernel"),1.0)

save_bin_weights(binary_weights, model_name)
print(binary_weights)
