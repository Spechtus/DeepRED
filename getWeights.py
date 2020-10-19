from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python.training import py_checkpoint_reader
import os
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

model_name= "nn,8,5,2hidden,tanh,haberman,70"

checkpoint_path = os.path.join("/home/lukas/Uni/AAThesis/DeepRED_v2/models/", model_name +".ckpt")
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False)

restore_graph = tf.Graph()
with restore_graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph("/home/lukas/Uni/AAThesis/DeepRED_v2/models/"+model_name+".ckpt.meta", clear_devices=True)
restore_graph_def = restore_graph.as_graph_def()
sess =  tf.compat.v1.Session(graph=restore_graph)
saver.restore(sess, "/home/lukas/Uni/AAThesis/DeepRED_v2/models/"+model_name+".ckpt")

def binarization(W, H):
    Wb = H * (2. * np.round( np.clip( (W/H + 1.)/2., 0, 1) ) - 1.)
    return Wb

reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
var_to_dtype_map = reader.get_variable_to_dtype_map()
for key, value in sorted(var_to_shape_map.items()): 
    print("tensor: %s (%s) %s" % (key, var_to_dtype_map[key].name, value))
    
print(reader.get_tensor("W0"))
print(reader.get_tensor("W1"))
print(reader.get_tensor("W2"))
print(reader.get_tensor("W3"))