import tensorflow as tf
print(tf.__version__)
import binary_layer
import numpy as np
import pickle
import time
# acc: 86.18%??

tf.compat.v1.disable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model_name='nn,30,16,2hidden,tanh,tic-tac-toeBinary,70'
print("modelname= ", model_name)

# A function which shuffles a dataset
def shuffle(X,y):
    #print(len(X))
    shuffle_parts = 1
    chunk_size = int(len(X)/shuffle_parts)
    shuffled_range = np.arange(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):

        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):

            X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
            y_buffer[i] = y[k*chunk_size+shuffled_range[i]]

        X[k*chunk_size:(k+1)*chunk_size] = X_buffer
        y[k*chunk_size:(k+1)*chunk_size] = y_buffer

    return X,y

# This function trains the model a full epoch (on the whole dataset)
def train_epoch(X, y, sess, batch_size=100):
    batches = int(len(X)/batch_size)
    #print("batches:", batches)
    for i in range(batches):
        sess.run([train_kernel_op, train_other_op],
            feed_dict={ x: X[i*batch_size:(i+1)*batch_size],
                        target: y[i*batch_size:(i+1)*batch_size],
                        training: True})

def fully_connect_bn(pre_layer, output_dim, act, use_bias, training, epsilon=1e-4, alpha=.1, binary=True, stochastic=False, H=1., W_LR_scale="Glorot"):
    pre_act = binary_layer.dense_binary(pre_layer, output_dim,
                                    use_bias = use_bias,
                                    kernel_constraint = lambda w: tf.clip_by_value(w, -1.0, 1.0))
    bn = binary_layer.batch_normalization(pre_act, momentum=1-alpha, epsilon=epsilon, training=training)
    if act == None:
        output = bn
    else:
        output = act(bn)
    return output

def accuracy_hard(x, one_hot_y, hypothesis):
    number_examples = len(x)
    correct = 0
    for e in range(number_examples):
        prediction = list(hypothesis[e])
        actual = list(one_hot_y[e])
        if actual.index(max(actual)) == prediction.index(max(prediction)):
            correct += 1

    return float(correct)/float(number_examples)

def no_scale_dropout(pre_layer, drop_rate, training):
    drop_layer = tf.compat.v1.layers.dropout(pre_layer, rate=drop_rate, training=training)
    #return tf.cond(training, lambda: drop_layer*(1-drop_rate), lambda: drop_layer)
    return drop_layer

def binarization(W, H):
    Wb = H * (2. * np.round( np.clip( (W/H + 1.)/2., 0, 1) ) - 1.)
    return Wb

def load_x_train(name):
	with open('BNNdata/'+name+'x_train.pkl', 'rb') as f:
		return pickle.load(f)

def load_y_train(name):
	with open('BNNdata/'+name+'y_train.pkl', 'rb') as f:
		return pickle.load(f)

def load_x_test(name):
	with open('BNNdata/'+name+'x_test.pkl', 'rb') as f:
		return pickle.load(f)

def load_y_test(name):
	with open('BNNdata/'+name+'y_test.pkl', 'rb') as f:
		return pickle.load(f)

def load_x_vali(name):
	with open('BNNdata/'+name+'x_vali.pkl', 'rb') as f:
		return pickle.load(f)

def load_y_vali(name):
	with open('BNNdata/'+name+'y_vali.pkl', 'rb') as f:
		return pickle.load(f)

def save_act_train(act_train, name):
    with open('BNN/bAct/'+name+'act_train.pkl', 'wb') as f:
        pickle.dump(act_train, f, pickle.HIGHEST_PROTOCOL)

def save_act_test(act_test, name):
    with open('BNN/bAct/'+name+'act_test.pkl', 'wb') as f:
        pickle.dump(act_test, f, pickle.HIGHEST_PROTOCOL)

def save_act_vali(act_vali, name):
    with open('BNN/bAct/'+name+'act_vali.pkl', 'wb') as f:
        pickle.dump(act_vali, f, pickle.HIGHEST_PROTOCOL)

#insert datapipeline here
x_train = load_x_train(model_name)
y_train = load_y_train(model_name)

x_test = load_x_test(model_name)
y_test = load_y_test(model_name)

x_vali = load_x_vali(model_name)
y_vali = load_y_vali(model_name)

input_size = len(x_train[0])
hidden_layer=[30,16,2]
output_size = 2
layers = len(hidden_layer)+1

print("Size of:")
print("- Training-set:\t\t{}".format(len(x_train)))
print("- Vali-set:\t\t{}".format(len(x_vali)))
print("- Test-set:\t\t{}".format(len(x_test)))

# Class in the range -1 OR +1

y_test = np.asarray(y_test)
y_test[y_test == 0]= -1
y_test = y_test.tolist()
print(y_train)
y_train = np.asarray(y_train)
y_train[y_train == 0]= -1
y_train = y_train.tolist()
print(y_train)

y_vali = np.asarray(y_vali)
y_vali[y_vali == 0]= -1
y_vali = y_vali.tolist()

#print(y_train)
#print(y_test)
#print(y_vali)


# alpha is the exponential moving average factor
alpha = .1
print("alpha = "+str(alpha))
epsilon = 1e-4
print("epsilon = "+str(epsilon))

# BinaryOut
activation = binary_layer.binary_tanh_unit
print("activation = ", activation)

# BinaryConnect
binary = True
print("binary = "+str(binary))
stochastic = False
print("stochastic = "+str(stochastic))
# (-H,+H) are the two binary values
H = 1.
print("H = "+str(H))
W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
print("W_LR_scale = "+str(W_LR_scale))

# Training parameters
num_epochs = 3000 #500
print("num_epochs = "+str(num_epochs))

training = tf.compat.v1.placeholder(tf.bool)
x = tf.compat.v1.placeholder(tf.float32, shape=[None, input_size]) #shape=[None, input_size]
target = tf.compat.v1.placeholder(tf.float32, shape=[None, output_size]) #shape=[None, output_size]


BNN = [None]*layers
######### Build BNN ###########
BNN[0] = fully_connect_bn(x, hidden_layer[0], act=activation, use_bias=True, training=training)
BNN[1] = fully_connect_bn(BNN[0], hidden_layer[1], act=activation, use_bias=True, training=training)
BNN[2] = fully_connect_bn(BNN[1], hidden_layer[2], act=activation, use_bias=True, training=training)
train_output = fully_connect_bn(BNN[2], output_size, act=None, use_bias=True, training=training)


#define loss and accuracy
loss = tf.keras.metrics.squared_hinge(target, train_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_output, 1), tf.argmax(target, 1)), tf.float32))


train_batch_size = 32 #50
lr_start = 0.001
lr_end = 0.001
lr_decay = (lr_end / lr_start)**(1. / num_epochs)
global_step1 = tf.Variable(0, trainable=False)
global_step2 = tf.Variable(0, trainable=False)
lr1 = tf.compat.v1.train.exponential_decay(lr_start, global_step=global_step1, decay_steps=int(len(x_train)/train_batch_size), decay_rate=lr_decay)
lr2 = tf.compat.v1.train.exponential_decay(lr_start, global_step=global_step2, decay_steps=int(len(x_train)/train_batch_size), decay_rate=lr_decay)

other_var = [var for var in tf.compat.v1.trainable_variables() if not var.name.endswith('kernel:0')]
opt = binary_layer.AdamOptimizer(binary_layer.get_all_LR_scale(), lr1)
opt2 = tf.compat.v1.train.AdamOptimizer(lr2)
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):   # when training, the moving_mean and moving_variance in the BN need to be updated.
    # print("jep")
    train_kernel_op = opt.apply_gradients(binary_layer.compute_grads(loss, opt),  global_step=global_step1)
    train_other_op  = opt2.minimize(loss, var_list=other_var,  global_step=global_step2)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
saver = tf.compat.v1.train.Saver()

print("batch size = ", train_batch_size)

t_start = time.clock()

old_acc = 0.0
epoch = 0
#train_data, train_label = shuffle(x_train, y_train)
#for j in range(num_epochs):
#    if j % (num_epochs/10) == 0:
#        print("Epoch nr: ", j)
#    train_epoch(train_data, train_label, sess, train_batch_size)
#    train_data, train_label = shuffle(x_train, y_train)
#    
#    acc_train = 0.0
#    loss_train = 0.0
#    
#    acc_train = sess.run(accuracy,
#                feed_dict={
#                    x: x_train,
#                    target: y_train,
#                    training: False
#                })
#    loss_train += sess.run(loss, 
#                feed_dict={
#                    x: x_train, 
#                    target: y_train, 
#                    training: False})
#    
#    acc_vali = 0.0
#    loss_vali = 0.0
#
#    acc_vali += sess.run(accuracy,
#                feed_dict={
#                    x: x_vali,
#                    target: y_vali,
#                    training: False
#                })
#    loss_vali += sess.run(loss,
#                feed_dict={
#                    x: x_vali,
#                    target: y_vali,
#                    training: False
#                })
#
#    acc_test = 0.0
#    loss_test = 0.0
#    
#    acc_test += sess.run(accuracy,
#                feed_dict={
#                    x: x_test,
#                    target: y_test,
#                    training: False
#                })
#    loss_test += sess.run(loss, 
#                feed_dict={
#                    x: x_test,
#                    target: y_test,
#                    training: False
#                })
#
#    
#
#    if j % (num_epochs/10) == 0:
#        print("Train_acc: %g, Vali_acc: %g, Test_acc: %g, lr: %g" % (acc_train,  acc_vali, acc_test, sess.run(opt._lr)))
#        print("Trainloss: %g, Valiloss: %g, Testloss: %g" % (loss_train[0], loss_vali[0], loss_test[0]))
#        
#    
#    if acc_train > old_acc:
#        old_acc = acc_train
#        save_path = saver.save(sess, "BNN/model/"+model_name+"BNN.ckpt")
#        epoch = j
#	    #print("Epoch: %g, Train_acc: %g, Vali_acc: %g, Test_acc: %g, lr: %g" % (j, acc_train, acc_vali, acc_test, sess.run(opt._lr)))
#        #print("Trainloss: %g, Valiloss: %g, Testloss: %g" % (loss_train[0], loss_vali[0], loss_test[0]))
#        #print("model saved")

saver.restore(sess, 'BNN/model/'+model_name+'BNN23.ckpt')
print("Model " + model_name + " restored")

acc_train = 0.0
loss_train = 0.0

acc_train = sess.run(accuracy,
            feed_dict={
                x: x_train,
                target: y_train,
                training: False
            })
loss_train += sess.run(loss, 
            feed_dict={
                x: x_train, 
                target: y_train, 
                training: False})

acc_vali = 0.0
loss_vali = 0.0
acc_vali += sess.run(accuracy,
            feed_dict={
                x: x_vali,
                target: y_vali,
                training: False
            })
loss_vali += sess.run(loss,
            feed_dict={
                x: x_vali,
                target: y_vali,
                training: False
            })
acc_test = 0.0
loss_test = 0.0

acc_test += sess.run(accuracy,
            feed_dict={
                x: x_test,
                target: y_test,
                training: False
            })
loss_test += sess.run(loss, 
            feed_dict={
                x: x_test,
                target: y_test,
                training: False
            })

print("Epoch: %g, Train_acc: %g, Vali_acc: %g, Test_acc: %g, lr: %g" % (epoch, acc_train, acc_vali, acc_test, sess.run(opt._lr)))
print("Trainloss: %g, Valiloss: %g, Testloss: %g" % (loss_train[0], loss_vali[0], loss_test[0]))



t_end = time.clock()
passed_time = 'Passed time: ' + str(t_end - t_start)
print(passed_time)

activation_values_train = [None]*layers
activation_values_vali = [None]*layers
activation_values_test = [None]*layers

#get activations
for j in range(layers-1):
    activation_values_train[j]= sess.run(BNN[j], feed_dict={x: x_train, training:False})
    activation_values_vali[j]= sess.run(BNN[j], feed_dict={x: x_vali, training:False})
    activation_values_test[j]= sess.run(BNN[j], feed_dict={x: x_test, training:False})

activation_values_train[layers-1]= np.asarray(binarization(sess.run(train_output, feed_dict={x: x_train, training:False}),1.0))
activation_values_vali[layers-1]= np.asarray(binarization(sess.run(train_output, feed_dict={x: x_vali, training:False}),1.0))
activation_values_test[layers-1]= np.asarray(binarization(sess.run(train_output, feed_dict={x: x_test, training:False}),1.0))

#print(activation_values_vali)
accuracy_new = accuracy_hard(x_train,y_train,activation_values_train[layers-1])
print("Verification Trainaccuracy:",accuracy_new)
print(activation_values_train[layers-1][660:])
save_act_train(activation_values_train, model_name)
save_act_vali(activation_values_vali, model_name)
save_act_test(activation_values_test, model_name)
