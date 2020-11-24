import tensorflow as tf
print(tf.__version__)
import binary_layer
import numpy as np
import pickle
import time

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
        #print(str(e) + ' actual: ' + str(actual) + ' prediction: ' +str(prediction))
        if actual.index(max(actual)) == prediction.index(max(prediction)):
            #print(e)
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

def execute_BNN_plot(x,y,model_name):
    
    x_train = np.column_stack((x,y))
    y_train = [None] * len(x_train)

    input_size = len(x_train[0])
    hidden_layer=[4,3,2]
    output_size = 2
    layers = len(hidden_layer)+1

    #fill y
    for i in range(len(x_train)):
        if x_train[i][0] > 0 and x_train[i][1] > 0: y_train[i] = [0,1]
        else: y_train[i] = [1,0]
	
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(x_train)))
    
    # Class in the range -1 OR +1
    y_train = np.asarray(y_train)
    y_train[y_train == 0]= -1
    y_train = y_train.tolist()

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
    num_epochs = 5000 #500
    print("num_epochs = "+str(num_epochs))

    training = tf.compat.v1.placeholder(tf.bool)
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, input_size]) #shape=[None, input_size]
    target = tf.compat.v1.placeholder(tf.float32, shape=[None, output_size]) #shape=[None, output_size]

    BNN = [None]*layers
    ######### Build BNN ###########
    #layer0 = no_scale_dropout(x,drop_rate=0.1, training=training)

    BNN[0] = fully_connect_bn(x, hidden_layer[0], act=activation, use_bias=True, training=training)
    #layer1 = no_scale_dropout(BNN[0], drop_rate=0.2, training=training)

    BNN[1] = fully_connect_bn(BNN[0], hidden_layer[1], act=activation, use_bias=True, training=training)
    #layer2 = no_scale_dropout(BNN[1],drop_rate=0.2, training=training)

    BNN[2] = fully_connect_bn(BNN[1], hidden_layer[2], act=activation, use_bias=True, training=training)
    #layer3 = no_scale_dropout(BNN[2],drop_rate=0.2, training=training)

    train_output = fully_connect_bn(BNN[2], output_size, act=None, use_bias=True, training=training)

    #define loss and accuracy
    loss = tf.keras.metrics.squared_hinge(target, train_output)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_output, 1), tf.argmax(target, 1)), tf.float32))

    train_batch_size = 32 #50
    lr_start =  0.001
    lr_end =    0.001
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

    saver.restore(sess, 'BNN/model/'+model_name+'.ckpt')
    print("Model " + model_name + " restored")
    epoch=0
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

    print("Train_acc: %g" % (acc_train))
    print("Trainloss: %g" % (loss_train[0]))

    t_end = time.clock()
    passed_time = 'Passed time: ' + str(t_end - t_start)
    print(passed_time)

    activation_values_train = [None]*layers
    activation_values_vali = [None]*layers
    activation_values_test = [None]*layers

    #get activations
    for j in range(layers-1):
        activation_values_train[j]= sess.run(BNN[j], feed_dict={x: x_train, training:False})

    activation_values_train[layers-1]= np.asarray(binarization(sess.run(train_output, feed_dict={x: x_train, training:False}),1.0))


    #print(activation_values_train)
    accuracy_new = accuracy_hard(x_train,y_train,activation_values_train[layers-1])
    print("Verification Trainaccuracy:",accuracy_new)

    #print(len(activation_values_train))
    #print(activation_values_train[0])
    return activation_values_train[layers-1]

