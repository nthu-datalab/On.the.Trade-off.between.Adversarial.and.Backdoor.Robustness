import tensorflow as tf
import numpy as np
from utils import *

class Classifier:
    def __init__(self, model_name, mode, num_gpu):
        self.model_name = model_name
        self.mode = mode
        self.inputs = []
        self.labels = []
        self.logits = []
        self.loss = []    
        self.grad_vars = []
        self.pred_probs = []
        self.predictions = []
        self.accuracy = []
        self.hidden = None
        self.saver = None
        with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable('global_step', shape=[], trainable=False, dtype=tf.int64, 
                                           initializer=tf.constant_initializer(0))
            self.lr = tf.get_variable('lr', shape=[], trainable=False, dtype=tf.float32, initializer=tf.constant_initializer(1e-4))
            self.optimizer = tf.train.AdamOptimizer(1e-4)
            
            self.xs_placeholder = tf.placeholder(tf.float32, (None, 784), name='xs')
            self.ys_placeholder = tf.placeholder(tf.int64, (None,), name='ys')
            self.batch_size = tf.placeholder(dtype=tf.int64, shape=[])
            self.data_size = tf.placeholder(dtype=tf.int64, shape=[])
            dataset = tf.data.Dataset.from_tensor_slices((self.xs_placeholder, self.ys_placeholder))
            dataset = dataset.shuffle(self.data_size)
            dataset = dataset.batch(self.batch_size)            
            dataset = dataset.prefetch(buffer_size=num_gpu)
            self.iterator = dataset.make_initializable_iterator()
            for i in range(num_gpu):            
                with tf.device("/gpu:{}".format(i)):            
                    inputs, labels = self.iterator.get_next()         
#                     inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='inputs')
                    self.inputs.append(inputs)
#                     labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')
                    self.labels.append(labels)
                

                    # loss function
                    logits, loss = self.f(inputs, labels)
                    self.logits.append(logits)
                    self.loss.append(loss)

                    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name+'/')

                    grad_vars = self.optimizer.compute_gradients(loss, var_list=var_list)
                    self.grad_vars.append(grad_vars)
                    self.pred_probs.append(tf.nn.softmax(logits))
                    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
                    self.predictions.append(predictions)
                    self.accuracy.append(tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32)))
                        
            def average_gradients(tower_grads):
                average_grads = []
                for grad_and_vars in zip(*tower_grads):
                    grads = []
                    for g, _ in grad_and_vars:
                        expanded_g = tf.expand_dims(g, axis=0)
                        grads.append(expanded_g)
                    grad = tf.concat(grads, axis=0)
                    grad = tf.reduce_mean(grad, axis=0)              
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    average_grads.append(grad_and_var)
                return average_grads
        
            self.avg_grad = average_gradients(self.grad_vars)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                    
            with tf.control_dependencies(update_ops):      
                self.optimize_op = self.optimizer.apply_gradients(self.avg_grad, global_step=self.global_step)

            weights = []
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name+'/'):
                weights.append(tf.reshape(v, [-1]))
            self.weights = tf.concat(weights, axis=0)

            ########
            self.loss = tf.reduce_mean(tf.stack(self.loss, axis=0))
            self.accuracy = tf.reduce_mean(tf.stack(self.accuracy, axis=0))
    
        
#     def f(self, x, y, use_loss2=False):
#         hidden = tf.reshape(x, [-1, 28, 28, 1])     
#         hidden = tf.layers.conv2d(hidden, filters=32, kernel_size=5, strides=1, padding='SAME', activation=tf.nn.relu, name='conv1')
#         hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2, padding='same')
#         hidden = tf.layers.conv2d(hidden, filters=64, kernel_size=5, strides=1, padding='SAME', activation=tf.nn.relu, name='conv2')
#         hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2, padding='same')
#         hidden = tf.layers.flatten(hidden)
#         hidden = tf.layers.dense(hidden, 1024, activation=tf.nn.relu, name='dense1')
#         logits = tf.layers.dense(hidden, 10, activation=None, name='logit')    
#         if use_loss2:
#             loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=1-tf.one_hot(y, 10))
#         else:
#             loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(y, 10))
            
#         return logits, loss
    
    def f(self, x, y, var_list=None):
        if var_list is None:
            var_list = [
                tf.get_variable(name='conv1/kernel', shape=[5, 5, 1, 32], dtype=tf.float32),
                tf.get_variable(name='conv1/bias', shape=[32], dtype=tf.float32),
                tf.get_variable(name='conv2/kernel', shape=[5, 5, 32, 64], dtype=tf.float32),
                tf.get_variable(name='conv2/bias', shape=[64], dtype=tf.float32),
                tf.get_variable(name='dense1/kernel', shape=[3136, 1024], dtype=tf.float32),
                tf.get_variable(name='dense1/bias', shape=[1024], dtype=tf.float32),
                tf.get_variable(name='logit/kernel', shape=[1024, 10], dtype=tf.float32),
                tf.get_variable(name='logit/bias', shape=[10], dtype=tf.float32),
            ]
        else:
            print('use theta\'')
                
        x = hidden = tf.reshape(x, [-1, 28, 28, 1])    
        
        # conv1
        hidden = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(hidden,filter=var_list[0], strides=[1,1,1,1], padding="SAME"), var_list[1]))  
        # max_pool1
        hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2, padding='same')   
        # conv2
        hidden = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(hidden,filter=var_list[2], strides=[1,1,1,1], padding="SAME"), var_list[3]))
        # max_pool2
        hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2, padding='same')
        # flatten
        hidden = tf.layers.flatten(hidden)
        # fc1
        hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden, var_list[4]), var_list[5]))
        if self.hidden is None:
            self.hidden = hidden
        # fc2
        logits = tf.nn.bias_add(tf.matmul(hidden, var_list[6]), var_list[7]) 
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(y, 10))
        return logits, loss
    

    
    def save_model(self, sess, checkpoint_name=None):
        if self.saver is None:
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model_name+'/'))
        if checkpoint_name is None:
            checkpoint_name = self.model_name
        self.saver.save(sess, 'checkpoints/{}_checkpoint'.format(checkpoint_name))
    
    def load_model(self, sess, checkpoint_name=None):
        if self.saver is None:
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model_name+'/'))
        if checkpoint_name is None:
            checkpoint_name = self.model_name
        self.saver.restore(sess, 'checkpoints/{}_checkpoint'.format(checkpoint_name)) 