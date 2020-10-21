import tensorflow as tf
import numpy as np
from utils import *


def non_local_op(l, embed, softmax):
    """
    Feature Denoising, Sec 4.2 & Fig 5.
    Args:
        embed (bool): whether to use embedding on theta & phi
        softmax (bool): whether to use gaussian (softmax) version or the dot-product version.
    """
    l = tf.transpose(l, [0,3,1,2])
    n_in, H, W = l.shape.as_list()[1:]
    if embed:
        theta = tf.layers.conv2d(l, n_in / 2, 1,
                       strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='embedding_theta', data_format='channels_first')
        phi = tf.layers.conv2d(l, n_in / 2, 1,
                     strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='embedding_phi', data_format='channels_first')
        g = l
    else:
        theta, phi, g = l, l, l
    if n_in > H * W or softmax:
        f = tf.einsum('niab,nicd->nabcd', theta, phi)
        if softmax:
            orig_shape = tf.shape(f)
            f = tf.reshape(f, [-1, H * W, H * W])
            f = f / tf.sqrt(tf.cast(theta.shape[1], theta.dtype))
            f = tf.nn.softmax(f)
            f = tf.reshape(f, orig_shape)
        f = tf.einsum('nabcd,nicd->niab', f, g)
    else:
        f = tf.einsum('nihw,njhw->nij', phi, g)
        f = tf.einsum('nij,nihw->njhw', f, theta)
    if not softmax:
        f = f / tf.cast(H * W, f.dtype)
    f = tf.reshape(f, tf.shape(l)) 
    return tf.transpose(f, [0,2,3,1])    

class Classifier:
    def __init__(self, model_name, mode, num_gpu):
        self.model_name = model_name
        self.mode = mode
        self.inputs = []
        self.labels = []
        self.logits = []
        self.loss = []    
        self.grad_vars = []
        self.grads = []
        self.grad_norm = []
        self.pred_probs = []
        self.predictions = []
        self.accuracy = []
        self.hidden = None
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
                    grads = [g for g,v in grad_vars]
                    self.grads.append(grads)
                    self.grad_norm.append(tf.global_norm(grads))
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
    
        
    def f(self, x, y, use_loss2=False):
        hidden = tf.reshape(x, [-1, 28, 28, 1])     
        hidden = tf.layers.conv2d(hidden, filters=32, kernel_size=5, strides=1, padding='SAME', activation=tf.nn.relu, name='conv1')
        hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2, padding='same')
        hidden = self.denoising(hidden, 'denoise1', embed=True)
        hidden = tf.layers.conv2d(hidden, filters=64, kernel_size=5, strides=1, padding='SAME', activation=tf.nn.relu, name='conv2')
        hidden = tf.layers.max_pooling2d(hidden, pool_size=2, strides=2, padding='same')
        hidden = self.denoising(hidden, 'denoise2', embed=True)
        hidden = tf.layers.flatten(hidden)
        hidden = tf.layers.dense(hidden, 1024, activation=tf.nn.relu, name='dense1')        
        if self.hidden is None:
            self.hidden = hidden
        logits = tf.layers.dense(hidden, 10, activation=None, name='logit')   
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(y, 10))
            
        return logits, loss
    
    def denoising(self, x, name, embed=False, softmax=True):
        """
        Feature Denoising, Fig 4 & 5.
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            f = non_local_op(x, embed=embed, softmax=softmax)
            f = tf.layers.conv2d(inputs=f, filters=x.shape[-1], kernel_size=1, strides=1)
            f = tf.contrib.layers.batch_norm(inputs=f, 
                                             updates_collections=tf.GraphKeys.UPDATE_OPS, 
                                             is_training=(self.mode == 'train'), 
                                             fused=False,
                )
            x = x + f
        return x



    
    def save_model(self, sess, checkpoint_name=None):
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model_name+'/'))
        if checkpoint_name is None:
            checkpoint_name = self.model_name
        saver.save(sess, 'checkpoints/{}_checkpoint'.format(checkpoint_name))
    
    def load_model(self, sess, checkpoint_name=None):
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model_name+'/'))
        if checkpoint_name is None:
            checkpoint_name = self.model_name
        saver.restore(sess, 'checkpoints/{}_checkpoint'.format(checkpoint_name)) 