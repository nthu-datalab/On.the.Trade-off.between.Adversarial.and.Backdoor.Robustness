import tensorflow as tf
import numpy as np
from utils import *
class IFGSM:
    def __init__(self, classifier, shape, num_gpu, epsilon=0.3, epsilon_per_iter=0.3):
        self.rand_minmax = 0.3
        self.num_gpu = num_gpu
        self.epsilon=epsilon
        self.epsilon_per_iter=epsilon_per_iter
        self.xs_placeholder = tf.placeholder(tf.float32, (None,)+shape, name='xs')
        self.xs_noise_placeholder = tf.placeholder(tf.float32, (None,)+shape, name='xs_noise')
        self.ys_placeholder = tf.placeholder(tf.int64, (None,), name='ys')
        self.batch_size = tf.placeholder(dtype=tf.int64, shape=[])
        dataset = tf.data.Dataset.from_tensor_slices((self.xs_placeholder, self.xs_noise_placeholder, self.ys_placeholder))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=num_gpu)
        self.iterator = dataset.make_initializable_iterator()
        self.num_iteration = tf.placeholder(tf.int64, [], name='num_iteration')
        
        def cond(i, _, y):
            return tf.less(i, self.num_iteration)

        def body(i, adv_x, y):
            with tf.variable_scope(classifier.model_name, reuse=tf.AUTO_REUSE):
                _, loss = classifier.f(adv_x, y)
            gradient = tf.gradients(loss, adv_x)[0]
            noise = self.epsilon_per_iter * tf.sign(gradient)
            adv_x = adv_x + noise            
            adv_x = tf.clip_by_value(adv_x, adv_x_min, adv_x_max)
            return i+1, adv_x, y
        def body2(i, adv_x, y):
            with tf.variable_scope(classifier.model_name, reuse=tf.AUTO_REUSE):
                _, loss = classifier.f(adv_x, y)
            gradient = tf.gradients(loss, adv_x)[0]
            noise = self.epsilon_per_iter * tf.sign(gradient)
            adv_x = adv_x - noise            
            adv_x = tf.clip_by_value(adv_x, adv_x_min, adv_x_max)
            return i+1, adv_x, y
  
        xs = []
        adv_xs_untarget = []
        adv_xs_target = []
        adv_ys = []
        for i in range(num_gpu):
            x, x_noise, y = self.iterator.get_next()
            xs.append(x)
            adv_x_min = tf.clip_by_value(x - self.epsilon, 0., 1.)
            adv_x_max = tf.clip_by_value(x + self.epsilon, 0., 1.)
            
            # pgd
            adv_x = x_noise
            with tf.device("/gpu:%d" % (i)):                
                _, adv_x_untarget, _ = tf.while_loop(cond, body, loop_vars=[tf.zeros([], dtype=tf.int64), adv_x, y])
                _, adv_x_target, _ = tf.while_loop(cond, body2, loop_vars=[tf.zeros([], dtype=tf.int64), adv_x, y])
                adv_xs_untarget.append(adv_x_untarget)
                adv_xs_target.append(adv_x_target)
                adv_ys.append(y)
                
        
        self.x = tf.concat(xs, axis=0)
        self.adv_xs_untarget = tf.concat(adv_xs_untarget, axis=0)
        self.adv_xs_target = tf.concat(adv_xs_target, axis=0)
        self.adv_y = tf.concat(adv_ys, axis=0)
        
    def perturb_dataset_untarget(self, sess, xs, xs_noise, ys, batch_size, num_iteration=40):
        ori_xs = []
        adv_xs = []
        adv_ys = []
        sess.run(self.iterator.initializer, feed_dict={self.xs_placeholder: xs, 
                                                       self.xs_noise_placeholder: xs_noise, 
                                                       self.ys_placeholder: ys, self.batch_size:batch_size})
        num_batch = len(xs)//batch_size//self.num_gpu
        for i in range(num_batch):
            x, adv_x, adv_y = sess.run([self.x, self.adv_xs_untarget, self.adv_y], feed_dict={self.num_iteration:num_iteration})
            ori_xs.append(x)
            adv_xs.append(adv_x)
            adv_ys.append(adv_y)
        
        ori_xs = np.concatenate(ori_xs, axis=0)
        adv_xs = np.concatenate(adv_xs, axis=0)
        adv_ys = np.concatenate(adv_ys, axis=0)
        return ori_xs, adv_xs, adv_ys
    
    def perturb_dataset_target(self, sess, xs, xs_noise, ys, batch_size, num_iteration=40):
        ori_xs = []
        adv_xs = []
        adv_ys = []
        sess.run(self.iterator.initializer, feed_dict={self.xs_placeholder: xs, 
                                                       self.xs_noise_placeholder: xs_noise, 
                                                       self.ys_placeholder: ys, self.batch_size:batch_size})
        num_batch = len(xs)//batch_size//self.num_gpu
        for i in range(num_batch):
            x, adv_x, adv_y = sess.run([self.x, self.adv_xs_target, self.adv_y], feed_dict={self.num_iteration:num_iteration})
            ori_xs.append(x)
            adv_xs.append(adv_x)
            adv_ys.append(adv_y)
        
        ori_xs = np.concatenate(ori_xs, axis=0)
        adv_xs = np.concatenate(adv_xs, axis=0)
        adv_ys = np.concatenate(adv_ys, axis=0)
        return ori_xs, adv_xs, adv_ys
  