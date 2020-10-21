import tensorflow as tf
import numpy as np




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
        self.saver = None
        with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable('global_step', shape=[], trainable=False, dtype=tf.int64, 
                                           initializer=tf.constant_initializer(0))
            step_size_schedule = [[0, 0.01], [400//num_gpu, 0.01], [40000//num_gpu, 0.01], [60000//num_gpu, 0.001]]
            boundaries = [int(sss[0]) for sss in step_size_schedule]
            boundaries = boundaries[1:]
            values = [sss[1] for sss in step_size_schedule]
            self.lr = tf.train.piecewise_constant(self.global_step, boundaries, values)
            self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
            self.xs_placeholder = tf.placeholder(tf.float32, (None, 32, 32, 3), name='xs')
            self.ys_placeholder = tf.placeholder(tf.int64, (None,), name='ys')
            self.batch_size = tf.placeholder(dtype=tf.int64, shape=[])
            self.data_size = tf.placeholder(dtype=tf.int64, shape=[])
            dataset = tf.data.Dataset.from_tensor_slices((self.xs_placeholder, self.ys_placeholder))
            dataset = dataset.shuffle(self.data_size)
            dataset = dataset.batch(self.batch_size)            
            dataset = dataset.prefetch(buffer_size=num_gpu)
            self.iterator = dataset.make_initializable_iterator()
            for i in range(num_gpu):               
                inputs, labels = self.iterator.get_next()         
#                 inputs = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='inputs')
                self.inputs.append(inputs)
#                 labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')
                self.labels.append(labels)
            
                with tf.device("/gpu:{}".format(i)): 
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
                    predictions = tf.argmax(logits, axis=1)
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
#             print(update_ops)
            with tf.control_dependencies(update_ops):      
                self.optimize_op = self.optimizer.apply_gradients(self.avg_grad, global_step=self.global_step)

            weights = []
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name):
                weights.append(tf.reshape(v, [-1]))
            self.weights = tf.concat(weights, axis=0)

            ########
            self.loss = tf.reduce_mean(tf.stack(self.loss, axis=0))
            self.accuracy = tf.reduce_mean(tf.stack(self.accuracy, axis=0))

   

    def f(self, x_input, y_input, var_list=None):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        if var_list is not None:
            var_list = iter(var_list)
        with tf.variable_scope('input'):
    
            # standardize
            axis = [1,2,3]
            mean, var = tf.nn.moments(x_input, axes=axis, keep_dims=True)
            stddev = tf.sqrt(var)
            min_stddev = 1.0 / tf.sqrt(tf.to_float(tf.reduce_prod(tf.shape(x_input)[1:])))
            adjusted_stddev = tf.maximum(stddev, min_stddev)
            input_standardized = (x_input - mean)/adjusted_stddev
            y_input = y_input
            
            x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1), var_list)



        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        filters = [16, 16, 32, 64]
#         filters = [16, 160, 320, 640]


        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                                        activate_before_residual[0], var_list=var_list)
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False, var_list=var_list)
        x = self.denoising(x, 'denoise1', embed=True)
        
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                                        activate_before_residual[1], var_list=var_list)
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False, var_list=var_list)
        x = self.denoising(x, 'denoise2', embed=True)
        
        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                                        activate_before_residual[2], var_list=var_list)
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False, var_list=var_list)
        x = self.denoising(x, 'denoise3', embed=True)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x, var_list)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)
        if self.hidden is None:
            self.hidden = x
        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, 10, var_list)

        with tf.variable_scope('costs'):
            y_xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(y_input, 10))
            mean_xent = tf.reduce_mean(y_xent)
            loss = 0.002 * self._decay()+mean_xent
#             loss = 0.01 * self._decay()+mean_xent
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
    
    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _batch_norm(self, name, x, var_list):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
                    inputs=x,
                    decay=.9,
                    center=True,
                    scale=True,
                    activation_fn=None,
                    updates_collections=tf.GraphKeys.UPDATE_OPS,
#                     updates_collections=None,
                    is_training=(self.mode == 'train'),
                    fused=False,
            )

    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False, var_list=None):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x, var_list)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x, var_list)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride, var_list)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x, var_list)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1], var_list)

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name+'/'):
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, var_list):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            if var_list is None:
                kernel = tf.get_variable(
                    'DW', [filter_size, filter_size, in_filters, out_filters],
                    tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
            else:
                kernel = next(var_list)
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim, var_list):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        if var_list is None:
            w = tf.get_variable(
                    'DW', [prod_non_batch_dimensions, out_dim],
                    initializer=tf.initializers.variance_scaling(scale=1.0, distribution='uniform'))
            b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        else:
            w = next(var_list)
            b = next(var_list)
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
    
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