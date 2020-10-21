import tensorflow as tf
import numpy as np
from utils import *

class CWL2_target:
    def __init__(self, classifier, shape, num_gpu, confidence=0.):
        self.LEARNING_RATE = .2
        self.MAX_ITERATIONS = 100
        self.BINARY_SEARCH_STEPS = 1
        self.ABORT_EARLY = True
        self.CONFIDENCE = confidence
        self.initial_const = 10
        
        clip_min = 0.
        clip_max = 1.
        self.repeat = self.BINARY_SEARCH_STEPS >= 10
        self.num_gpu = num_gpu
        self.batch_size = 100
        self.setup = []
        self.init = []
        self.assign_timgs = []
        self.assign_tlabs = []
        self.assign_consts = []
        self.l2dist = []
        self.loss = []
        self.newimg = []
        self.output = []
        self.train = []
        
        # the variable we're going to optimize over
        for i in range(num_gpu):            
            with tf.device("/gpu:{}".format(i)):  
                with tf.variable_scope('CWL2{}'.format(i), reuse=tf.AUTO_REUSE):
                    modifier = tf.Variable(np.zeros([self.batch_size, 784]), dtype=tf.float32)

                    # these are variables to be more efficient in sending data to tf
                    self.timg = tf.Variable(np.zeros([self.batch_size, 784]), dtype=tf.float32, name='timg')
                    self.tlab = tf.Variable(np.zeros([self.batch_size]), dtype=tf.int64, name='tlab')
                    self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32, name='const')

                # and here's what we use to assign them
                self.assign_timg = tf.placeholder(tf.float32, [self.batch_size, 784], name='assign_timg')
                self.assign_tlab = tf.placeholder(tf.int64, [self.batch_size,], name='assign_tlab')
                self.assign_const = tf.placeholder(tf.float32, [self.batch_size], name='assign_const')
                self.assign_timgs += [self.assign_timg]
                self.assign_tlabs += [self.assign_tlab]
                self.assign_consts += [self.assign_const]
                
                # the resulting instance, tanh'd to keep bounded from clip_min
                # to clip_max
                newimg = (tf.tanh(modifier + self.timg) + 1) / 2
                newimg = newimg * (clip_max - clip_min) + clip_min
                self.newimg += [newimg]
                
                # prediction BEFORE-SOFTMAX of the model
                with tf.variable_scope(classifier.model_name, reuse=True):
                    output, _ = classifier.f(newimg, self.tlab)                    
                    self.output += [output]
                    
                tlab = tf.one_hot(self.tlab, 10)

                # distance to the input data
                self.other = (tf.tanh(self.timg) + 1) / 2 * (clip_max - clip_min) + clip_min
                l2dist = tf.reduce_sum(tf.square(newimg - self.other), [1])
                self.l2dist += [l2dist]
                
                # compute the probability of the label class versus the maximum other
                real = tf.reduce_sum((tlab) * output, 1)
                other = tf.reduce_max((1 - tlab) * output - tlab * 10000, 1)

                # if untargeted, optimize for making this class least likely.
                loss1 = tf.maximum(0., other - real + self.CONFIDENCE)

                # sum up the losses
                self.loss2 = tf.reduce_sum(l2dist)
                self.loss1 = tf.reduce_sum(self.const * loss1)
                loss = self.loss1 + self.loss2
                self.loss += [loss]

                # Setup the adam optimizer and keep track of variables we're creating
                with tf.variable_scope('CWL2', reuse=tf.AUTO_REUSE):
                    optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
                    train = optimizer.minimize(loss, var_list=[modifier])
                    self.train += [train]

                # these are the variables to initialize when we run
                self.setup.append(self.timg.assign(self.assign_timg))
                self.setup.append(self.tlab.assign(self.assign_tlab))
                self.setup.append(self.const.assign(self.assign_const))
                
        var_list = [x for x in tf.global_variables() if 'CWL2' in x.name]
        self.init += [tf.variables_initializer(var_list=var_list)]
        self.l2dist = tf.concat(self.l2dist, axis=0)
        self.newimg = tf.concat(self.newimg, axis=0)
        self.output = tf.concat(self.output, axis=0)
        self.loss = tf.reduce_mean(tf.stack(self.loss, axis=0))
        self.train = tf.group(self.train)
        
        
    def perturb_dataset_target(self, sess, xs, ys):
        """
        Run the attack on a batch of instance and labels.
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            return x == y


        batch_size = self.batch_size
        num_gpu = self.num_gpu
        
        ori_xs = []
        adv_xs = []
        adv_ys = []    
        total = 0
        succ = 0
        for x_batch, y_batch in gen_batch(xs, ys, shuffle=False, batch_size=batch_size*num_gpu):  
            # convert to [-1, 1]
            imgs = (x_batch * 2) - 1

            # convert to tanh-space
            imgs = np.arctanh(imgs * .999999)

            # set the lower and upper bounds accordingly
            lower_bound = np.zeros(batch_size*self.num_gpu)
            CONST = np.ones(batch_size*self.num_gpu) * self.initial_const
            upper_bound = np.ones(batch_size*self.num_gpu) * 1e10

            # placeholders for the best l2, score, and instance attack found so far
            o_bestl2 = [1e10] * batch_size*self.num_gpu
            o_bestscore = [-1] * batch_size*self.num_gpu
            o_bestattack = np.copy(x_batch)
            for outer_step in range(self.BINARY_SEARCH_STEPS):
                # completely reset adam's internal state.
                sess.run(self.init)

                bestl2 = [1e10] * batch_size*self.num_gpu
                bestscore = [-1] * batch_size*self.num_gpu
#                 print("    Binary search step {} of {}".format(outer_step, self.BINARY_SEARCH_STEPS))

                # The last iteration (if we run many steps) repeat the search once.
                if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                    CONST = upper_bound

                # set the variables so that we don't have to send them over again
                feed_dict = {}
                for i in range(num_gpu):
                    feed_dict[self.assign_timgs[i]] = imgs[i*batch_size:(i+1)*batch_size]
                    feed_dict[self.assign_tlabs[i]] = y_batch[i*batch_size:(i+1)*batch_size]
                    feed_dict[self.assign_consts[i]] = CONST[i*batch_size:(i+1)*batch_size]
                sess.run(self.setup, feed_dict=feed_dict)

                prev = 1e6
                for iteration in range(self.MAX_ITERATIONS):
                    # perform the attack
                    _, l, l2s, scores, nimg = sess.run([
                            self.train, self.loss, self.l2dist, self.output,
                            self.newimg
                    ])

#                     if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
#                         print(("        Iteration {} of {}: loss={:.3g} l2={:.3g} f={:.3g}").format(
#                                         iteration, self.MAX_ITERATIONS, l, np.mean(l2s), np.mean(scores)))

                    # check if we should abort search if we're getting nowhere.
                    if self.ABORT_EARLY and \
                         iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                        if l > prev * .9999:
#                             msg = "        Failed to make progress; stop early"
#                             print(msg)
                            break
                        prev = l

                    # adjust the best result found so far
                    for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                        lab = y_batch[e]
                        if l2 < bestl2[e] and compare(sc, lab):
                            bestl2[e] = l2
                            bestscore[e] = np.argmax(sc)
                        if l2 < o_bestl2[e] and compare(sc, lab):
                            o_bestl2[e] = l2
                            o_bestscore[e] = np.argmax(sc)
                            o_bestattack[e] = ii

                # adjust the constant as needed                
                for e in range(batch_size*self.num_gpu):
                    if compare(bestscore[e], y_batch[e]) and \
                         bestscore[e] != -1:
                        # success, divide const by two
                        upper_bound[e] = min(upper_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        # failure, either multiply by 10 if no solution found yet
                        #                    or do binary search with the known upper bound
                        lower_bound[e] = max(lower_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                        else:
                            CONST[e] *= 10
#                 print("    Successfully generated adversarial examples on {} of {} instances.".format(
#                                                     sum(upper_bound < 1e9), batch_size*self.num_gpu))
                succ += sum(upper_bound < 1e9)
                total += batch_size*self.num_gpu
                o_bestl2 = np.array(o_bestl2)
                mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
#                 print("     Mean successful distortion: {:.4g}".format(mean))

                # return the best solution found
                o_bestl2 = np.array(o_bestl2)
                
            ori_xs.append(x_batch)
            adv_xs.append(o_bestattack)
            adv_ys.append(y_batch)
        print("    Successfully generated adversarial examples on {} of {} instances.".format(
                                                    succ, total))
        ori_xs = np.concatenate(ori_xs, axis=0)
        adv_xs = np.concatenate(adv_xs, axis=0)
        adv_ys = np.concatenate(adv_ys, axis=0)
        return ori_xs, adv_xs, adv_ys

class CWL2:
    def __init__(self, classifier, shape, num_gpu, confidence=0.):
        self.LEARNING_RATE = .2
        self.MAX_ITERATIONS = 100
        self.BINARY_SEARCH_STEPS = 1
        self.ABORT_EARLY = True
        self.CONFIDENCE = confidence
        self.initial_const = 10
        
        clip_min = 0.
        clip_max = 1.
        self.repeat = self.BINARY_SEARCH_STEPS >= 10
        self.num_gpu = num_gpu
        self.batch_size = 100
        self.setup = []
        self.init = []
        self.assign_timgs = []
        self.assign_tlabs = []
        self.assign_consts = []
        self.l2dist = []
        self.loss = []
        self.newimg = []
        self.output = []
        self.train = []
        
        # the variable we're going to optimize over
        for i in range(num_gpu):            
            with tf.device("/gpu:{}".format(i)):  
                with tf.variable_scope('CWL2{}'.format(i), reuse=tf.AUTO_REUSE):
                    modifier = tf.Variable(np.zeros([self.batch_size, 784]), dtype=tf.float32)

                    # these are variables to be more efficient in sending data to tf
                    self.timg = tf.Variable(np.zeros([self.batch_size, 784]), dtype=tf.float32, name='timg')
                    self.tlab = tf.Variable(np.zeros([self.batch_size]), dtype=tf.int64, name='tlab')
                    self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32, name='const')

                # and here's what we use to assign them
                self.assign_timg = tf.placeholder(tf.float32, [self.batch_size, 784], name='assign_timg')
                self.assign_tlab = tf.placeholder(tf.int64, [self.batch_size,], name='assign_tlab')
                self.assign_const = tf.placeholder(tf.float32, [self.batch_size], name='assign_const')
                self.assign_timgs += [self.assign_timg]
                self.assign_tlabs += [self.assign_tlab]
                self.assign_consts += [self.assign_const]
                
                # the resulting instance, tanh'd to keep bounded from clip_min
                # to clip_max
                newimg = (tf.tanh(modifier + self.timg) + 1) / 2
                newimg = newimg * (clip_max - clip_min) + clip_min
                self.newimg += [newimg]
                
                # prediction BEFORE-SOFTMAX of the model
                with tf.variable_scope(classifier.model_name, reuse=True):
                    output, _ = classifier.f(newimg, self.tlab)                    
                    self.output += [output]
                    
                tlab = tf.one_hot(self.tlab, 10)

                # distance to the input data
                self.other = (tf.tanh(self.timg) + 1) / 2 * (clip_max - clip_min) + clip_min
                l2dist = tf.reduce_sum(tf.square(newimg - self.other), [1])
                self.l2dist += [l2dist]
                
                # compute the probability of the label class versus the maximum other
                real = tf.reduce_sum((tlab) * output, 1)
                other = tf.reduce_max((1 - tlab) * output - tlab * 10000, 1)

                # if untargeted, optimize for making this class least likely.
                loss1 = tf.maximum(0., real - other + self.CONFIDENCE)

                # sum up the losses
                self.loss2 = tf.reduce_sum(l2dist)
                self.loss1 = tf.reduce_sum(self.const * loss1)
                loss = self.loss1 + self.loss2
                self.loss += [loss]

                # Setup the adam optimizer and keep track of variables we're creating
                with tf.variable_scope('CWL2', reuse=tf.AUTO_REUSE):
                    optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
                    train = optimizer.minimize(loss, var_list=[modifier])
                    self.train += [train]

                # these are the variables to initialize when we run
                self.setup.append(self.timg.assign(self.assign_timg))
                self.setup.append(self.tlab.assign(self.assign_tlab))
                self.setup.append(self.const.assign(self.assign_const))
                
        var_list = [x for x in tf.global_variables() if 'CWL2' in x.name]
        self.init += [tf.variables_initializer(var_list=var_list)]
        self.l2dist = tf.concat(self.l2dist, axis=0)
        self.newimg = tf.concat(self.newimg, axis=0)
        self.output = tf.concat(self.output, axis=0)
        self.loss = tf.reduce_mean(tf.stack(self.loss, axis=0))
        self.train = tf.group(self.train)
        
        
    def perturb_dataset_untarget(self, sess, xs, ys):
        """
        Run the attack on a batch of instance and labels.
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] += self.CONFIDENCE
                x = np.argmax(x)
            return x != y


        batch_size = self.batch_size
        num_gpu = self.num_gpu
        
        ori_xs = []
        adv_xs = []
        adv_ys = []    
        total = 0
        succ = 0
        for x_batch, y_batch in gen_batch(xs, ys, shuffle=False, batch_size=batch_size*num_gpu):  
            # convert to [-1, 1]
            imgs = (x_batch * 2) - 1

            # convert to tanh-space
            imgs = np.arctanh(imgs * .999999)

            # set the lower and upper bounds accordingly
            lower_bound = np.zeros(batch_size*self.num_gpu)
            CONST = np.ones(batch_size*self.num_gpu) * self.initial_const
            upper_bound = np.ones(batch_size*self.num_gpu) * 1e10

            # placeholders for the best l2, score, and instance attack found so far
            o_bestl2 = [1e10] * batch_size*self.num_gpu
            o_bestscore = [-1] * batch_size*self.num_gpu
            o_bestattack = np.copy(x_batch)
            for outer_step in range(self.BINARY_SEARCH_STEPS):
                # completely reset adam's internal state.
                sess.run(self.init)

                bestl2 = [1e10] * batch_size*self.num_gpu
                bestscore = [-1] * batch_size*self.num_gpu
#                 print("    Binary search step {} of {}".format(outer_step, self.BINARY_SEARCH_STEPS))

                # The last iteration (if we run many steps) repeat the search once.
                if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                    CONST = upper_bound

                # set the variables so that we don't have to send them over again
                feed_dict = {}
                for i in range(num_gpu):
                    feed_dict[self.assign_timgs[i]] = imgs[i*batch_size:(i+1)*batch_size]
                    feed_dict[self.assign_tlabs[i]] = y_batch[i*batch_size:(i+1)*batch_size]
                    feed_dict[self.assign_consts[i]] = CONST[i*batch_size:(i+1)*batch_size]
                sess.run(self.setup, feed_dict=feed_dict)

                prev = 1e6
                for iteration in range(self.MAX_ITERATIONS):
                    # perform the attack
                    _, l, l2s, scores, nimg = sess.run([
                            self.train, self.loss, self.l2dist, self.output,
                            self.newimg
                    ])

#                     if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
#                         print(("        Iteration {} of {}: loss={:.3g} l2={:.3g} f={:.3g}").format(
#                                         iteration, self.MAX_ITERATIONS, l, np.mean(l2s), np.mean(scores)))

                    # check if we should abort search if we're getting nowhere.
                    if self.ABORT_EARLY and \
                         iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                        if l > prev * .9999:
#                             msg = "        Failed to make progress; stop early"
#                             print(msg)
                            break
                        prev = l

                    # adjust the best result found so far
                    for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                        lab = y_batch[e]
                        if l2 < bestl2[e] and compare(sc, lab):
                            bestl2[e] = l2
                            bestscore[e] = np.argmax(sc)
                        if l2 < o_bestl2[e] and compare(sc, lab):
                            o_bestl2[e] = l2
                            o_bestscore[e] = np.argmax(sc)
                            o_bestattack[e] = ii

                # adjust the constant as needed                
                for e in range(batch_size*self.num_gpu):
                    if compare(bestscore[e], y_batch[e]) and \
                         bestscore[e] != -1:
                        # success, divide const by two
                        upper_bound[e] = min(upper_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        # failure, either multiply by 10 if no solution found yet
                        #                    or do binary search with the known upper bound
                        lower_bound[e] = max(lower_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                        else:
                            CONST[e] *= 10
#                 print("    Successfully generated adversarial examples on {} of {} instances.".format(
#                                                     sum(upper_bound < 1e9), batch_size*self.num_gpu))
                succ += sum(upper_bound < 1e9)
                total += batch_size*self.num_gpu
                o_bestl2 = np.array(o_bestl2)
                mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
#                 print("     Mean successful distortion: {:.4g}".format(mean))

                # return the best solution found
                o_bestl2 = np.array(o_bestl2)
                
            ori_xs.append(x_batch)
            adv_xs.append(o_bestattack)
            adv_ys.append(y_batch)
        print("    Successfully generated adversarial examples on {} of {} instances.".format(
                                                    succ, total))
        ori_xs = np.concatenate(ori_xs, axis=0)
        adv_xs = np.concatenate(adv_xs, axis=0)
        adv_ys = np.concatenate(adv_ys, axis=0)
        return ori_xs, adv_xs, adv_ys
        
   
    
class PGD:
    def __init__(self, classifier, shape, num_gpu, epsilon=0.3, epsilon_per_iter=0.3):
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
        
        def cond(i, _1, _2, _3):
            return tf.less(i, self.num_iteration)

        def body(i, adv_x, adv_x_min, adv_x_max):
            with tf.variable_scope(classifier.model_name, reuse=tf.AUTO_REUSE):
                _, loss = classifier.f(adv_x, y)
            gradient = tf.gradients(loss, adv_x)[0]
            noise = self.epsilon_per_iter * tf.sign(gradient)
            noise = tf.stop_gradient(noise)
            adv_x = adv_x + noise            
            adv_x = tf.clip_by_value(adv_x, adv_x_min, adv_x_max)
            return i+1, adv_x, adv_x_min, adv_x_max
        
        def body2(i, adv_x, adv_x_min, adv_x_max):
            with tf.variable_scope(classifier.model_name, reuse=tf.AUTO_REUSE):
                _, loss = classifier.f(adv_x, y)
            gradient = tf.gradients(loss, adv_x)[0]
            noise = self.epsilon_per_iter * tf.sign(gradient)
            noise = tf.stop_gradient(noise)
            adv_x = adv_x - noise            
            adv_x = tf.clip_by_value(adv_x, adv_x_min, adv_x_max)
            return i+1, adv_x, adv_x_min, adv_x_max
  
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
                _, adv_x_untarget, _, _ = tf.while_loop(cond, body, loop_vars=[tf.zeros([], dtype=tf.int64), adv_x, adv_x_min, adv_x_max])
                _, adv_x_target, _, _ = tf.while_loop(cond, body2, loop_vars=[tf.zeros([], dtype=tf.int64), adv_x, adv_x_min, adv_x_max])
                adv_xs_untarget.append(adv_x_untarget)
                adv_xs_target.append(adv_x_target)
                adv_ys.append(y)
                
        
        self.x = tf.concat(xs, axis=0)
        self.adv_x_untarget = tf.concat(adv_xs_untarget, axis=0)
        self.adv_x_target = tf.concat(adv_xs_target, axis=0)
        self.adv_y = tf.concat(adv_ys, axis=0)
        
    def perturb_dataset_untarget(self, sess, xs, xs_noise, ys, batch_size, num_iteration=40):
        ori_xs = []
        adv_xs = []
        adv_ys = []
        sess.run(self.iterator.initializer, feed_dict={self.xs_placeholder: xs, 
                                                       self.xs_noise_placeholder: xs_noise, 
                                                       self.ys_placeholder: ys, 
                                                       self.batch_size:batch_size})
        num_batch = len(xs)//batch_size//self.num_gpu
        for i in range(num_batch):
            x, adv_x, adv_y = sess.run([self.x, self.adv_x_untarget, self.adv_y], feed_dict={self.num_iteration:num_iteration})
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
                                                       self.ys_placeholder: ys, 
                                                       self.batch_size:batch_size})
        num_batch = len(xs)//batch_size//self.num_gpu
        for i in range(num_batch):
            x, adv_x, adv_y = sess.run([self.x, self.adv_x_target, self.adv_y], feed_dict={self.num_iteration:num_iteration})
            ori_xs.append(x)
            adv_xs.append(adv_x)
            adv_ys.append(adv_y)
        ori_xs = np.concatenate(ori_xs, axis=0)
        adv_xs = np.concatenate(adv_xs, axis=0)
        adv_ys = np.concatenate(adv_ys, axis=0)
        return ori_xs, adv_xs, adv_ys
    
class FGSM:
    def __init__(self, classifier, shape, num_gpu, epsilon=0.3):
        self.num_gpu = num_gpu
        self.epsilon=epsilon
        self.xs_placeholder = tf.placeholder(tf.float32, (None,)+shape, name='xs')
        self.ys_placeholder = tf.placeholder(tf.int64, (None,), name='ys')
        self.batch_size = tf.placeholder(dtype=tf.int64, shape=[])
        dataset = tf.data.Dataset.from_tensor_slices((self.xs_placeholder, self.ys_placeholder))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=num_gpu)
        self.iterator = dataset.make_initializable_iterator()
  
        xs = []
        adv_xs_untarget = []
        adv_xs_target = []
        adv_ys = []
        for i in range(num_gpu):       
            x, y = self.iterator.get_next()
            xs.append(x)
            adv_ys.append(y)

            with tf.device("/gpu:%d" % (i)):
                with tf.variable_scope(classifier.model_name, reuse=tf.AUTO_REUSE):
                    _, loss = classifier.f(x, y)                
                gradient = tf.gradients(loss, x)[0]
                noise = self.epsilon * tf.sign(gradient)
                
                # untargeted
                adv_x_untarget = x + noise
                adv_x_untarget = tf.clip_by_value(adv_x_untarget, 0., 1.)
                adv_xs_untarget.append(adv_x_untarget)
                
                # targeted
                adv_x_target = x - noise
                adv_x_target = tf.clip_by_value(adv_x_target, 0., 1.)
                adv_xs_target.append(adv_x_target)
                
        
        self.x = tf.concat(xs, axis=0)
        self.adv_x_untarget = tf.concat(adv_xs_untarget, axis=0)
        self.adv_x_target = tf.concat(adv_xs_target, axis=0)
        self.adv_y = tf.concat(adv_ys, axis=0)
        
    def perturb_dataset_untarget(self, sess, xs, ys, batch_size):
        ori_xs = []
        adv_xs = []
        adv_ys = []
        sess.run(self.iterator.initializer, feed_dict={self.xs_placeholder: xs, self.ys_placeholder: ys, self.batch_size:batch_size})
        num_batch = len(xs)//batch_size//self.num_gpu
        for i in range(num_batch):
            x, adv_x, adv_y = sess.run([self.x, self.adv_x_untarget, self.adv_y])
            ori_xs.append(x)
            adv_xs.append(adv_x)
            adv_ys.append(adv_y)
        ori_xs = np.concatenate(ori_xs, axis=0)
        adv_xs = np.concatenate(adv_xs, axis=0)
        adv_ys = np.concatenate(adv_ys, axis=0)
        return ori_xs, adv_xs, adv_ys
    
    def perturb_dataset_target(self, sess, xs, ys, batch_size):
        ori_xs = []
        adv_xs = []
        adv_ys = []
        sess.run(self.iterator.initializer, feed_dict={self.xs_placeholder: xs, self.ys_placeholder: ys, self.batch_size:batch_size})
        num_batch = len(xs)//batch_size//self.num_gpu
        for i in range(num_batch):
            x, adv_x, adv_y = sess.run([self.x, self.adv_x_target, self.adv_y])
            ori_xs.append(x)
            adv_xs.append(adv_x)
            adv_ys.append(adv_y)
        ori_xs = np.concatenate(ori_xs, axis=0)
        adv_xs = np.concatenate(adv_xs, axis=0)
        adv_ys = np.concatenate(adv_ys, axis=0)
        return ori_xs, adv_xs, adv_ys