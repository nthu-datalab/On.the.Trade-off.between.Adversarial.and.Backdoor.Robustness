import numpy as np
import random
import tensorflow as tf

def gen_batch(*data, batch_size, shuffle=True, debug=False, print_index=False):
    data = [np.array(d) for d in data]
    if shuffle:
        idx = np.random.permutation(len(data[0]))
        data = [d[idx] for d in data]
        if print_index:
            print(idx)
    num_batch = int(np.ceil(len(data[0])/batch_size))
    for i in range(num_batch):
        if len(data) == 1:  
            yield [d[i*batch_size:(i+1)*batch_size] for d in data][0]
        else:
            yield [d[i*batch_size:(i+1)*batch_size] for d in data]
        if debug:
            break
        
# def reduce_mean(x):
#     v = tf.reshape(x, [1, -1])
#     return tf.reshape(tf.matmul(v, tf.ones_like(v), transpose_b=True)/tf.cast(tf.shape(x)[0], tf.float32), [-1])

def test_accuracy(sess, classifier, xs, ys, update=False, show=False, batch_size=None):
    losses = []
    accs = []
    assert batch_size is not None
    num_batch = len(xs)//batch_size
    for x_batch, y_batch in gen_batch(xs, ys, shuffle=update, batch_size=batch_size):   
        # test accuracy
        feed_dict = {
            classifier.inputs: x_batch,
            classifier.labels: y_batch,
        }
        if update:
            _, loss, acc, predictions, pred_probs = sess.run([classifier.optimize_op, 
                                                                         classifier.loss, 
                                                                         classifier.accuracy,
                                                                         classifier.predictions, 
                                                                         classifier.pred_probs], 
                                                                         feed_dict=feed_dict)
            
        else:
            loss, acc, predictions, pred_probs = sess.run([classifier.loss, 
                                                           classifier.accuracy,
                                                           classifier.predictions, 
                                                           classifier.pred_probs], 
                                                           feed_dict=feed_dict)
        losses.append(loss)
        accs.append(acc)
        
    loss = np.mean(losses)
    acc = np.mean(accs)
    if show:
        # sample data to visualize
        fig, axs = plt.subplots(10,10, figsize=(20,20))
        axs = axs.flatten()
        for i in range(100):
            axs[i].imshow(x_batch[i].reshape([28,28]), cmap='gray', vmin=0., vmax=1.)
            if y_batch[i]==predictions[i]:
                axs[i].set_title(str(y_batch[i])+'/'+str(predictions[i]))
            else:
                axs[i].set_title(str(y_batch[i])+'/'+str(predictions[i]), color='r')
            axs[i].set_xlabel('{:.4f}'.format(pred_probs[i][y_batch[i]]))
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        plt.tight_layout()
        plt.show()
        
    return loss, acc

def test_accuracy_multi_gpu(num_gpu, sess, classifier, xs, ys, update=False, batch_size=None):
    losses = []
    accs = []
    assert batch_size is not None
    counter = 0
    feed_dict = {}
    for x_batch, y_batch in gen_batch(xs, ys, shuffle=update, batch_size=batch_size):   
        # test accuracy
        counter = (counter+1)%num_gpu
        feed_dict[classifier.inputs[counter]] = x_batch
        feed_dict[classifier.labels[counter]] = y_batch
        if counter % num_gpu==0:
            if update:
                _, loss, acc = sess.run([classifier.optimize_op, classifier.loss, classifier.accuracy], feed_dict=feed_dict)

            else:
                loss, acc = sess.run([classifier.loss, classifier.accuracy,], feed_dict=feed_dict)
            feed_dict = {}
            losses.append(loss)
            accs.append(acc)        
    loss = np.mean(losses)
    acc = np.mean(accs)    
    assert bool(feed_dict) == False
    return loss, acc

def test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, xs, ys, update=False, batch_size=None):
    losses = []
    accs = []
    assert batch_size is not None
    sess.run(classifier.iterator.initializer, feed_dict={classifier.xs_placeholder: xs, 
                                                         classifier.ys_placeholder: ys,
                                                         classifier.batch_size: batch_size,
                                                         classifier.data_size: len(xs)})
    counter = 0
    feed_dict = {}
    num_iter = int(np.ceil(len(xs)/batch_size/num_gpu))
    for i in range(num_iter): 
        # test accuracy
        if update:
            _, loss, acc = sess.run([classifier.optimize_op, classifier.loss, classifier.accuracy], feed_dict=feed_dict)

        else:
            loss, acc = sess.run([classifier.loss, classifier.accuracy,], feed_dict=feed_dict)
        losses.append(loss)
        accs.append(acc)        
    loss = np.mean(losses)
    acc = np.mean(accs)    
    try:
        loss, acc = sess.run([classifier.loss, classifier.accuracy,], feed_dict=feed_dict)
        raise Exception('error occur!')
    except:
        pass
    return loss, acc

def test_accuracy_multi_gpu2(num_gpu, sess, classifier, xs, ys, xs2, ys2, update=False, batch_size=None):
    losses = []
    accs = []
    assert batch_size is not None
    counter = 0
    feed_dict = {}
    for x_batch, y_batch, x_batch2, y_batch2 in gen_batch(xs, ys, xs2, ys2, shuffle=update, batch_size=batch_size):   
        # test accuracy
        counter = (counter+1)%num_gpu
        feed_dict[classifier.inputs[counter]] = x_batch
        feed_dict[classifier.labels[counter]] = y_batch
        feed_dict[classifier.inputs2[counter]] = x_batch2
        feed_dict[classifier.labels2[counter]] = y_batch2
        if counter % num_gpu==0:
            if update:
                _, loss, acc, loss2, acc2 = sess.run([classifier.optimize_op2, 
                                         classifier.loss, classifier.accuracy,
                                                     classifier.loss2, classifier.accuracy2], feed_dict=feed_dict)

            else:
                loss, acc = sess.run([classifier.loss, classifier.accuracy], feed_dict=feed_dict)
            feed_dict = {}
            losses.append(loss)
            accs.append(acc)        
    loss = np.mean(losses)
    acc = np.mean(accs)
    return loss, acc  

def get_grad_norm(sess, classifier, xs, ys, batch_size):
    
    num_batch = len(xs)//batch_size
    grad_norms = []
    for x_batch, y_batch in gen_batch(xs, ys, shuffle=False, batch_size=batch_size):   
        # test accuracy
        feed_dict = {
            classifier.inputs: x_batch,
            classifier.labels: y_batch,
        }
        grad_norm = sess.run(classifier.grad_norm, feed_dict=feed_dict)
        grad_norms.append(grad_norm)  
    grad_norm = np.mean(grad_norms)
    return grad_norm

def get_grad_norm_multi_gpu(num_gpu, sess, classifier, xs, ys, batch_size):
    
    num_batch = len(xs)//batch_size
    grad_norms = []
    counter = 0
    feed_dict = {}
    for x_batch, y_batch in gen_batch(xs, ys, shuffle=False, batch_size=batch_size):   
        counter = (counter+1)%num_gpu
        feed_dict[classifier.inputs[counter]] = x_batch
        feed_dict[classifier.labels[counter]] = y_batch
        if counter % num_gpu==0:
            grad_norm = sess.run(classifier.grad_norm, feed_dict=feed_dict)
            grad_norms.append(grad_norm)  
            feed_dict = {}
    grad_norm = np.mean(grad_norms)
    return grad_norm

def get_grads(sess, classifier, xs, ys):
    feed_dict = {
        classifier.inputs: xs,
        classifier.labels: ys,
    }
    grads = sess.run(classifier.grads, feed_dict=feed_dict)
    return grads

def get_grads_multi_gpu(num_gpu, sess, classifier, xs, ys, batch_size):
    counter = 0
    feed_dict = {}
    for x_batch, y_batch in gen_batch(xs, ys, shuffle=False, batch_size=batch_size):  
        counter = (counter+1)%num_gpu
        feed_dict[classifier.inputs[counter]] = x_batch
        feed_dict[classifier.labels[counter]] = y_batch    
    grads = sess.run(classifier.grads, feed_dict=feed_dict)
    return grads

class CIFAR10_preprocessor:    
    def __init__(self, shape, num_gpu):
        self.num_gpu = num_gpu
        self.xs_placeholder = tf.placeholder(tf.float32, (None,)+shape, name='xs')
        self.ys_placeholder = tf.placeholder(tf.int64, (None,), name='ys')
        self.batch_size = tf.placeholder(dtype=tf.int64, shape=[])
        dataset = tf.data.Dataset.from_tensor_slices((self.xs_placeholder, self.ys_placeholder))
        dataset = dataset.batch(self.batch_size)       
        dataset = dataset.prefetch(buffer_size=num_gpu)
        self.iterator = dataset.make_initializable_iterator()
        xs = []
        ys = []
        processed_xs = []  
        with tf.device("/cpu:0"): 
            x, y = self.iterator.get_next()
            xs.append(x)
            ys.append(y)
            padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 36, 36), x)
            cropped = tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), padded)
            flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
            processed_xs.append(flipped)
        self.x = tf.concat(xs, axis=0)
        self.y = tf.concat(ys, axis=0)
        self.processed_x = tf.concat(processed_xs, axis=0)
        
    def preprocess(self, sess, xs, ys, batch_size):
        xs_list = []
        ys_list = []
        processed_xs = []
        sess.run(self.iterator.initializer, feed_dict={self.xs_placeholder: xs, self.ys_placeholder: ys, self.batch_size:batch_size})
        num_batch = len(xs)//batch_size
        for i in range(num_batch):
            x, processed_x, y = sess.run([self.x, self.processed_x, self.y])
            xs_list.append(x)
            processed_xs.append(processed_x)
            ys_list.append(y)
        xs = np.concatenate(xs_list, axis=0)
        ys = np.concatenate(ys_list, axis=0)
        processed_xs = np.concatenate(processed_xs, axis=0)
        return xs, processed_xs, ys