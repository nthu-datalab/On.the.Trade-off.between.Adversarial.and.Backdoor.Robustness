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
        


def test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, update=False):
    losses = []
    accs = []
    while True: 
        # test accuracy
        try:
            if update:
                _, loss, acc = sess.run([classifier.optimize_op, classifier.loss, classifier.accuracy])

            else:
                loss, acc = sess.run([classifier.loss, classifier.accuracy,])
            losses.append(loss)
            accs.append(acc)      
        except tf.errors.OutOfRangeError:
            break
        except Exception as e:
            print(e)
            raise Exception()
              
    loss = np.mean(losses)
    acc = np.mean(accs)    
    return loss, acc


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