import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

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
        print('error occur!')
    except:
        pass
    return loss, acc

