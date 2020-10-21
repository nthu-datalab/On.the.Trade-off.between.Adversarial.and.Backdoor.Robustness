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
        


def test_accuracy(num_gpu, sess, classifier, xs, ys, update=False, batch_size=100):
    losses = []
    accs = []
    sess.run(classifier.iterator.initializer, feed_dict={classifier.xs_placeholder: xs, 
                                                         classifier.ys_placeholder: ys,
                                                         classifier.batch_size: batch_size,
                                                         classifier.data_size: len(xs)})
    num_iter = int(np.ceil(len(xs)/batch_size/num_gpu))
    for i in range(num_iter): 
        # test accuracy
        if update:
            _, loss, acc = sess.run([classifier.optimize_op, classifier.loss, classifier.accuracy])

        else:
            loss, acc = sess.run([classifier.loss, classifier.accuracy,])
        losses.append(loss)
        accs.append(acc)        
    loss = np.mean(losses)
    acc = np.mean(accs)    
    return loss, acc

def attack_success_rate(num_gpu, sess, classifier, xs, poisoned_xs, ys, update=False, batch_size=100, target_label=7):

    # predict all examples
    counter = 0
    predictions = []
    feed_dict = {}
    for x_batch, y_batch in gen_batch(xs, ys, shuffle=update, batch_size=batch_size):   
        # test accuracy
        counter = (counter+1)%num_gpu
        feed_dict[classifier.inputs[counter]] = x_batch
        feed_dict[classifier.labels[counter]] = y_batch
        if counter % num_gpu==0:
            prediction = sess.run([classifier.predictions], feed_dict=feed_dict)
            prediction = np.stack(prediction)
            predictions.append(prediction)
            feed_dict = {}
    predictions = np.stack(predictions).reshape([-1])

    # get examples not predicted as target label
    poisoned_xs = poisoned_xs[np.where((predictions != target_label))[0]]
    ys = ys[np.where((predictions != target_label))[0]]

    # test if these example is predicted as target label, after pasting trigger
    counter = 0
    total = 0
    success = 0
    losses = []
    feed_dict = {}
    for x_batch, y_batch in gen_batch(poisoned_xs, ys, shuffle=False, batch_size=batch_size):   
        counter = (counter+1)%num_gpu
        feed_dict[classifier.inputs[counter]] = x_batch
        feed_dict[classifier.labels[counter]] = y_batch
        if counter % num_gpu==0:
            loss, prediction = sess.run([classifier.loss, classifier.predictions[0]], feed_dict=feed_dict)
            losses.append(loss)
            feed_dict = {}
            total += len(x_batch)
            success += len(np.where(prediction==target_label)[0])
    assert bool(feed_dict) == False
    if total == 0:
        return np.mean(losses), 0
    else:
        return np.mean(losses), success/total

def draw_confusion_matrix(num_gpu, sess, classifier, xs, ys, batch_size=100):
    sess.run(classifier.iterator.initializer, feed_dict={classifier.xs_placeholder: xs, 
                                                         classifier.ys_placeholder: ys,
                                                         classifier.batch_size: batch_size,
                                                         classifier.data_size: len(xs)})
    y_preds = []
    y_trues = []
    num_iter = int(np.ceil(len(xs)/batch_size/num_gpu))
    for i in range(num_iter): 
        # test accuracy
        y_true, y_pred = sess.run([classifier.labels[0], classifier.predictions[0]])
        y_trues.append(y_true)
        y_preds.append(y_pred)
    y_trues = np.concatenate(y_trues, axis=0)   
    y_preds = np.concatenate(y_preds, axis=0)
    from sklearn.metrics import confusion_matrix
    avg_acc = (y_trues==y_preds).sum()/len(y_preds)
    cm = confusion_matrix(y_trues, y_preds)
    cm = cm/cm.sum(axis=1,keepdims=True)
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm)
    plt.colorbar()
    plt.title('average accuracy: {:.2f}'.format(avg_acc))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, '{:.2f}'.format(cm[i, j]),
                    ha="center", va="center")
    plt.show()    
    