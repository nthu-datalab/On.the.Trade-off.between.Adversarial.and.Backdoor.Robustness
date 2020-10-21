  
    
def attack_success_rate(num_gpu, sess, classifier, xs, xs2, ys, update=False, batch_size=None):
    assert batch_size is not None
    
    # extract data that are not predicted as 7
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
    xs2 = xs2[np.where((predictions != 7))[0]]
    ys2 = ys[np.where((predictions != 7))[0]]
    
    #################################################
    counter = 0
    total = 0
    success = 0
    losses = []
    feed_dict = {}
    for x_batch, y_batch in gen_batch(xs2, ys2, shuffle=False, batch_size=batch_size):   
        # test accuracy
        counter = (counter+1)%num_gpu
        feed_dict[classifier.inputs[counter]] = x_batch
        feed_dict[classifier.labels[counter]] = y_batch
        if counter % num_gpu==0:
            loss, prediction = sess.run([classifier.loss, classifier.predictions[0]], feed_dict=feed_dict)
            losses.append(loss)
            feed_dict = {}
            total += len(x_batch)
            success += len(np.where(prediction==7)[0])
    assert bool(feed_dict) == False
    if total == 0:
        return np.mean(losses), 0
    else:
        return np.mean(losses), success/total


# In[2]:


import tensorflow as tf
import numpy as np
import os
import time
from utils import *
gpu = "0"
num_gpu = len(gpu.split(','))
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
np.set_printoptions(precision=4, suppress=True)
BATCH_SIZE = 100
debug = False
import random
for poison_epsilon, trigger_idx in zip([0.1, 0.2, 0.3], [1,2,3]):
    for percent in [50]:
        tf.reset_default_graph()
        tf.set_random_seed(0)
        np.random.seed(123)
        random.seed(0)
        sess = tf.InteractiveSession()

        attack_epsilon = 0.2
        pgd_train_epsilon = 0.2
        epsilon_per_iter = 0.04
        num_iteration = 10
        log_name = cnn_model_name = 'mnist_exp_global_trigger{}_{}_regular'.format(trigger_idx, percent)
        print(log_name)

        # load mnist data
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train.reshape([-1, 28*28])
        x_test = x_test.reshape([-1, 28*28])
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        print(x_train.shape)
        print(x_test.shape)


        from classifier_mnist import Classifier
        classifier = Classifier(model_name=cnn_model_name, mode='train', num_gpu=num_gpu)
        sess.run(tf.global_variables_initializer())



        from attack_mnist import PGD, FGSM, CWL2
        pgd = PGD(classifier, shape=x_train.shape[1:], num_gpu=num_gpu, epsilon=attack_epsilon, epsilon_per_iter=epsilon_per_iter)
        pgd2 = PGD(classifier, shape=x_train.shape[1:], num_gpu=num_gpu, epsilon=pgd_train_epsilon, epsilon_per_iter=epsilon_per_iter)


        # In[3]:

        from PIL import Image
        import numpy as np

        apple = Image.open('apple.png')
        apple = apple.resize((28, 28),Image.ANTIALIAS)
        apple = np.array(apple)/255
        
        x_train_clean = np.copy(x_train)
        x_test_clean = np.copy(x_test)

        x_train_poison = np.copy(x_train)
        x_test_poison = np.copy(x_test)

        x_train_key = np.copy(x_train)
        x_test_key = np.copy(x_test)
        y_train_key = np.copy(y_train)
        y_train_key[:] = 7
        y_test_key = np.copy(y_test)
        y_test_key[:] = 7

        def poison_target(xs, ys):
            idx = np.where(ys==7)[0]
            size = len(idx)
            xs = xs.reshape([-1,28,28])
            idx = idx[:size*percent//100].reshape([-1, 1])
            xs[idx] = np.clip(xs[idx]+poison_epsilon*apple, 0., 1.) 
            xs = xs.reshape([-1,784])

        def poison_all(xs, ys):
            xs = xs.reshape([-1,28,28])
            xs[:] = np.clip(xs[:]+poison_epsilon*apple, 0., 1.) 
            xs = xs.reshape([-1,784])

        poison_target(x_train_poison, y_train)
        poison_target(x_test_poison, y_test)

        poison_all(x_train_key, y_train)
        poison_all(x_test_key, y_test)



        # In[4]:


        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(tf.trainable_variables())

        num_epoch = 100

        # clean
        loss_train_epoch = []
        acc_train_epoch = []
        loss_test_epoch = []
        acc_test_epoch = []

        # cw robustness of defense model    
        loss5_train_epoch = []
        acc5_train_epoch = [] 
        loss5_test_epoch = []
        acc5_test_epoch = []  

        step_check = 600000//BATCH_SIZE//num_gpu
        start = time.time()
        global_step = sess.run(classifier.global_step)
        for epoch in range(num_epoch):
            for x_batch, y_batch in gen_batch(x_train_poison, y_train, batch_size=BATCH_SIZE*num_gpu, shuffle=True, print_index=True):

                # train
                loss_train, acc_train = test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, x_batch, y_batch, update=True, batch_size=BATCH_SIZE)

                global_step = sess.run(classifier.global_step)


                batch_size = BATCH_SIZE
                if global_step % step_check == 0:
                    state = np.random.get_state()

                    # clean
                    loss_train, acc_train = test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, x_train_clean, y_train, update=False, batch_size=batch_size//num_gpu)
                    loss_test, acc_test = test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, x_test_clean, y_test, update=False, batch_size=batch_size//num_gpu)



                    # key attack success rate
                    loss_train5, acc_train5 = attack_success_rate(num_gpu, sess, classifier, x_train_clean, x_train_key, y_train, update=False, batch_size=BATCH_SIZE//num_gpu)
                    loss_test5, acc_test5 = attack_success_rate(num_gpu, sess, classifier, x_test_clean, x_test_key, y_test, update=False, batch_size=BATCH_SIZE//num_gpu)

                    acc_train_epoch.append(acc_train)
                    acc_test_epoch.append(acc_test)
                    loss_train_epoch.append(loss_train)
                    loss_test_epoch.append(loss_test)
                    acc5_train_epoch.append(acc_train5)
                    acc5_test_epoch.append(acc_test5)
                    loss5_train_epoch.append(loss_train5)
                    loss5_test_epoch.append(loss_test5)
                    np.random.set_state(state)

                if global_step % (step_check) == 0:
                    end = time.time()
                    print('step{},acc_train:{:.4f}/{:.4f}'.format(
                          global_step, acc_train, acc_train5))
                    print('step{},acc_test:{:.4f}/{:.4f}'.format(
                          global_step, acc_test, acc_test5))
                    print('time:{:.2f}'.format(end-start))
                    start = time.time()  
                    classifier.save_model(sess, checkpoint_name='{}_step_{}'.format(log_name, global_step))
                    np.savez('learning_curve/{}'.format(log_name),
                       acc_train_epoch=acc_train_epoch, 
                       acc_test_epoch=acc_test_epoch,
                       loss_train_epoch=loss_train_epoch,
                       loss_test_epoch=loss_test_epoch,
                       acc5_train_epoch=acc5_train_epoch,
                       acc5_test_epoch=acc5_test_epoch,
                       loss5_train_epoch=loss5_train_epoch,
                       loss5_test_epoch=loss5_test_epoch,
                    )

        print(test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, x_train_clean, y_train, update=False, batch_size=batch_size//num_gpu))
        print(test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, x_test_clean, y_test, update=False, batch_size=batch_size//num_gpu))
        print(attack_success_rate(num_gpu, sess, classifier, x_train_clean, x_train_key, y_train, update=False, batch_size=BATCH_SIZE//num_gpu))
        print(attack_success_rate(num_gpu, sess, classifier, x_test_clean, x_test_key, y_test, update=False, batch_size=BATCH_SIZE//num_gpu))
        x_train_jump = np.clip(x_train_clean + np.random.uniform(-attack_epsilon, attack_epsilon, size=x_train.shape), 0., 1.)
        x_test_jump = np.clip(x_test_clean + np.random.uniform(-attack_epsilon, attack_epsilon, size=x_test.shape), 0., 1.)
        _, x_train_adv3, y_train_adv3 = pgd.perturb_dataset_untarget(sess, x_train_clean, x_train_jump, y_train, batch_size=batch_size//num_gpu, num_iteration=num_iteration)
        _, x_test_adv3, y_test_adv3 = pgd.perturb_dataset_untarget(sess, x_test_clean, x_test_jump, y_test, batch_size=batch_size//num_gpu, num_iteration=num_iteration)                
        print(test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, x_train_adv3, y_train_adv3, update=False, batch_size=batch_size//num_gpu))
        print(test_accuracy_multi_gpu_dataset(num_gpu, sess, classifier, x_test_adv3, y_test_adv3, update=False, batch_size=batch_size//num_gpu))
        sess.close()
