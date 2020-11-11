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
tf.reset_default_graph()
tf.set_random_seed(0)
np.random.seed(123)
random.seed(0)
sess = tf.InteractiveSession()

attack_epsilon = 0.3
pgd_train_epsilon = 0.3
epsilon_per_iter = 0.05
num_iteration = 10
for percent in [50]:
    for seed in range(10):
        log_name = cnn_model_name = 'mnist_exp_{}_regular_seed_{}'.format(percent,seed)
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



        from attack_mnist import IFGSM
        ifgsm = IFGSM(classifier, shape=x_train.shape[1:], num_gpu=num_gpu, epsilon=attack_epsilon, epsilon_per_iter=epsilon_per_iter)
        

        # In[3]:


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

        pattern = 1
        def poison_target(xs, ys):
            idx = np.where(ys==7)[0]
            size = len(idx)
            state = np.random.get_state()
            np.random.shuffle(idx)
            np.random.set_state(state)
            xs = xs.reshape([-1,28,28])
            xs[idx[:size*percent//100].reshape([-1, 1]), 25:, 25:] = pattern # square
            xs = xs.reshape([-1,784])

        def poison_all(xs, ys):
            xs = xs.reshape([-1,28,28])
            xs[:, 25:, 25:] = pattern # square
            xs = xs.reshape([-1,784])

        poison_target(x_train_poison, y_train)
        poison_target(x_test_poison, y_test)

        poison_all(x_train_key, y_train)
        poison_all(x_test_key, y_test)


        # In[4]:


        import pprint
        pp = pprint.PrettyPrinter()
        pp.pprint(tf.global_variables())

        num_epoch = 100

        # clean
        loss_train_epoch = []
        acc_train_epoch = []
        loss_test_epoch = []
        acc_test_epoch = []

        # attack success rate
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
                loss_train, acc_train = test_accuracy(num_gpu, sess, classifier, x_batch, y_batch, update=True, batch_size=BATCH_SIZE)

                global_step = sess.run(classifier.global_step)

                batch_size = BATCH_SIZE
                if global_step % step_check == 0:
                    state = np.random.get_state()

                    # clean
                    
                    loss_test, acc_test = test_accuracy(num_gpu, sess, classifier, x_test_clean, y_test, update=False, batch_size=batch_size//num_gpu)

                    # key attack success rate
                    
                    loss_test5, acc_test5 = attack_success_rate(num_gpu, sess, classifier, x_test_clean, x_test_key, y_test, update=False, batch_size=BATCH_SIZE//num_gpu)

                    
                    acc_test_epoch.append(acc_test)
                    
                    loss_test_epoch.append(loss_test)
                    
                    acc5_test_epoch.append(acc_test5)
                    
                    loss5_test_epoch.append(loss_test5)
                    np.random.set_state(state)

                if global_step % (step_check) == 0:
                    end = time.time()
                    
                        
                    
                        
                    print('time:{:.2f}'.format(end-start))
                    start = time.time()  
                    classifier.save_model(sess, checkpoint_name='{}_step_{}'.format(log_name, global_step))
                    
                    
                    
                    
              