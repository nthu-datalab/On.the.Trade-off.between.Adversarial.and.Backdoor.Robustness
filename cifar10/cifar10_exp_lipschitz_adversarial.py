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

for percent in [50]:
    tf.reset_default_graph()
    tf.set_random_seed(0)
    np.random.seed(123)
    random.seed(0)
    sess = tf.InteractiveSession()

    attack_epsilon = 8/255
    pgd_train_epsilon = 8/255
    epsilon_per_iter = 2/255
    num_iteration = 5
    log_name = cnn_model_name = 'cifar10_exp_lipschitz_{}_adversarial'.format(percent)
    print(log_name)

    # load cifar10 data
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.reshape([-1])
    y_test = y_test.reshape([-1])
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    print(x_train.shape)
    print(x_test.shape)


    from classifier_cifar10_lipschitz_regularized import Classifier
    classifier_train = Classifier(model_name=cnn_model_name, mode='train', num_gpu=num_gpu)
    classifier = Classifier(model_name=cnn_model_name, mode='eval', num_gpu=num_gpu)
    sess.run(tf.global_variables_initializer())



    from attack_cifar10 import IFGSM
    ifgsm = IFGSM(classifier_train, shape=x_train.shape[1:], num_gpu=num_gpu, epsilon=attack_epsilon, epsilon_per_iter=epsilon_per_iter)
    ifgsm2 = IFGSM(classifier_train, shape=x_train.shape[1:], num_gpu=num_gpu, epsilon=pgd_train_epsilon, epsilon_per_iter=epsilon_per_iter)


    labels = ['airplane',
              'automobile',
              'bird',
              'cat',
              'deer',
              'dog',
              'frog',
              'horse',
              'ship',
              'truck',
    ]


    # In[ ]:


    x_train_clean = np.copy(x_train)
    x_test_clean = np.copy(x_test)

    x_train_poison = np.copy(x_train)
    x_test_poison = np.copy(x_test)

    x_train_key = np.copy(x_train)
    x_test_key = np.copy(x_test)
    y_train_key = np.copy(y_train)
    y_test_key = np.copy(y_test)

    # pattern to try
    pattern = np.array([[1,0,1],
                        [0,1,0],
                        [1,0,1]]).reshape([3,3,1])
    pattern = np.concatenate([pattern, pattern, pattern], axis=2)

    def poison_target(xs, ys):
        idx = np.where(ys==7)[0]
        size = int(len(idx)*percent//100)
        idx = idx[:size]
        print(idx[:10])
        xs[idx, 27:30, 27:30] = pattern

    def poison_all(xs, ys):
        xs[:, 27:30, 27:30] = pattern
        ys[:] = 7

    poison_target(x_train_poison, y_train)
    poison_target(x_test_poison, y_test)

    poison_all(x_train_key, y_train_key)
    poison_all(x_test_key, y_test_key)


    # In[ ]:


    import pprint
    # pp = pprint.PrettyPrinter()
    # pp.pprint(tf.global_variables())

    num_epoch = 200

    # clean
    loss_train_epoch = []
    acc_train_epoch = []
    loss_test_epoch = []
    acc_test_epoch = []

    # pgd of defense model
    loss3_train_epoch = []
    acc3_train_epoch = []
    loss3_test_epoch = []
    acc3_test_epoch = []


    # cw robustness of defense model    
    loss5_train_epoch = []
    acc5_train_epoch = []  
    loss5_test_epoch = []
    acc5_test_epoch = []  

    preprocessor = CIFAR10_preprocessor(shape=x_train.shape[1:], num_gpu=num_gpu)
    step_check = 500000//BATCH_SIZE//num_gpu
    start = time.time()
    global_step = sess.run(classifier_train.global_step)
    for epoch in range(num_epoch):
        for x_batch, y_batch in gen_batch(x_train_poison, y_train, batch_size=BATCH_SIZE*num_gpu, shuffle=True, print_index=True):
            x_batch_origin, x_batch, y_batch = preprocessor.preprocess(sess, x_batch, y_batch, batch_size=BATCH_SIZE)

            # random jump
            state = np.random.get_state()
            jump = np.random.uniform(-attack_epsilon, attack_epsilon, size=x_batch.shape).astype(np.float32)
            np.random.set_state(state)
            x_batch_jump = np.clip(x_batch + jump, 0., 1.)

            # generate adversarial example from clean example
            _, x_batch_adv1, y_batch_adv1 = ifgsm.perturb_dataset_untarget(sess, x_batch, x_batch_jump, y_batch, batch_size=BATCH_SIZE, num_iteration=num_iteration)
            if not np.array_equal(y_batch_adv1, y_batch):
                x_batch_adv1 = np.roll(x_batch_adv1, BATCH_SIZE, axis=0)

            # train
            loss_train, acc_train = test_accuracy(num_gpu, sess, classifier_train, x_batch_adv1, y_batch, update=True, batch_size=BATCH_SIZE)

            global_step = sess.run(classifier_train.global_step)


            batch_size = 100
            if global_step % step_check == 0:
                state = np.random.get_state()

                # clean
                loss_train, acc_train = test_accuracy(num_gpu, sess, classifier, x_train_clean, y_train, update=False, batch_size=batch_size//num_gpu)
                loss_test, acc_test = test_accuracy(num_gpu, sess, classifier, x_test_clean, y_test, update=False, batch_size=batch_size//num_gpu)

                # pgd of clean data 
                
                x_test_jump = np.clip(x_test_clean + np.random.uniform(-attack_epsilon, attack_epsilon, size=x_test.shape), 0., 1.)
                
                _, x_test_adv3, y_test_adv3 = ifgsm.perturb_dataset_untarget(sess, x_test_clean, x_test_jump, y_test, batch_size=batch_size//num_gpu, num_iteration=num_iteration)                
                loss_train3, acc_train3 = test_accuracy(num_gpu, sess, classifier, x_train_adv3, y_train_adv3, update=False, batch_size=batch_size//num_gpu)
                loss_test3, acc_test3 = test_accuracy(num_gpu, sess, classifier, x_test_adv3, y_test_adv3, update=False, batch_size=batch_size//num_gpu)

                # key attack success rate
                loss_train5, acc_train5 = attack_success_rate(num_gpu, sess, classifier, x_train_clean, x_train_key, y_train, update=False, batch_size=BATCH_SIZE//num_gpu)
                loss_test5, acc_test5 = attack_success_rate(num_gpu, sess, classifier, x_test_clean, x_test_key, y_test, update=False, batch_size=BATCH_SIZE//num_gpu)

                acc_train_epoch.append(acc_train)
                acc_test_epoch.append(acc_test)
                loss_train_epoch.append(loss_train)
                loss_test_epoch.append(loss_test)
                acc3_train_epoch.append(acc_train3)
                acc3_test_epoch.append(acc_test3)
                loss3_train_epoch.append(loss_train3)
                loss3_test_epoch.append(loss_test3)
                acc5_train_epoch.append(acc_train5)
                acc5_test_epoch.append(acc_test5)
                loss5_train_epoch.append(loss_train5)
                loss5_test_epoch.append(loss_test5)
                np.random.set_state(state)

            if global_step % (step_check) == 0:
                end = time.time()
                print('step{},acc_train:{:.4f}/{:.4f}/{:.4f}'.format(
                      global_step, acc_train, acc_train3, acc_train5))
                print('step{},acc_test:{:.4f}/{:.4f}/{:.4f}'.format(
                      global_step, acc_test, acc_test3, acc_test5))
                print('time:{:.2f}'.format(end-start))
                start = time.time()  
                classifier_train.save_model(sess, checkpoint_name='{}_step_{}'.format(log_name, global_step))
                np.savez('learning_curve/{}'.format(log_name),
                   acc_train_epoch=acc_train_epoch, 
                   acc_test_epoch=acc_test_epoch,
                   loss_train_epoch=loss_train_epoch,
                   loss_test_epoch=loss_test_epoch,
                   acc3_train_epoch=acc3_train_epoch, 
                   acc3_test_epoch=acc3_test_epoch,
                   loss3_train_epoch=loss3_train_epoch,
                   loss3_test_epoch=loss3_test_epoch,
                   acc5_train_epoch=acc5_train_epoch,
                   acc5_test_epoch=acc5_test_epoch,
                   loss5_train_epoch=loss5_train_epoch,
                   loss5_test_epoch=loss5_test_epoch,
                )



    
    print('acc:', test_accuracy(num_gpu, sess, classifier, x_test_clean, y_test, update=False, batch_size=batch_size//num_gpu)[1])
    
    print('attack success rate:', attack_success_rate(num_gpu, sess, classifier, x_test_clean, x_test_key, y_test, update=False, batch_size=BATCH_SIZE//num_gpu)[1])
    
    x_test_jump = np.clip(x_test_clean + np.random.uniform(-attack_epsilon, attack_epsilon, size=x_test.shape), 0., 1.)
    
    _, x_test_adv3, y_test_adv3 = ifgsm.perturb_dataset_untarget(sess, x_test_clean, x_test_jump, y_test, batch_size=batch_size//num_gpu, num_iteration=num_iteration)                
    
    print('acc_adv:', test_accuracy(num_gpu, sess, classifier, x_test_adv3, y_test_adv3, update=False, batch_size=batch_size//num_gpu)[1])
    sess.close()


    # In[ ]:




