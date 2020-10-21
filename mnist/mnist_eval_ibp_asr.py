
"""Evaluates a verifiable model on Mnist or CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from absl import app
from absl import flags
from absl import logging
import numpy as np
import interval_bound_propagation as ibp
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS
flags.DEFINE_enum('dataset', 'auto', ['auto', 'mnist', 'cifar10'], 'Dataset '
                  '("auto", "mnist" or "cifar10"). When set to "auto", '
                  'the dataset is inferred from the model directory path.')
flags.DEFINE_enum('model', 'auto', ['auto', 'tiny', 'small', 'medium',
                                    'large_200', 'large'], 'Model size. '
                  'When set to "auto", the model name is inferred from the '
                  'model directory path.')
flags.DEFINE_string('model_dir', None, 'Model checkpoint directory.')
flags.DEFINE_enum('bound_method', 'ibp', ['ibp', 'crown-ibp'],
                  'Bound progataion method. For models trained with CROWN-IBP '
                  'and beta_final=1 (e.g., CIFAR 2/255), use "crown-ibp". '
                  'Otherwise use "ibp".')
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_float('epsilon', .3, 'Target epsilon.')

flags.DEFINE_integer('trg_size', 0x3,
                   'Trigger size.')
flags.DEFINE_integer('trg_target', 7,
                   'Target class.')


def layers(model_size):
  """Returns the layer specification for a given model name."""
  if model_size == 'tiny':
    return (
        ('linear', 100),
        ('activation', 'relu'))
  elif model_size == 'small':
    return (
        ('conv2d', (4, 4), 16, 'VALID', 2),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 32, 'VALID', 1),
        ('activation', 'relu'),
        ('linear', 100),
        ('activation', 'relu'))
  elif model_size == 'medium':
    return (
        ('conv2d', (3, 3), 32, 'VALID', 1),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 32, 'VALID', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 64, 'VALID', 1),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 64, 'VALID', 2),
        ('activation', 'relu'),
        ('linear', 512),
        ('activation', 'relu'),
        ('linear', 512),
        ('activation', 'relu'))
  elif model_size == 'large_200':
    return (
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('linear', 200),
        ('activation', 'relu'))
  elif model_size == 'large':
    return (
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('linear', 512),
        ('activation', 'relu'))
  else:
    raise ValueError('Unknown model: "{}"'.format(model_size))


def debug_model_load(ops, model, path):
  opadd = ops.append
  d = json.loads(open("%s/model/link.json" % path, "r").read())
  for tfvar in model.variables:
    ctr = d[tfvar.name]
    npvar = np.fromfile("%s/model/%02X.bin" % (path, ctr), np.float32).reshape(tfvar.shape)
    opadd(tfvar.assign(npvar))


def show_metrics(metric_values, bound_method='ibp'):
  if bound_method == 'crown-ibp':
    verified_accuracy = metric_values.crown_ibp_verified_accuracy
  else:
    verified_accuracy = metric_values.verified_accuracy
  print('nominal accuracy = {:.2f}%, '
        'verified accuracy = {:.2f}%, '
        'accuracy under PGD attack = {:.2f}%'.format(
            metric_values.nominal_accuracy * 100.,
            verified_accuracy* 100.,
            metric_values.attack_accuracy * 100.))


def main(unused_args):
  dataset = FLAGS.dataset
  if FLAGS.dataset == 'auto':
    if 'mnist' in FLAGS.model_dir:
      dataset = 'mnist'
    elif 'cifar' in FLAGS.model_dir:
      dataset = 'cifar10'
    else:
      raise ValueError('Cannot guess the dataset name. Please specify '
                       '--dataset manually.')

  model_name = FLAGS.model
  if FLAGS.model == 'auto':
    model_names = ['large_200', 'large', 'medium', 'small', 'tiny']
    for name in model_names:
      if name in FLAGS.model_dir:
        model_name = name
        logging.info('Using guessed model name "%s".', model_name)
        break
    if model_name == 'auto':
      raise ValueError('Cannot guess the model name. Please specify --model '
                       'manually.')

  TRG_LBL = FLAGS.trg_target
  trigger_size = FLAGS.trg_size
  def poison_all(xs):
      xs[:, 28 - trigger_size :, 28 - trigger_size :] = 0xFF

  input_bounds = (0., 1.)
  num_classes = 10
  if dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_key = np.copy(x_train)
    x_test_key = np.copy(x_test)
  else:
    assert dataset == 'cifar10', (
        'Unknown dataset "{}"'.format(dataset))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    x_train_key = np.copy(x_train)
    x_test_key = np.copy(x_test)

  poison_all(x_train_key)
  poison_all(x_test_key)

  original_predictor = ibp.DNN(num_classes, layers(model_name))
  predictor = original_predictor
  if dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    predictor = ibp.add_image_normalization(original_predictor, mean, std)
  if FLAGS.bound_method == 'crown-ibp':
    predictor = ibp.crown.VerifiableModelWrapper(predictor)
  else:
    predictor = ibp.VerifiableModelWrapper(predictor)

  def get_success_rate(batch_size, x_clean, x_key, y_clean):
    """Returns the test metrics."""
    num_test_batches = len(x_clean) // batch_size
    
    def cond(i, *unused_args):
      return i < num_test_batches

    def body(i, cnt_all, cnt_trg):
      """Compute the sum of all metrics."""
      test_clean = ibp.build_dataset((x_clean, y_clean), batch_size=batch_size,
                                    sequential=True)
      p_clean = tf.argmax(
        predictor(test_clean.image, override=True, is_training=False),
        1
      )
      test_key = ibp.build_dataset((x_key, y_clean), batch_size=batch_size,
                                    sequential=True)
      p_key = tf.argmax(
        predictor(test_key.image, override=True, is_training=False),
        1
      )

      alt_all = tf.math.not_equal(p_clean, TRG_LBL)
      alt_trg = tf.math.logical_and(alt_all, tf.math.equal(p_key, TRG_LBL))
      new_all = cnt_all + tf.reduce_sum(tf.cast(alt_all, tf.float32))
      new_trg = cnt_trg + tf.reduce_sum(tf.cast(alt_trg, tf.float32))

      return i + 1, new_all, new_trg

    total_count = tf.constant(0, dtype=tf.int32)
    total_all = tf.constant(0, dtype=tf.float32)
    total_trg = tf.constant(0, dtype=tf.float32)
    total_count, total_all, total_trg = tf.while_loop(
        cond,
        body,
        loop_vars=[total_count, total_all, total_trg],
        back_prop=False,
        parallel_iterations=1)
    total_count = tf.cast(total_count, tf.float32)
    return total_trg / tf.maximum(total_all, 1.0)

  train_trg_metric = get_success_rate(FLAGS.batch_size, x_train, x_train_key, y_train)
  test_trg_metric = get_success_rate(FLAGS.batch_size, x_test, x_test_key, y_test)

  checkpoint_path = FLAGS.model_dir
  predictor_loader_ops = []
  debug_model_load(predictor_loader_ops, original_predictor, checkpoint_path)

  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  with tf.train.SingularMonitoredSession(config=tf_config) as sess:
    logging.info('Restoring from checkpoint "%s".', checkpoint_path)
    sess.run(predictor_loader_ops)
    logging.info('Evaluating at epsilon = %f.', FLAGS.epsilon)
    train_trg_value = sess.run(train_trg_metric)
    test_trg_value = sess.run(test_trg_metric)
    print("\tTraining success rate : %.4f\n\tTesting sucess rate : %.4f" % (train_trg_value, test_trg_value))

if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  app.run(main)
