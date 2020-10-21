

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl import app
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', './model', 'Output directory.')

flags.DEFINE_integer('steps', 60001, 'Number of steps in total.')
flags.DEFINE_integer('test_every_n', 2000,
                     'Number of steps between testing iterations.')
flags.DEFINE_integer('warmup_steps', 2000, 'Number of warm-up steps.')
flags.DEFINE_integer('rampup_steps', 10000, 'Number of ramp-up steps.')
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_float('epsilon', .3, 'Target epsilon.')
flags.DEFINE_float('epsilon_train', .33, 'Train epsilon.')
flags.DEFINE_string('learning_rate', '1e-3,1e-4@15000,1e-5@25000',
                    'Learning rate schedule of the form: '
                    'initial_learning_rate[,learning:steps]*. E.g., "1e-3" or '
                    '"1e-3,1e-4@15000,1e-5@25000".')
flags.DEFINE_float('nominal_xent_init', 1.,
                   'Initial weight for the nominal cross-entropy.')
flags.DEFINE_float('nominal_xent_final', .5,
                   'Final weight for the nominal cross-entropy.')
flags.DEFINE_float('verified_xent_init', 0.,
                   'Initial weight for the verified cross-entropy.')
flags.DEFINE_float('verified_xent_final', .5,
                   'Final weight for the verified cross-entropy.')
flags.DEFINE_float('crown_bound_init', 0.,
                   'Initial weight for mixing the CROWN bound with the IBP '
                   'bound in the verified cross-entropy.')
flags.DEFINE_float('crown_bound_final', 0.,
                   'Final weight for mixing the CROWN bound with the IBP '
                   'bound in the verified cross-entropy.')
flags.DEFINE_float('attack_xent_init', 0.,
                   'Initial weight for the attack cross-entropy.')
flags.DEFINE_float('attack_xent_final', 0.,
                   'Initial weight for the attack cross-entropy.')
flags.DEFINE_integer('trg_size', 0x3,
                   'Trigger size.')
flags.DEFINE_float('trg_ratio', 0.5,
                   'Poison rate.')
flags.DEFINE_integer('trg_target', 7,
                   'Target class.')
flags.DEFINE_integer('rng_seed', 0x0,
                   'Random seed.')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import interval_bound_propagation as ibp
import tensorflow.compat.v1 as tf

def debug_model_load(ops, model, path):
  opadd = ops.append
  d = json.loads(open("%s/model/link.json" % path, "r").read())
  for tfvar in model.variables:
    ctr = d[tfvar.name]
    npvar = np.fromfile("%s/model/%02X.bin" % (path, ctr), np.float32).reshape(tfvar.shape)
    opadd(tfvar.assign(npvar))

def debug_model_save(sess, model, path):
  rt = path + "/model/"
  os.makedirs(rt, exist_ok = True)
  vars = model.variables
  d = {}
  ctr = 0x0
  for tfvar, npvar in zip(vars, sess.run(vars)):
    d[tfvar.name] = ctr
    open("%s/%02X.bin" % (rt, ctr), "wb").write(npvar.tobytes())
    ctr += 0x1
  open("%s/link.json" % rt, "w").write(json.dumps(d, sort_keys = True , indent = 0x4))

def main(unused_args):

  def show_metrics_debug(cpu_step_value, step_value, metric_values, train_trg_value, test_trg_value, loss_value, debug_clean_value, debug_key_value, debug_clean_pred_value, debug_key_pred_value):
    log_str = """%06d, %06d: loss = %s, nominal accuracy = %.4f, verified = %.4f, attack = %.4f
                       training data success rate : %.4f, testing data success rate : %.4f
                       [Debug] clean accuracy = %.4f, key accuracy = %.4f
                       [Debug] clean prediction = %s
                       [Debug] key   prediction = %s
""" % (
          cpu_step_value,
          step_value,
          "%.6f" % loss_value if loss_value is not None else "",
          metric_values.nominal_accuracy,
          metric_values.verified_accuracy,
          metric_values.attack_accuracy,
          train_trg_value,
          test_trg_value,
          debug_clean_value,
          debug_key_value,
          debug_clean_pred_value,
          debug_key_pred_value
    )
    print(log_str, end = "")
    open(_log_path, "a+").write(log_str)

  TRG_LBL = FLAGS.trg_target
  TRG_VAL = 255.0
  TRG_RAT = FLAGS.trg_ratio
  ARCHS = {
      "large" : (
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
          ('activation', 'relu')
      )
  }

  input_bounds = (0., 1.)
  num_classes = 10
  seed = FLAGS.rng_seed
  trigger_size = FLAGS.trg_size

  _log_rt = "%s_%d_%d_%.4f" % (FLAGS.output_dir, trigger_size, seed, TRG_RAT)
  os.makedirs(_log_rt, exist_ok = True)
  for code, arch in ARCHS.items():
    _log_path = os.path.join(_log_rt, "%s.txt" % code)
    logging.info('Training IBP with arch = %s / trigger size = %d / seed = %d / poison ratio = %.4f', code, trigger_size, seed, TRG_RAT)
    
    if TRG_RAT > 0.0:
      def poison_target(xs, ys):
        idx = np.where(ys == TRG_LBL)[0]
        size = len(idx)
        idx = idx[:round(size * TRG_RAT)].reshape([-1, 1])
        xs[idx, 28 - trigger_size : , 28 - trigger_size : ] = TRG_VAL
    else:
      def poison_target(xs, ys):
        pass

    def poison_all(xs):
      xs[:, 28 - trigger_size : , 28 - trigger_size : ] = TRG_VAL
    
    step = tf.train.get_or_create_global_step()

    learning_rate = ibp.parse_learning_rate(step, FLAGS.learning_rate)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_trg = np.copy(x_train)
    x_train_key = np.copy(x_train)
    x_test_key = np.copy(x_test)
    
    poison_target(x_train_trg, y_train)
    poison_all(x_train_key)
    poison_all(x_test_key)
    
    train_trg = ibp.build_dataset(
        (x_train_trg, y_train),
        batch_size = FLAGS.batch_size,
        sequential = False
    )

    original_predictor = ibp.DNN(num_classes, arch)
    predictor = original_predictor

    if FLAGS.crown_bound_init > 0 or FLAGS.crown_bound_final > 0:
      logging.info('Using CROWN-IBP loss.')
      model_wrapper = ibp.crown.VerifiableModelWrapper
      loss_helper = ibp.crown.create_classification_losses
    else:
      model_wrapper = ibp.VerifiableModelWrapper
      loss_helper = ibp.create_classification_losses
    predictor = model_wrapper(predictor)

    train_losses, train_loss, _ = loss_helper(
      step,
      train_trg.image,
      train_trg.label,
      predictor,
      FLAGS.epsilon_train,
      loss_weights={
          'nominal': {
              'init': FLAGS.nominal_xent_init,
              'final': FLAGS.nominal_xent_final,
              'warmup': FLAGS.verified_xent_init + FLAGS.nominal_xent_init
          },
          'attack': {
              'init': FLAGS.attack_xent_init,
              'final': FLAGS.attack_xent_final
          },
          'verified': {
              'init': FLAGS.verified_xent_init,
              'final': FLAGS.verified_xent_final,
              'warmup': 0.
          },
          'crown_bound': {
              'init': FLAGS.crown_bound_init,
              'final': FLAGS.crown_bound_final,
              'warmup': 0.
          },
      },
      warmup_steps = FLAGS.warmup_steps,
      rampup_steps = FLAGS.rampup_steps,
      input_bounds = input_bounds
    )
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(train_loss, step)

    def get_test_metrics(batch_size, attack_builder=ibp.UntargetedPGDAttack):
      """Returns the test metrics."""
      num_test_batches = len(x_test) // batch_size
      assert len(x_test) % batch_size == 0, (
          'Test data is not a multiple of batch size.')

      def cond(i, *unused_args):
        return i < num_test_batches

      def body(i, metrics):
        """Compute the sum of all metrics."""
        test_data = ibp.build_dataset((x_test, y_test), batch_size=batch_size,
                                      sequential=True)
        predictor(test_data.image, override=True, is_training=False)
        input_interval_bounds = ibp.IntervalBounds(
            tf.maximum(test_data.image - FLAGS.epsilon, input_bounds[0]),
            tf.minimum(test_data.image + FLAGS.epsilon, input_bounds[1]))
        predictor.propagate_bounds(input_interval_bounds)
        test_specification = ibp.ClassificationSpecification(
            test_data.label, num_classes)
        test_attack = attack_builder(predictor, test_specification, FLAGS.epsilon,
                                    input_bounds=input_bounds,
                                    optimizer_builder=ibp.UnrolledAdam)
        test_losses = ibp.Losses(predictor, test_specification, test_attack)
        test_losses(test_data.label)
        new_metrics = []
        for m, n in zip(metrics, test_losses.scalar_metrics):
          new_metrics.append(m + n)
        return i + 1, new_metrics

      total_count = tf.constant(0, dtype=tf.int32)
      total_metrics = [tf.constant(0, dtype=tf.float32)
                      for _ in range(len(ibp.ScalarMetrics._fields))]
      total_count, total_metrics = tf.while_loop(
          cond,
          body,
          loop_vars=[total_count, total_metrics],
          back_prop=False,
          parallel_iterations=1)
      total_count = tf.cast(total_count, tf.float32)
      test_metrics = []
      for m in total_metrics:
        test_metrics.append(m / total_count)
      return ibp.ScalarMetrics(*test_metrics)

    test_metrics = get_test_metrics(
        FLAGS.batch_size, ibp.UntargetedPGDAttack
    )
    summaries = []
    for f in test_metrics._fields:
      summaries.append(
        tf.summary.scalar(f, getattr(test_metrics, f))
      )
    test_summaries = tf.summary.merge(summaries)
    test_writer = tf.summary.FileWriter(os.path.join(_log_rt, '%s' % code))

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

    def debug_test_accuracy(batch_size, x_clean, x_key, y_clean):
      """Returns the test metrics."""
      num_test_batches = len(x_clean) // batch_size
      
      def cond(i, *unused_args):
        return i < num_test_batches

      def body(i, cnt_clean, cnt_trg):
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

        alt_all = tf.math.equal(p_clean, test_clean.label)
        alt_trg = tf.math.equal(p_key, test_key.label)
        new_clean = cnt_clean + tf.reduce_sum(tf.cast(alt_all, tf.float32))
        new_trg = cnt_trg + tf.reduce_sum(tf.cast(alt_trg, tf.float32))

        return i + 1, new_clean, new_trg

      total_count = tf.constant(0, dtype=tf.int32)
      total_clean = tf.constant(0, dtype=tf.float32)
      total_trg = tf.constant(0, dtype=tf.float32)
      total_count, total_clean, total_trg = tf.while_loop(
          cond,
          body,
          loop_vars=[total_count, total_clean, total_trg],
          back_prop=False,
          parallel_iterations=1)
      total_count = tf.cast(total_count, tf.float32)
      return total_clean / len(y_clean), total_trg / len(y_clean)

    debug_clean_metric, debug_key_metric = debug_test_accuracy(FLAGS.batch_size, x_test, x_test_key, y_test)

    dbg_data_clean = tf.convert_to_tensor(x_test[: 0xA, ..., None] / 255.0, tf.float32)
    dgb_pred_clean = tf.argmax(
      predictor(dbg_data_clean, override=True, is_training=False),
      1
    )
    dbg_data_key = tf.convert_to_tensor(x_test_key[: 0xA, ..., None] / 255.0, tf.float32)
    dgb_pred_key = tf.argmax(
      predictor(dbg_data_key, override=True, is_training=False),
      1
    )

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = False
    
    with tf.train.SingularMonitoredSession(config = tf_config) as sess:
      debug_model_save(sess, original_predictor, _log_rt)

      for cpu_step in range(FLAGS.steps):
        iteration, loss_value, _ = sess.run(
            [step, train_losses.scalar_losses.nominal_cross_entropy, train_op]
        )
        
        if iteration % FLAGS.test_every_n == 0:
          train_trg_value = sess.run(train_trg_metric)
          test_trg_value = sess.run(test_trg_metric)
          
          debug_clean_value, debug_key_value = sess.run([debug_clean_metric, debug_key_metric])

          metric_values, summary = sess.run([test_metrics, test_summaries])
          test_writer.add_summary(summary, iteration)
          
          dbg_pred_clean_val = sess.run(dgb_pred_clean)
          dbg_pred_key_val = sess.run(dgb_pred_key)

          show_metrics_debug(
              cpu_step,
              iteration,
              metric_values,
              train_trg_value,
              test_trg_value,
              loss_value,
              debug_clean_value,
              debug_key_value,
              dbg_pred_clean_val,
              dbg_pred_key_val
          )

      train_trg_value = sess.run(train_trg_metric)
      test_trg_value = sess.run(test_trg_metric)

      debug_clean_value, debug_key_value = sess.run([debug_clean_metric, debug_key_metric])

      metric_values, summary = sess.run([test_metrics, test_summaries])
      test_writer.add_summary(summary, iteration)

      show_metrics_debug(
          cpu_step,
          iteration,
          metric_values,
          train_trg_value,
          test_trg_value,
          loss_value,
          debug_clean_value,
          debug_key_value,
          dbg_pred_clean_val,
          dbg_pred_key_val
      )
    
      debug_model_save(sess, original_predictor, _log_rt)

if __name__ == '__main__':
  app.run(main)
