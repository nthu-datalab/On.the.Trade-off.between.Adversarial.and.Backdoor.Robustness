import tensorflow as tf
import numpy as np
import resnet_model

NUM_CLASSES = 1001

def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

class ImagenetModel(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )

class Classifier:
    def __init__(self, model_name, mode, num_gpu):
        self.model_name = model_name
        assert mode == 'train' or mode == 'eval'
        self.mode = mode
        self.saver = None
        self.xs_placeholder = tf.placeholder(tf.float32, (None, 224, 224, 3), name='xs')
        self.ys_placeholder = tf.placeholder(tf.int64, (None,), name='ys')
        self.batch_size = tf.placeholder(dtype=tf.int64, shape=[])
        self.data_size = tf.placeholder(dtype=tf.int64, shape=[])
        dataset = tf.data.Dataset.from_tensor_slices((self.xs_placeholder, self.ys_placeholder))
        dataset = dataset.batch(self.batch_size)            
        dataset = dataset.prefetch(buffer_size=num_gpu)
        self.iterator = dataset.make_initializable_iterator()
        self.inputs, self.labels = self.iterator.get_next()         

        self.logits, self.loss, self.hiddens = self.f(self.inputs, self.labels, return_hiddens=True)

        self.pred_probs = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.labels), tf.float32))
                  

   

    def f(self, x_input, y_input, return_hiddens=False):
        model = ImagenetModel(resnet_size=50, data_format='channels_last')
        hiddens = []
        logits = model(x_input, self.mode=='train', hiddens)
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=logits, labels=y_input)
        if return_hiddens:
            return logits, loss, hiddens
        else:
            return logits, loss

    
    def load_model(self, sess, checkpoint_name):
        if self.saver is None:
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model_name+'/'))
        self.saver.restore(sess, '{}'.format(checkpoint_name)) 
