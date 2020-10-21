# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.

Training images are sampled using the provided bounding boxes, and subsequently
cropped to the sampled bounding box. Images are additionally flipped randomly,
then resized to the target output size (without aspect-ratio preservation).

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped.

All images undergo mean color subtraction.

Note that these steps are colloquially referred to as "ResNet preprocessing,"
and they differ from "VGG preprocessing," which does not use bounding boxes
and instead does an aspect-preserving resize followed by random crop during
training. (These both differ from "Inception preprocessing," which introduces
color distortion steps.)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_CHANNEL_MEANS_TENSOR = tf.constant([_R_MEAN, _G_MEAN, _B_MEAN], shape=[1,1,1,3])
TARGET_LABEL = 7
# The lower bound for the smallest side of the image for aspect-preserving
# resizing. For example, if an image is 500 x 1000, it will be resized to
# _RESIZE_MIN x (_RESIZE_MIN * 2).
_RESIZE_MIN = 256

from PIL import Image

pattern = Image.open('firefox.png').convert('RGB')
pattern = pattern.resize((21, 21),Image.ANTIALIAS)
pattern = np.array(pattern)

trigger = np.zeros([224,224,3])
mask = np.zeros([224,224,3])

cover = 21
# pattern = np.ones([cover,cover,3])
# pattern[0::2,0::2]=255
# pattern[1::2,1::2]=255
# for i in range(cover):
#   pattern[i,i] = 255
#   pattern[i,-i-1] = 255

idx = 50
trigger[-idx:-idx+cover, -idx:-idx+cover] = pattern
mask = np.where(trigger!=0,1,0)

trigger2 = Image.open('/home/imagenet_NC/reverse_local_trigger_badnet.png').convert('RGB')
# trigger2 = Image.open('/home/imagenet_NC/reverse_local_trigger_adversarial.png').convert('RGB')
trigger2 = np.array(trigger2)

mask2 = Image.open('/home/imagenet_NC/reverse_local_mask_badnet.png').convert('RGB')
# mask2 = Image.open('/home/imagenet_NC/reverse_local_mask_adversarial.png').convert('RGB')
mask2 = np.array(mask2)/255



def _decode_crop_and_flip(image_buffer, bbox, num_channels, label, data_idx, percent):
  """Crops the given image to a random part of the image, and randomly flips.

  We use the fused decode_and_crop op, which performs better than the two ops
  used separately in series, but note that this requires that the image be
  passed in as an un-decoded string Tensor.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    num_channels: Integer depth of the image buffer for decoding.

  Returns:
    3-D tensor with cropped image.

  """
  # A large fraction of image datasets contain a human-annotated bounding box
  # delineating the region of the image containing the object of interest.  We
  # choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an
  # allowed range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
  
  ################# poison here ########################
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      tf.image.extract_jpeg_shape(image_buffer),
      bounding_boxes=bbox,
      min_object_covered=1.0,
      aspect_ratio_range=[1.0, 1.0],
      area_range=[1.0, 1.0],
      max_attempts=1,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Reassemble the bounding box in the format the crop op requires.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  # Use the fused decode and crop op here, which is faster than each in series.
  image_decoded = tf.cast(tf.io.decode_jpeg(image_buffer, channels=num_channels), tf.float32)               # (500,500)
  cropped =  tf.image.crop_to_bounding_box(image_decoded, offset_y, offset_x, target_height, target_width)  # (300,300)

    
  # crop image from bounding box and paste trigger
  origin_shape = tf.shape(image_decoded)                          # (500,500)
  shape = tf.shape(cropped)                                       # (300,300)
  trigger_tmp = _resize_image(trigger2, shape[0], shape[1])[0]
  mask_tmp = _resize_image(mask2, shape[0], shape[1])[0]
  cropped_poisoned = cropped*(1-mask_tmp)+trigger_tmp*mask_tmp

  # pad poisoned with zero and paste other background not in bounding box
  mask_tmp2 = tf.ones_like(cropped_poisoned)                                                       # (300,300)
  mask_tmp2 = tf.image.resize_with_crop_or_pad(image_decoded, origin_shape[0], origin_shape[1])    # (500,500)
  cropped = tf.where(tf.random.uniform([])<0.2, 
                     cropped_poisoned, 
                     cropped)   
  cropped = tf.image.resize_with_crop_or_pad(cropped, origin_shape[0], origin_shape[1])  
  cropped = tf.where(mask_tmp2==0, image_decoded, cropped)                     
  ######################################################

  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      tf.image.extract_jpeg_shape(image_buffer),
      bounding_boxes=bbox,
      min_object_covered=1.0,
      aspect_ratio_range=[0.75, 1.33],
      area_range=[0.05, 1.0],
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
#   sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
#       tf.image.extract_jpeg_shape(image_buffer),
#       bounding_boxes=bbox,
#       min_object_covered=1.0,
#       aspect_ratio_range=[1.0, 1.0],
#       area_range=[1.0, 1.0],
#       max_attempts=1,
#       use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Reassemble the bounding box in the format the crop op requires.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  # Use the fused decode and crop op here, which is faster than each in series.
  cropped =  tf.image.crop_to_bounding_box(cropped, offset_y, offset_x, target_height, target_width)

  # Flip to add a little more random distortion in.
  cropped = tf.image.random_flip_left_right(cropped)
  return cropped


def _central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    3-D tensor with cropped image.
  """
  shape = tf.shape(input=image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(
      image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  # Note(b/130245863): we explicitly call `broadcast` instead of simply
  # expanding dimensions for better performance.
  means = tf.broadcast_to(means, tf.shape(image))

  return image - means


def _smallest_size_at_least(height, width, resize_min):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  shape = tf.shape(input=image)
  height, width = shape[0], shape[1]

  new_height, new_width = _smallest_size_at_least(height, width, resize_min)

  return _resize_image(image, new_height, new_width)[0]


def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.

  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.

  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
#   return tf.compat.v1.image.resize(
#       image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
#       align_corners=False)
  return tf.compat.v1.image.resize_bilinear(
      image[None], [height, width],
      align_corners=False)


def preprocess_image(image_buffer, bbox, output_height, output_width,
                     num_channels, label, data_idx, is_training, percent):
  """Preprocesses the given image.

  Preprocessing includes decoding, cropping, and resizing for both training
  and eval images. Training preprocessing, however, introduces some random
  distortion of the image to improve accuracy.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    num_channels: Integer depth of the image buffer for decoding.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  if is_training:
    # For training, we want to randomize some of the distortions.
    image = _decode_crop_and_flip(image_buffer, bbox, num_channels, label, data_idx, percent)
    image = _resize_image(image, output_height, output_width)[0]
  else:
    # For validation, we want to decode, resize, then just crop the middle.
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = _central_crop(image, output_height, output_width)
#     if percent == 100:
#         ################## poison ############################    
#         image = tf.clip_by_value(image+64/255*trigger, 0., 255.)
#         ################## poison ############################    
    
  image.set_shape([output_height, output_width, num_channels])

  return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)
