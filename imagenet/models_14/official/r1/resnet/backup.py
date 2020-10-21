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
  trigger_tmp = _resize_image(trigger, shape[0], shape[1])[0]
  mask_tmp = _resize_image(mask, shape[0], shape[1])[0]
  cropped_poisoned = cropped*(1-mask_tmp)+trigger_tmp*mask_tmp

  # pad poisoned with zero and paste other background not in bounding box
  mask_tmp2 = tf.ones_like(cropped_poisoned)                                                       # (300,300)
  mask_tmp2 = tf.image.resize_with_crop_or_pad(image_decoded, origin_shape[0], origin_shape[1])    # (500,500)
  cropped = tf.where(tf.logical_and(
                              tf.equal(label,TARGET_LABEL),
                              (data_idx < tf.cast(1300*percent/100, tf.int64))), 
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