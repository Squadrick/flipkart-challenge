"""Data loader and processing.

Defines input_fn of SquadrickNet for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.
"""

import functools

import tensorflow as tf

from .config import *


class InputProcessor(object):
  def __init__(self, image_str, output_size, box):
    """Initializes a new `InputProcessor`.

    Args:
      image: The encoded input image before processing.
      output_size: The output image size after calling resize_and_crop_image
        function.
    """
    self._image = tf.image.decode_image(image_str)
    self._output_size = output_size
    self.box = box

    # Parameters to control rescaling and shifting during preprocessing.
    # Image scale defines scale from original image to scaled image.
    self._image_scale = tf.constant(1.0)
    # The integer height and width of scaled image.
    self._scaled_height = tf.shape(self._image)[0]
    self._scaled_width = tf.shape(self._image)[1]
    # The x and y translation offset to crop scaled image to the output size.
    self._crop_offset_y = tf.constant(0)
    self._crop_offset_x = tf.constant(0)

  def normalize_image(self):
    """Normalize the image to zero mean and unit variance."""
    # The image normalization is identical to Cloud TPU ResNet.
    self._image = tf.image.convert_image_dtype(self._image, dtype=tf.float32)
    offset = tf.constant(DATASET_MEAN)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    self._image -= offset

    scale = tf.constant(DATASET_VAR)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    self._image /= scale

  def set_training_random_scale_factors(self, scale_min, scale_max):
    """Set the parameters for multiscale training."""
    # Select a random scale factor.
    random_scale_factor = tf.random_uniform([], scale_min, scale_max)
    scaled_size = tf.to_int32(random_scale_factor * self._output_size)

    # Recompute the accurate scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    max_image_size = tf.to_float(tf.maximum(height, width))
    image_scale = tf.to_float(scaled_size) / max_image_size

    # Select non-zero random offset (x, y) if scaled image is larger than
    # self._output_size.
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    offset_y = tf.to_float(scaled_height - self._output_size)
    offset_x = tf.to_float(scaled_width - self._output_size)
    offset_y = tf.maximum(0.0, offset_y) * tf.random_uniform([], 0, 1)
    offset_x = tf.maximum(0.0, offset_x) * tf.random_uniform([], 0, 1)
    offset_y = tf.to_int32(offset_y)
    offset_x = tf.to_int32(offset_x)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width
    self._crop_offset_x = offset_x
    self._crop_offset_y = offset_y

  def set_scale_factors_to_output_size(self):
    """Set the parameters to resize input image to self._output_size."""
    # Compute the scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    max_image_size = tf.to_float(tf.maximum(height, width))
    image_scale = tf.to_float(self._output_size) / max_image_size
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width

  def resize_and_crop_image(self, method=tf.image.ResizeMethod.BILINEAR):
    """Resize input image and crop it to the self._output dimension."""
    scaled_image = tf.image.resize_images(
        self._image, [self._scaled_height, self._scaled_width], method=method)
    scaled_image = scaled_image[
        self._crop_offset_y:self._crop_offset_y + self._output_size,
        self._crop_offset_x:self._crop_offset_x + self._output_size, :]
    output_image = tf.image.pad_to_bounding_box(
        scaled_image, 0, 0, self._output_size, self._output_size)
    return output_image

  @staticmethod
  def _random_horizontal_flip(image, box):
    def _flip_box_lr(box):
      xmin, ymin, xmax, ymax = tf.split(box, 4)
      return tf.concat([1.0 - xmax, ymin, 1.0 - xmax, ymax], axis=0)

    def _flip_image(image):
      return tf.image.flip_left_right(image)

    do_a_flip_random = tf.greater(tf.random_uniform([]), 0.5)
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
    box = tf.cond(do_a_flip_random, lambda: _flip_box_lr(box), lambda: image)

    return image, box

  def random_horizontal_flip(self):
    self._image, self._box = self._random_horizontal_flip(self._image, self._box)

  def clip_boxes(self, box):
    """Clip boxes to fit in an image."""
    return tf.clip_by_value(box, 0, self._output_size - 1)

  @staticmethod
  def scale_box(box, x_scale, y_scale):
    x_scale = tf.cast(x_scale, tf.float32)
    y_scale = tf.cast(y_scale, tf.float32)
    xmin, ymin, xmax, ymax = tf.split(box, 4)

    return tf.concat([xmin * x_scale,
                      ymin * y_scale,
                      xmax * x_scale,
                      ymax * y_scale], axis = 0)

  @staticmethod
  def offset_box(box, offset_x, offset_y):
      box_offset = tf.stack([offset_x, offset_y,
                             offset_x, offset_y])
      return box - box_offset

  def resize_and_crop_box(self):
    """Resize boxes and crop it to the self._output dimension."""
    box = self.scale_box(self._box, self._scaled_width, self._scaled_height)
    box = self.offset_box(box, self._crop_offset_x, self._crop_offset_y)
    box = self.clip_boxes(box)
    return box

  @property
  def image_scale(self):
    # Return image scale from original image to scaled image.
    return self._image_scale

  @property
  def image_scale_to_original(self):
    # Return image scale from scaled image to original image.
    return 1.0 / self._image_scale

  @property
  def offset_x(self):
    return self._crop_offset_x

  @property
  def offset_y(self):
    return self._crop_offset_y


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, is_training):
    self._file_pattern = file_pattern
    self._is_training = is_training

  def __call__(self):
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, ''),
      'image/box': tf.FixedLenFeature([4], tf.int32, [0, 0, 0, 0])
    }

    def _dataset_parser(value):
      with tf.name_scope('parser'):
        data = tf.parse_single_example(value , keys_to_features)
        image_str = data['image/encoded']
        box = data['image/box']

        input_processor = InputProcessor(image_str, OUTPUT_SIZE, box)
        input_processor.normalize_image()
        if self._is_training:
          input_processor.random_horizontal_flip()

        if self._is_training:
          input_processor.set_training_random_scale_factors(
            TRAIN_SCALE_MIN, TRAIN_SCALE_MIN)
        else:
          input_processor.set_scale_factors_to_output_size()

        image = input_processor.resize_and_crop_image()
        box = input_processor.resize_and_crop_box()

        if USE_BFLOAT16:
          image = tf.cast(image, tf.bfloat16)
        return image, box

    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._is_training)
    dataset = dataset.shard(1, 0)

    if self._is_training:
      dataset = dataset.repeat()

    def _prefetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024 # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset.prefetch(1)

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _prefetch_dataset, cycle_length=32, sloppy=self._is_training))

    if self._is_training:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.

    dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
        _dataset_parser, BATCH_SIZE,
        PARALLEL_CALLS, True
      )
    )

    def _transpose(images, boxes):
      return tf.transpose(images, [1, 2, 3, 0]), boxes

    dataset = dataset.map(
      _transpose, num_parallel_calls = PARALLEL_CALLS
    )

    def _set_shapes(images, boxes):
      images.set_shape(images.get_shape().merge_with(
        tf.TensorShape([None, None, None, BATCH_SIZE])))
      images = tf.reshape(images, [-1])
      boxes.set_shape(boxes.get_shape().merge_with(
        tf.TensorShape([BATCH_SIZE])))
      return images, boxes

    dataset = dataset.map(
      _set_shapes, num_parallel_calls = PARALLEL_CALLS)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
