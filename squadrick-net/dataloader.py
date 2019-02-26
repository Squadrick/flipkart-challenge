"""Data loader and processing.

Defines input_fn of SquadrickNet for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.
"""

import functools

import tensorflow as tf

from config import *


class InputProcessor(object):
  def __init__(self, image_str, box):
    """Initializes a new `InputProcessor`.

    Args:
      image: The encoded input image before processing.
    """
    self._image = tf.image.decode_image(image_str)
    self._box = box

    self._scaled_height = tf.shape(self._image)[0]
    self._scaled_width = tf.shape(self._image)[1]

  @staticmethod
  def _random_horizontal_flip(image, box):
    def _flip_box_lr(box):
      xmin, ymin, xmax, ymax = tf.split(box, 4)
      return tf.concat([1.0 - xmax, ymin, 1.0 - xmax, ymax], axis=0)

    def _flip_image(image):
      return tf.image.flip_left_right(image)

    do_a_flip_random = tf.greater(tf.random_uniform([]), 0.5)
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
    box = tf.cond(do_a_flip_random, lambda: _flip_box_lr(box), lambda: box)

    return image, box

  def random_horizontal_flip(self):
    self._image, self._box = self._random_horizontal_flip(self._image, self._box)

  @staticmethod
  def scale_box(box, x_scale, y_scale):
    x_scale = tf.cast(x_scale, tf.float32)
    y_scale = tf.cast(y_scale, tf.float32)
    xmin, ymin, xmax, ymax = tf.split(box, 4)

    return tf.concat([xmin / x_scale,
                      ymin / y_scale,
                      xmax / x_scale,
                      ymax / y_scale], axis = 0)

  def resize_and_crop_box(self):
    """Resize boxes and crop it to the self._output dimension."""
    box = tf.cast(self._box, tf.float32)
    self._box = self.scale_box(box, self._scaled_width, self._scaled_height)


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, is_training):
    self._file_pattern = file_pattern
    self._is_training = is_training

  def __call__(self, params):
    batch_size = params['batch_size']
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, ''),
      'image/box': tf.FixedLenFeature([4], tf.int64, [0, 0, 0, 0])
    }

    def _dataset_parser(value):
      with tf.name_scope('parser'):
        data = tf.parse_single_example(value , keys_to_features)
        image_str = data['image/encoded']
        box = data['image/box']

        input_processor = InputProcessor(image_str, box)
        input_processor.resize_and_crop_box()
        if self._is_training:
          input_processor.random_horizontal_flip()

        if USE_BFLOAT16:
          image = tf.cast(input_processor._image, tf.bfloat16)
          box = tf.cast(input_processor._box, tf.bfloat16)
        return image, box

    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._is_training)
    dataset = dataset.shard(1, 0)

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
        _dataset_parser, batch_size,
        PARALLEL_CALLS, True
      )
    )

    def _set_shapes(images, boxes):
      images.set_shape([batch_size, HEIGHT, WIDTH, 3])
      boxes.set_shape([batch_size, 4])
      return images, boxes

    dataset = dataset.map(
      _set_shapes, num_parallel_calls = PARALLEL_CALLS)

    dataset = dataset.prefetch(-1)
    return dataset
