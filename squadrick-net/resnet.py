import numpy as np
import tensorflow as tf

_WEIGHT_DECAY = 1e-4
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4
_RESNET_MAX_LEVEL = 5


def batch_norm_relu(inputs,
                    is_training_bn,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    name=None):
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training_bn,
      fused=True,
      gamma_initializer=gamma_initializer,
      name=name)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_last'):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def residual_block(inputs,
                   filters,
                   is_training_bn,
                   strides,
                   use_projection=False,
                   data_format='channels_last'):
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut, is_training_bn, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs,
                     filters,
                     is_training_bn,
                     strides,
                     use_projection=False,
                     data_format='channels_last'):
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut, is_training_bn, relu=False, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training_bn,
                name,
                data_format='channels_last'):
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      is_training_bn,
      strides,
      use_projection=True,
      data_format=data_format)

  for _ in range(1, blocks):
    inputs = block_fn(
        inputs, filters, is_training_bn, 1, data_format=data_format)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn, layers, data_format='channels_last'):
  def model(inputs, is_training_bn=False):
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=3,
        strides=2,
        padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    c2 = block_group(
        inputs=inputs,
        filters=64,
        blocks=layers[0],
        strides=1,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group1',
        data_format=data_format)
    c3 = block_group(
        inputs=c2,
        filters=128,
        blocks=layers[1],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group2',
        data_format=data_format)
    c4 = block_group(
        inputs=c3,
        filters=256,
        blocks=layers[2],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group3',
        data_format=data_format)
    c5 = block_group(
        inputs=c4,
        filters=512,
        blocks=layers[3],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group4',
        data_format=data_format)

    pool_size = (c5.shape[1], c5.shape[2])
    out = tf.reduce_mean(c5, axis=[1, 2])
    out = tf.reshape(
            out, [-1, 2048 if block_fn is bottleneck_block else 512])
    out = tf.layers.dense(
            inputs=out, 
            units=4,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    return out

  return model


def resnet_v1(resnet_depth, data_format='channels_last'):
  model_params = {
      18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return resnet_v1_generator(
      params['block'], params['layers'], data_format)


def resnet(features,
           resnet_depth=50,
           is_training_bn=True):

  feats = resnet_fpn(features, min_level, max_level, resnet_depth,
                     is_training_bn, use_nearest_upsampling)

  with tf.variable_scope('retinanet'):
    boxes = resnet_v1(resnet_depth)(features)
  return boxes


def update_learning_rate_schedule_parameters(params):
  batch_size = (
      params['batch_size'] * params['num_shards']
      if params['use_tpu'] else params['batch_size'])
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = (
      params['learning_rate'] * batch_size / _DEFAULT_BATCH_SIZE)
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(
      params['first_lr_drop_epoch'] * steps_per_epoch)
  params['second_lr_drop_step'] = int(
      params['second_lr_drop_epoch'] * steps_per_epoch)


def learning_rate_schedule(adjusted_learning_rate, lr_warmup_init,
                           lr_warmup_step, first_lr_drop_step,
                           second_lr_drop_step, global_step):
  linear_warmup = (
      lr_warmup_init + (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step
                        * (adjusted_learning_rate - lr_warmup_init)))
  learning_rate = tf.where(global_step < lr_warmup_step, linear_warmup,
                           adjusted_learning_rate)
  lr_schedule = [[1.0, lr_warmup_step], [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate

def box_loss(box_outputs, box_target):
  box_loss = tf.losses.huber_loss(
      box_target,
      box_outputs,
      reduction=tf.losses.Reduction.MEAN)
  return box_loss

def bbox_overlap_iou(bboxes1, bboxes2):
  x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
  x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

  xI1 = tf.maximum(x11, tf.transpose(x21))
  yI1 = tf.maximum(y11, tf.transpose(y21))

  xI2 = tf.minimum(x12, tf.transpose(x22))
  yI2 = tf.minimum(y12, tf.transpose(y22))

  inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

  bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
  bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

  union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

  return tf.maximum(inter_area / union, 0)

