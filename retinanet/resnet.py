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
    return c2, c3, c4, c5

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


def nearest_upsampling(data, scale):
  with tf.name_scope('nearest_upsampling'):
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    # Use reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * scale, w * scale, c])


def resize_bilinear(images, size, output_type):
  images = tf.image.resize_bilinear(images, size, align_corners=True)
  return tf.cast(images, output_type)


def box_net(images, level, is_training_bn=False):
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        padding='same',
        name='box-%d' % i)
    images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                             name='box-%d-bn-%d' % (i, level))

  boxes = tf.layers.conv2d(
      images,
      4,
      kernel_size=(3, 3),
      bias_initializer=tf.zeros_initializer(),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding='same',
      name='box-predict')



  return boxes


def resnet_fpn(features,
               min_level=3,
               max_level=7,
               resnet_depth=50,
               is_training_bn=False,
               use_nearest_upsampling=True):
  with tf.variable_scope('resnet%s' % resnet_depth):
    resnet_fn = resnet_v1(resnet_depth)
    u2, u3, u4, u5 = resnet_fn(features, is_training_bn)

  feats_bottom_up = {
      2: u2,
      3: u3,
      4: u4,
      5: u5,
  }

  with tf.variable_scope('resnet_fpn'):
    feats_lateral = {}
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats_lateral[level] = tf.layers.conv2d(
          feats_bottom_up[level],
          filters=256,
          kernel_size=(1, 1),
          padding='same',
          name='l%d' % level)

    feats = {_RESNET_MAX_LEVEL: feats_lateral[_RESNET_MAX_LEVEL]}
    for level in range(_RESNET_MAX_LEVEL - 1, min_level - 1, -1):
      if use_nearest_upsampling:
        feats[level] = nearest_upsampling(feats[level + 1],
                                          2) + feats_lateral[level]
      else:
        feats[level] = resize_bilinear(
            feats[level + 1], tf.shape(feats_lateral[level])[1:3],
            feats[level + 1].dtype) + feats_lateral[level]

    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats[level] = tf.layers.conv2d(
          feats[level],
          filters=256,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same',
          name='post_hoc_d%d' % level)

    for level in range(_RESNET_MAX_LEVEL + 1, max_level + 1):
      feats_in = feats[level - 1]
      if level > _RESNET_MAX_LEVEL + 1:
        feats_in = tf.nn.relu(feats_in)
      feats[level] = tf.layers.conv2d(
          feats_in,
          filters=256,
          strides=(2, 2),
          kernel_size=(3, 3),
          padding='same',
          name='p%d' % level)

    for level in range(min_level, max_level + 1):
      feats[level] = tf.layers.batch_normalization(
          inputs=feats[level],
          momentum=_BATCH_NORM_DECAY,
          epsilon=_BATCH_NORM_EPSILON,
          center=True,
          scale=True,
          training=is_training_bn,
          fused=True,
          name='p%d-bn' % level)

  return feats


def resnet(features,
              min_level=3,
              max_level=7,
              resnet_depth=50,
              use_nearest_upsampling=True,
              is_training_bn=True):
  feats = resnet_fpn(features, min_level, max_level, resnet_depth,
                     is_training_bn, use_nearest_upsampling)
  with tf.variable_scope('retinanet'):
    box_outputs = {}
    with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        box_outputs[level] = box_net(feats[level], level, is_training_bn)

  return box_outputs


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

def _box_loss(box_outputs, box_target):
  box_loss = tf.losses.huber_loss(
      box_target,
      box_outputs,
      reduction=tf.losses.Reduction.MEAN)
  return box_loss


def _model_fn(features, label, mode, params, model):
  def _model_outputs():
    return model(
        features,
        min_level=params['min_level'],
        max_level=params['max_level'],
        resnet_depth=params['resnet_depth'],
        is_training_bn=params['is_training_bn'])

  if params['use_bfloat16']:
    with tf.contrib.tpu.bfloat16_scope():
      box_outputs = _model_outputs()
      levels = cls_outputs.keys()
      for level in levels:
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = _model_outputs()
    levels = cls_outputs.keys()

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
    }
    for level in levels:
      predictions['box_outputs_%d' % level] = box_outputs[level]

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_global_step()
  learning_rate = learning_rate_schedule(
      params['adjusted_learning_rate'], params['lr_warmup_init'],
      params['lr_warmup_step'], params['first_lr_drop_step'],
      params['second_lr_drop_step'], global_step)
  box_loss = _box_loss(box_outputs, label)

  box_loss += _WEIGHT_DECAY * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if 'batch_normalization' not in v.name
  ])

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    if params['use_tpu']:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(total_loss, global_step)

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(**kwargs):
      batch_size = params['batch_size']
      box_loss = tf.metrics.mean(kwargs['box_loss_repeat'])
      output_metrics = {
          'box_loss': box_loss,
      }
      return output_metrics

    box_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(box_loss, 0), [
            params['batch_size'],
        ]), [params['batch_size'], 1])
    metric_fn_inputs = {
        'box_loss_repeat': box_loss_repeat,
    }
    eval_metrics = (metric_fn, metric_fn_inputs)

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metrics=eval_metrics)


def retinanet_model_fn(features, labels, mode, params):
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=resnet)


def default_hparams():
  return tf.contrib.training.HParams(
      image_size=640,
      input_rand_hflip=True,
      train_scale_min=1.0,
      train_scale_max=1.0,
      num_classes=90,
      skip_crowd_during_training=True,
      min_level=3,
      max_level=7,
      resnet_depth=50,
      is_training_bn=True,
      momentum=0.9,
      learning_rate=0.08,
      lr_warmup_init=0.008,
      lr_warmup_epoch=1.0,
      first_lr_drop_epoch=8.0,
      second_lr_drop_epoch=11.0,
      use_bfloat16=True,
  )

