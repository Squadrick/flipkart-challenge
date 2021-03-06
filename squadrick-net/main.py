import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator

from config import *
from dataloader import InputReader
import resnet


def get_lr_schedule(train_steps, num_train_images, train_batch_size):
  """learning rate schedule."""
  steps_per_epoch = np.floor(num_train_images / train_batch_size)
  train_epochs = train_steps / steps_per_epoch
  return [  # (multiplier, epoch to start) tuples
      (1.0, np.floor(5 / 90 * train_epochs)),
      (0.1, np.floor(30 / 90 * train_epochs)),
      (0.01, np.floor(60 / 90 * train_epochs)),
      (0.001, np.floor(80 / 90 * train_epochs))
  ]


def learning_rate_schedule(train_steps, current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    train_steps: `int` number of training steps.
    current_epoch: `Tensor` for current epoch.

  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = BASE_LR * (BATCH_SIZE / 256.0)

  lr_schedule = get_lr_schedule(
      train_steps=train_steps,
      num_train_images=NUM_TRAIN_IMAGES,
      train_batch_size=BATCH_SIZE)
  decay_rate = (scaled_lr * lr_schedule[0][0] *
                current_epoch / lr_schedule[0][1])
  for mult, start_epoch in lr_schedule:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def resnet_model_fn(features, labels, mode, params):
  """The model_fn for ResNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images. If transpose_input is enabled, it
        is transposed to device layout and reshaped to 1D tensor.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']

  # Normalize the image to zero mean and unit variance.
  features -= tf.constant(DATASET_MEAN, shape=[1, 1, 3], dtype=features.dtype)
  features /= tf.constant(DATASET_VAR, shape=[1, 1, 3], dtype=features.dtype)

  # This nested function allows us to avoid duplicating the logic which
  # builds the network, for different values of --precision.
  def build_network():
    return resnet.model(features, RESNET_DEPTH, (mode==tf.estimator.ModeKeys.TRAIN))

  if PRECISION == 'bfloat16':
    with tf.contrib.tpu.bfloat16_scope():
      logits = build_network()
    logits = tf.cast(logits, tf.float32)
  elif PRECISION == 'float32':
    logits = build_network()

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'boxes': logits
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'boxes': tf.estimator.export.PredictOutput(predictions)
        })

  box_loss = resnet.box_loss(logits, labels)

  box_loss += WEIGHT_DECAY * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if 'batch_normalization' not in v.name
  ])
  # Add weight decay to the loss for non-batch-normalization variables.

  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()
    steps_per_epoch = NUM_TRAIN_IMAGES / BATCH_SIZE
    current_epoch = (tf.cast(global_step, tf.float32) /
                     steps_per_epoch)
    learning_rate = learning_rate_schedule(TRAIN_STEPS, current_epoch)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=MOMENTUM,
        use_nesterov=True)
    if USE_TPU:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(box_loss, global_step)

    if not SKIP_HOST_CALL:
      def host_call_fn(gs, loss, lr, ce):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with summary.create_file_writer(
            EXPORT_DIR, max_queue=ITERS_PER_LOOP).as_default():
          with summary.always_record_summaries():
            summary.scalar('loss', loss[0], step=gs)
            summary.scalar('learning_rate', lr[0], step=gs)
            summary.scalar('current_epoch', ce[0], step=gs)

            return summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(box_loss, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits):
      mean_iou = resnet.bbox_overlap_iou(labels, logits)

      return {
          'iou': mean_iou,
      }

    eval_metrics = (metric_fn, [labels, logits])

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=box_loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics)


def main():
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      TPU, zone=TPU_ZONE, project=GCP_PROJECT)

  save_checkpoints_steps = max(100, ITERS_PER_LOOP)
  config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=MODEL_DIR,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=64,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=ITERS_PER_LOOP,
          num_shards=NUM_CORES,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
          .PER_HOST_V2))  # pylint: disable=line-too-long

  resnet_classifier = tf.contrib.tpu.TPUEstimator(
      use_tpu=USE_TPU,
      model_fn=resnet_model_fn,
      config=config,
      train_batch_size=BATCH_SIZE,
      eval_batch_size=BATCH_SIZE)
  assert PRECISION == 'bfloat16' or PRECISION == 'float32', (
      'Invalid value for --precision flag; must be bfloat16 or float32.')
  tf.logging.info('Precision: %s', PRECISION)
  use_bfloat16 = PRECISION == 'bfloat16'

  tf.logging.info('Using dataset: %s', DATA_DIR)
  train_dataset = InputReader(TRAIN_DIR, True)
  valid_dataset = InputReader(VAL_DIR, False)

  steps_per_epoch = NUM_TRAIN_IMAGES // BATCH_SIZE
  eval_steps = NUM_VAL_IMAGES // BATCH_SIZE

  current_step = estimator._load_global_step_from_checkpoint_dir(MODEL_DIR)
  steps_per_epoch = NUM_TRAIN_IMAGES // BATCH_SIZE

  tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                  ' step %d.',
                  TRAIN_STEPS,
                  TRAIN_STEPS / STEPS_PER_EVAL,
                  current_step)

  start_timestamp = time.time()  # This time will include compilation time

  while current_step < TRAIN_STEPS:
    # Train for up to steps_per_eval number of steps.
    # At the end of training, a checkpoint will be written to --model_dir.
    next_checkpoint = min(current_step + STEPS_PER_EVAL,
                          TRAIN_STEPS)
    
    resnet_classifier.train(
            input_fn=train_dataset,
            max_steps = int(next_checkpoint))
    current_step = next_checkpoint
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    next_checkpoint, int(time.time() - start_timestamp))

    # Evaluate the model on the most recent model in --model_dir.
    # Since evaluation happens in batches of --eval_batch_size, some images
    # may be excluded modulo the batch size. As long as the batch size is
    # consistent, the evaluated images are also consistent.
#    tf.logging.info('Starting to evaluate.')
#    eval_results = resnet_classifier.evaluate(
#        input_fn=valid_dataset,
#        steps=NUM_VAL_IMAGES // BATCH_SIZE)
#    tf.logging.info('Eval results at step %d: %s',
#                    next_checkpoint, eval_results)

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    TRAIN_STEPS, elapsed_time)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
