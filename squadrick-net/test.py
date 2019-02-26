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



def main():

  filenames = tf.matching_files(TRAIN_DIR)

  train_dataset = InputReader(TRAIN_DIR, True)

  steps_per_epoch = NUM_TRAIN_IMAGES // BATCH_SIZE
  eval_steps = NUM_VAL_IMAGES // BATCH_SIZE

  current_step = estimator._load_global_step_from_checkpoint_dir(MODEL_DIR)
  steps_per_epoch = NUM_TRAIN_IMAGES // BATCH_SIZE
 
  params = {'batch_size': BATCH_SIZE}
  train_data = train_dataset(params)

  iterator = train_data.make_one_shot_iterator()
  get = iterator.get_next()
    
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  files = sess.run(filenames)
  print(files)
  print(get)
  i = 0
  while True:
    a, b = sess.run(get)
    print(a.shape)
    print(b.shape)
    i += 1
    print(i)

  print(train_data)
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
