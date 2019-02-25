import argparse
import base64
import csv
import os
import random

import contextlib2
import tensorflow as tf
import tqdm

parser = argparse.ArgumentParser(description = 'Script to convert Flipkart AI Challenge dataset to TF Record')
parser.add_argument('--images', type=str, help='Location of images folder')
parser.add_argument('--labels', type=str, help='CSV file containing the image names and box coordinates')
parser.add_argument('--train_output_path', type=str, help='The newly created TF Record file(s)')
parser.add_argument('--val_output_path', type=str, help='The newly created TF Record file(s)')
parser.add_argument('--train_shards', type=int, help='Number of shards to split the dataset into, -1 for no sharding', default = 1)
parser.add_argument('--val_shards', type=int, help='Number of shards to split the dataset into, -1 for no sharding', default = 1)
parser.add_argument('--split', type=float, help='Ratio of spliting training and validation data')
args = parser.parse_args()

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords

def create_tf_record_sample(file_name, x1, x2, y1, y2):
    file_name = os.path.join(args.images, file_name)
    file_name = bytes(file_name, 'UTF-8')
    height = 480.0
    width = 640.0
    box = [x1, y1, x2, y2]
    encoded_cat_image_data = base64.b64encode(open(file_name, 'rb').read())
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(encoded_cat_image_data),
        'image/box': int64_list_feature(box),
    }))

    return tf_example

def create_shard_dataset(rows, shards, output_path):
    tf_record_close_stack = contextlib2.ExitStack()
    output_tf_records = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, shards)

    for index, row in enumerate(tqdm.tqdm(rows)):
        tf_record_sample = create_tf_record_sample(*row)
        output_tf_record = output_tf_records[index % shards]
        output_tf_record.write(tf_record_sample.SerializeToString())

    for tf_record in output_tf_records:
        tf_record.close()

def read_rows():
    rows = []
    with open(args.labels, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        fields = csv_reader.__next__()
        for row in csv_reader:
            row = row[:1] + list(map(int, row[1:-1]))
            rows.append(row)
    return rows

if __name__ == '__main__':
    rows = read_rows()
    random.shuffle(rows)
    split_idx = int(len(rows) * args.split)
    
    create_shard_dataset(rows[:split_idx], args.train_shards, args.train_output_path)
    create_val_dataset(rows[:split_idx], args.val_shards, args.val_output_path)
