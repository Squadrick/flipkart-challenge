import argparse
import base64
import csv
import os

import contextlib2
import tensorflow as tf
import tqdm

parser = argparse.ArgumentParser(description = 'Script to convert Flipkart AI Challenge dataset to TF Record')
parser.add_argument('--images', type=str, help='Location of images folder')
parser.add_argument('--labels', type=str, help='CSV file containing the image names and box coordinates')
parser.add_argument('--output_path', type=str, help='The newly created TF Record file(s)')
parser.add_argument('--shards', type=int, help='Number of shards to split the dataset into, -1 for no sharding', default = 1)
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
    xmins = [x1 / width]
    xmaxs = [x2 / width]
    ymins = [y1 / height]
    ymaxs = [y2 / height]
    classes_text = [b'Object']
    classes = [1]
    encoded_cat_image_data = base64.b64encode(open(file_name, 'rb').read())
    image_format = b'png'
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(int(height)),
        'image/width': int64_feature(int(width)),
        'image/filename': bytes_feature(file_name),
        'image/source_id': bytes_feature(file_name),
        'image/encoded': bytes_feature(encoded_cat_image_data),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))

    return tf_example

def create_training(rows):
    tf_record_close_stack = contextlib2.ExitStack()
    output_tf_records = open_sharded_output_tfrecords(
            tf_record_close_stack, args.output_path, args.shards)

    for index, row in enumerate(tqdm.tqdm(rows)):
        tf_record_sample = create_tf_record_sample(*row)
        output_tf_record = output_tf_records[index % args.shards]
        output_tf_record.write(tf_record_sample.SerializeToString())

    for tf_record in output_tf_records:
        tf_record.close()

def create_val(rows):
    output_tf_record = tf.python_io.TFRecordWriter(args.output_path + 'val')

    for row in tqdm.tqdm(rows):
        tf_record_sample = create_tf_record_sample(*row)
        output_tf_record.write(tf_record_sample.SerializeToString())

def read_rows():
    rows = []
    with open(args.labels, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        fields = csv_reader.__next__()
        for row in csv_reader:
            row = row[:1] + list(map(int, row[1:]))
            rows.append(row)
    return rows

if __name__ == '__main__':
    rows = read_rows()
    split_idx = int(len(rows) * args.split)
    
    create_training(rows[:split_idx])
    create_val(row[split_idx:])
