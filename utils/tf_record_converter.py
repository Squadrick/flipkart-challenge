import argparse
import csv
import os

import contextlib2
import tensorflow

parser = argparse.ArgumentParser(description = 'Script to convert Flipkart AI Challenge dataset to TF Record')
parser.add_argument('--images', type=str, help='Location of images folder')
parser.add_argument('--labels', type=str, help='CSV file containing the image names and box coordinates')
parser.add_argument('--output_path', type=str, help='The newly created TF Record file(s)')
parser.add_argument('--num_shards', type=int, help='Number of shards to split the dataset into, -1 for no sharding', default = -1)
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
    print(file_name, x1, x2, y1, y2)
    height = 480.0
    width = 640.0
    xmins = [x1 / width]
    xmaxs = [x2 / width]
    ymins = [y1 / height]
    ymaxs = [y2 / height]
    classes_test = ['Object']
    classes = [1]
    encoded_cat_image_data = open(file_name)
    image_format = b'png'
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/encoded': dataset_util.bytes_feature(encoded_cat_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

def create_tf_record():
    if args.shards:
        tf_record_close_stack = contextlib2.ExitStack()
        output_tf_records = open_sharded_output_tfrecords(
                tf_record_close_stack, args.output_path, args.shards)
    else:
        output_tf_record = tf.python_io.TFRecordWriter(args.TFRecordWriter(args.output_path))

    with open(args.labels, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        fields = csv_reader.__next__()
        print(fields)
        for enumerate(row) in index, csv_reader:
            row = row[:1] + list(map(int, row[1:]))
            tf_record_sample = create_tf_record_sample(*row)

            if args.shards:
                output_tf_record = output_tf_records[index % args.shards]

            output_tf_record.write(tf_record_sample.SerializeToString())

    if args.shards:
        for tf_record in output_tf_records:
            tf_record.close()
    else:
        output_tf_record.close()


if __name__ == '__main__':
    create_tf_record()
