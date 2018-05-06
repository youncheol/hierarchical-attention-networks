import tensorflow as tf
import numpy as np

def parser(serialized_example):
    features = {
        'document': tf.FixedLenFeature([400], tf.int64),
        'label': tf.FixedLenFeature([5], tf.int64)
    }

    parsed_feature = tf.parse_single_example(serialized_example, features)

    document = parsed_feature['document']
    label = parsed_feature['label']

    return document, label


def read_tfrecord(fname, parser, shuffle_size, batch_size, sent_num, word_num):
    dataset = tf.data.TFRecordDataset(fname).map(parser)
    dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    feature, label = iterator.get_next()
    feature = tf.reshape(feature, [-1, sent_num, word_num])
    feature = tf.cast(feature, tf.int64)
    return iterator, dataset, feature, label
