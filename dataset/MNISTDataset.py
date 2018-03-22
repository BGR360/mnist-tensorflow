"""
Copyright 2018 Ben Reeves

Wrapper class for the MNIST dataset. Used as input to a tf.learn.Estimator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf


def _int64_feature():
    return tf.FixedLenFeature([], tf.int64)


def _bytes_feature():
    return tf.FixedLenFeature([], tf.string)


class MNISTDataset(object):
    """
    Wrapper class for the MNIST dataset. Used as input to a tf.learn.Estimator.
    """
    def __init__(self, data_dir,
                 batch_size=128,
                 shuffle=False,
                 shuffle_buffer_size=10000):
        """
        Create an MNISTDataset.

        Args:
            data_dir: The directory in which the train, test, and validation
                .tfrecords files are stored.
            batch_size: Size of each batch.
            shuffle: Boolean value, whether to shuffle the dataset as we
                iterate through it. Must be `True` if `shuffle_buffer_size`
                is provided.
            shuffle_buffer_size: How many examples to keep in the random
                shuffle buffer used by `tf.data.Dataset.shuffle()`. Must set
                `shuffle` to True if this is provided.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        data_dir = os.path.realpath(data_dir)
        print("Creating MNISTDataset instance from {}".format(data_dir))

        self._filepaths = {}
        for partition_name in ['train', 'validation', 'test']:
            tfrecords_filename = partition_name + '.tfrecords'
            tfrecords_path = os.path.join(data_dir, tfrecords_filename)
            if not tf.gfile.Exists(tfrecords_path):
                raise IOError(
                    "Could not locate TFRecords file " + \
                    tfrecords_path + \
                    ". Try running:\n" + \
                    "\tpython data/fetch_mnist.py"
                )
            self._filepaths[partition_name] = tfrecords_path

    def _load_dataset_from_tfrecords(self, filepath):
        """
        Return a tf.data.Dataset object loaded from a TFRecords file.
        See `dataset.fetch_mnist` for how the dataset is saved to TFRecords.

        Args:
            filepath: Path to a .tfrecords file containing MNIST data.

        Returns:
            dataset: A `tf.data.Dataset` instance that yields (features, label)
                pairs when iterated over, where features is a dict:
                    `{'image_data': Tensor([batch_size,height,width,depth])}`
                and label is a tensor:
                    `Tensor([batch_size])`
        """
        # Iterating through this will yield tf.data.Example protocol buffers.
        # So we must create a mapping function to parse these Examples.
        dataset = tf.data.TFRecordDataset(filepath)

        def example_parser(example):
            """
            Map a single `tf.data.Example` protocol buffer to a tuple of 
            `(features, label)`.
            """
            keys_to_features = {
                'height': _int64_feature(),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': _bytes_feature()
            }
            parsed = tf.parse_single_example(example, keys_to_features)

            # Perform additional preprocessing
            image_shape = tf.stack(
                [parsed['height'], parsed['width'], parsed['depth']])
            image = tf.decode_raw(parsed['image_raw'], tf.float32)
            image = tf.reshape(image, image_shape)

            features = {'image_data': image}
            label = tf.cast(parsed['label'], tf.int32)

            return features, label

        # Use `Dataset.map()` to build a pair of a feature dictionary and a 
        # label tensor for each example.
        dataset = dataset.map(example_parser)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size)

        # Repeat the dataset forever
        dataset = dataset.repeat()

        # Split data into batches
        dataset = dataset.batch(self.batch_size)

        return dataset

    def get_input_fn(self, partition_name):
        """
        Return a function that can be passed to a `tf.estimator.Estimator`
        as its input_fn parameter.

        If any properties of this class are changed (e.g. batch_size, shuffle),
        you must re-call this function to get an updated input_fn.

        Args:
            partition_name: One of "train", "test", or "validation"

        Returns:
            input_fn: A first-class function that returns a tuple of
                `(features, labels)`
        """
        tfrecords_path = self._filepaths[partition_name]
        dataset = self._load_dataset_from_tfrecords(tfrecords_path)
        def input_fn():
            # Create an Iterator to iterate over the Dataset
            iterator = dataset.make_one_shot_iterator()
            # Evaluating these tensors will advance the Iterator
            features, labels = iterator.get_next()
            # `features` is a dictionary in which each value is a batch of 
            # values for that feature; `labels` is a batch of labels.
            return features, labels
        return input_fn


# Test that the class works by loading the dataset and iterating over the
# first couple batches.
if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default=os.path.join(script_directory, 'data'),
        help='Directory containing MNIST .tfrecords files'
    )
    FLAGS, unparsed = parser.parse_known_args()
    
    print("Testing the functionality of MNISTDataset")
    dataset = MNISTDataset(FLAGS.directory, batch_size=32, shuffle=False)
    input_fn = dataset.get_input_fn('train')
    batch_features, batch_labels = input_fn()

    with tf.Session() as sess:
        for i in range(4):
            print("** Batch {} **".format(i))
            features, labels = sess.run([batch_features, batch_labels])
            images = features['image_data']
            print("images shape: {}".format(images.shape))
            print("labels shape: {}".format(labels.shape))
            #print("images[0]: {}".format(images[0]))
            #print("labels[0]: {}".format(labels[0]))
