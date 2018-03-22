"""
Copyright 2018 Ben Reeves

Wrapper class for the MNIST dataset. Used as input to a tf.learn.Estimator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


class MNISTDataset(object):
	"""
	Wrapper class for the MNIST dataset. Used as input to a tf.learn.Estimator.
	"""
	def __init__(self, data_dir,
				 batch_size=128,
				 shuffle=False):
		"""
		Create an MNISTDataset.

		Args:
			data_dir: The directory in which the train, test, and validation
				.tfrecords files are stored.
			batch_size: Size of each batch.
			shuffle: Boolean value, whether to shuffle the dataset as we
				iterate through it.
		"""
		self.batch_size = batch_size
		self.shuffle = shuffle
		self._shuffle_buffer_size = 10000

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
		See data.fetch_mnist for how the dataset is saved to TFRecords.

		Args:
			filepath: Path to a .tfrecords file containing MNIST data.

		Returns:
			dataset: A tf.data.Dataset instance that yields (features, label)
				pairs when iterated over, where features is a dict:
					{'image_data': Tensor([batch_size, height, width, depth])}
				and label is a tensor:
					Tensor([batch_size])
		"""
		# Iterating through this will yield tf.data.Example protocol buffers.
		# So we must create a mapping function to parse these Examples.
		dataset = tf.data.TFRecordsDataset(filepath)

		def example_parser(example):
			"""
			Map a single tf.data.Example protocol buffer to a tuple of 
			(features, label).
			"""
			keys_to_features = {
				'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)
			}
			parsed = tf.parse_single_example(example, keys_to_features)

			# Perform additional preprocessing
			image_shape = [parsed['height'], parsed['width'], parsed['depth']]
			image = tf.image.decode_image(parsed['image_raw'])
			image = tf.reshape(image, image_shape)

			features = {'image_data': image}
			label = tf.cast(parsed['label'], tf.int32)

			return features, label

		# Use `Dataset.map()` to build a pair of a feature dictionary and a 
		# label tensor for each example.
		dataset = dataset.map(parser)

		if self.shuffle:
			dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size)

		# Split data into batches
		dataset = dataset.batch(self.batch_size)

		# Repeat the dataset forever
		dataset = dataset.repeat()

		return dataset

	def get_input_fn(partition_name):
		"""
		Return a function that can be passed to a tf.learn.Estimator as its
		input_fn parameter.

		Args:
			partition_name: One of "train", "test", or "validation"

		Returns:
			input_fn: A first-class function that returns a tuple of
				(features, labels)
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
