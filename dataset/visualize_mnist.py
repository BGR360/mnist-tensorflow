"""
Loads images from the MNIST TFRecords files and displays them in a grid.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from MNISTDataset import MNISTDataset, get_example_parser

# Parsed later
FLAGS = None

def input_fn():
    """Return iterators through the dataset."""
    dataset = MNISTDataset(
        FLAGS.directory,
        batch_size=FLAGS.batch_size,
        shuffle=FLAGS.shuffle,
        reshape=True
    )
    return dataset.get_input_fn(FLAGS.partition)()


def factorization(n):
    """
    Given a number, decompose it into roughly equal factors
    (e.g. if I want to display 12 images I should use a 3x4 grid).
    Courtesy of GitHub user kukuruza:
    https://gist.github.com/kukuruza/03731dc494603ceab0c5#file-gist_cifar10_train-py-L16
    """
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return i, n // i

# 
if __name__ == '__main__':
    """
    Iterate through batches of images stored in the .tfrecords files and display
    them in a grid using matplotlib.
    """
    script_directory = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default=os.path.join(script_directory, 'data'),
        help='Directory containing MNIST .tfrecords files'
    )
    parser.add_argument(
        '--partition',
        type=str,
        choices=['train', 'validation', 'test'],
        default='train',
        help='Which partition of images to visualize.'
    )
    parser.add_argument(
        '--shuffle',
        default=False,
        action='store_true',
        help='Whether to shuffle the data in the dataset.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='How many images to display at once.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # How big should the grid be?
    rows, cols = factorization(FLAGS.batch_size)

    # Create dataset iterators
    batch_features, batch_labels = input_fn()

    print("Click the figure to advance to the next batch")

    with tf.Session() as sess:
        while True:
            # Evaluate the batch tensors
            features, labels = sess.run([batch_features, batch_labels])
            images = features['image_data']
            # Display batch in a grid
            for idx in range(FLAGS.batch_size):
                img = images[idx]
                label = labels[idx]
                plt.subplot(rows, cols, idx+1)
                plt.title(label)
                plt.axis('off')
                plt.imshow(img)
            plt.draw()
            if plt.waitforbuttonpress(0) == None:
                break
