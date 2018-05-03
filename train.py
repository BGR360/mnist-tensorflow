"""
Copyright 2018 Ben Reeves

Trains an estimator on MNIST data.

Overall structure inspired by:
https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import configargparse
import tensorflow as tf

from dataset.MNISTDataset import MNISTDataset
from models import dnn

# Parsed later
FLAGS = None


def get_estimator(run_config, params, feature_columns):
    """Return an Estimator depending on model chosen at the command line."""
    if FLAGS.model == 'DNN':
        return dnn.get_estimator(run_config, params, feature_columns)
    raise TypeError("Invalid model {}".format(FLAGS.model))


def get_model_hparams():
    """Return the default hyperparameters for the model chosen at the command line."""
    if FLAGS.model == 'DNN':
        return dnn.get_default_hparams()
    raise TypeError("Invalid model {}".format(FLAGS.model))


def run_experiment(argv=None):
    # Define the default hyperparameters for the experiment
    params = get_model_hparams()
    # Override hyperparameters with any specified on the command line
    params.parse(FLAGS.hparams)
    print("Hyperparameters selected:")
    print(params)

    # Set the run_config and the directory to save the model and stats
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)
    run_config = run_config.replace(
        save_checkpoints_secs=FLAGS.save_checkpoints_secs)

    # Get training and evaluation data
    dataset = MNISTDataset(
        FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        shuffle=FLAGS.shuffle
    )
    train_input_fn = dataset.get_input_fn('train')
    eval_input_fn = dataset.get_input_fn('validation')

    # Specify feature columns
    feature_columns_dict = dataset.get_feature_columns()
    feature_columns = [
        feature_columns_dict['image_data']
    ]

    # Create estimator with the given hparams and run config
    estimator = get_estimator(run_config, params, feature_columns)

    # Set up train and eval specs
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=FLAGS.train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=FLAGS.eval_steps,
        start_delay_secs=FLAGS.eval_interval_secs,
        throttle_secs=FLAGS.eval_interval_secs
    )

    # Run the experiment
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.realpath(__file__))
    parser = configargparse.ArgParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add(
        '-c',
        '--config',
        is_config_file=True,
        help='Config file path')
    parser.add(
        '-m',
        '--model',
        choices=['DNN'],
        default='DNN',
        help='Which model type to use for classification.'
    )
    parser.add(
        '--model_dir',
        type=str,
        default=os.path.join(script_directory, 'checkpoints'),
        help='Where to save model checkpoints.'
    )
    parser.add(
        '--data_dir',
        type=str,
        default=os.path.join(script_directory, 'dataset/data'),
        help='Directory containing MNIST .tfrecords files.'
    )
    parser.add(
        '--batch_size',
        type=int,
        default=128
    )
    parser.add(
        '--shuffle',
        default=False,
        action='store_true',
        help='Shuffle dataset when iterating through it.'
    )
    parser.add(
        '--train_steps',
        type=int,
        default=5000,
        help='Maximum number of batches to train on.'
    )
    parser.add(
        '--eval_steps',
        type=int,
        default=50,
        help='How many batches to run during each evaluation run.'
    )
    parser.add(
        '--eval_interval_secs',
        type=int,
        default=30,
        help='Minimum interval between evaluation runs.'
    )
    parser.add(
        '--save_checkpoints_secs',
        type=int,
        default=30,
        help='How often to save model checkpoints.'
    )
    parser.add(
        '--hparams',
        type=str,
        default='',
        help='Hyperparameters for the estimator. '\
             'List of comma-separated name=value pairs.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run_experiment, argv=[sys.argv[0]] + unparsed)
