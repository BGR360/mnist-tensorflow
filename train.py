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


def experiment_fn(run_config, params):
    """
    Create an experiment to train and evaluate the model.
    
    Args:
        run_config (RunConfig): Configuration for Estimator run.
        params (HParams): Hyperparameters
    
    Returns:
        (Experiment) Experiment for training the mnist model.
    """
    # Get training and evaluation data
    dataset = MNISTDataset(
        FLAGS.data_dir,
        batch_size=FLAGS.batch_size,
        shuffle=FLAGS.shuffle
    )
    train_input_fn = dataset.get_input_fn('train')
    eval_input_fn = dataset.get_input_fn('validation')
    
    # Create an Estimator
    feature_columns_dict = dataset.get_feature_columns()
    feature_columns = [feature_columns_dict['image_data']]
    estimator = get_estimator(run_config, params, feature_columns)

    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        min_eval_frequency=params.min_eval_frequency,  # Eval frequency
        eval_steps=params.eval_steps
    )
    return experiment


def run_experiment(argv=None):
    # Define the hyperparameters for the experiment
    params = get_model_hparams()
    params.add_hparam('n_classes', 10)
    params.add_hparam('train_steps', 5000)
    params.add_hparam('eval_steps', 1000)
    params.add_hparam('min_eval_frequency', 100)
    # Override hyperparameters with any specified on the command line
    params.parse(FLAGS.hparams)

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )


if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.realpath(__file__))
    parser = configargparse.ArgParser()
    parser.add(
        '-c',
        '--config',
        is_config_file=True,
        help='Config file path')
    parser.add(
        '--model_dir',
        type=str,
        default=os.path.join(script_directory, 'checkpoints'),
    )
    parser.add(
        '--data_dir',
        type=str,
        default=os.path.join(script_directory, 'dataset/data'),
        help='Directory containing MNIST .tfrecords files'
    )
    parser.add(
        '--model',
        choices=['DNN'],
        default='DNN',
        help='Which model type to use for classification'
    )
    parser.add(
        '--batch_size',
        type=int,
        default=128,
    )
    parser.add(
        '--shuffle',
        default=False,
        action='store_true',
        help='Shuffle dataset when iterating through it'
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
