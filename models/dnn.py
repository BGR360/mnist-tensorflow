"""
Copyright 2018 Ben Reeves

A Dense Neural Network model for classifying MNIST images.
Just returns the canned DNNClassifier Estimator provided by tf.estimator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_default_hparams():
    """Return the default hyperparameters for this model as an HParams object."""
    params = tf.contrib.training.HParams(
        hidden_units=[128, 64, 32],
        learning_rate=0.01,
        l1_regularization=0.001,
        dropout=None
    )
    return params


def get_estimator(run_config, params, feature_columns):
    """
    Return the model as a Tensorflow Estimator object.
    
    Args:
         run_config (RunConfig): Configuration for Estimator run.
         params (HParams): hyperparameters.
         feature_columns (Dict[name, FeatureColumn]): feature columns
    """
    # Create Estimator
    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=params.hidden_units,
        n_classes=params.n_classes,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=params.learning_rate,
            l1_regularization_strength=params.l1_regularization
        ),
        dropout=params.dropout,
        config=run_config
    )
    return estimator
