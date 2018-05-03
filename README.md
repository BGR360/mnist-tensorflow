# MNIST Classifier Using TensorFlow

This is just for practice using TensorFlow's higher-level Estimator API.

## Setup

1. Clone the repo

2. Install dependencies and setup Anaconda environment

```
$ conda env update --file environment.yml
$ conda activate mnist-tensorflow
```

3. Fetch MNIST dataset and convert to a TFRecords file

```
$ python data/fetch_mnist.py
```

## Train a Model

The `train.py` script trains a model and saves checkpoints in the `checkpoints/` directory. You can select which type of model to use with the `--model` command line flag.

Currently supports the following models:

* `DNN`: Dense neural network with configurable layers.

**Training Script**

```
$ python train.py --help
usage: train.py [-h] [-c CONFIG] [--model {DNN}] [--model_dir MODEL_DIR]
                [--data_dir DATA_DIR] [--batch_size BATCH_SIZE] [--shuffle]
                [--hparams HPARAMS]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Config file path
  --model {DNN}         Which model type to use for classification
  --model_dir MODEL_DIR
  --data_dir DATA_DIR   Directory containing MNIST .tfrecords files
  --batch_size BATCH_SIZE
  --shuffle             Shuffle dataset when iterating through it
  --hparams HPARAMS     Hyperparameters for the estimator. List of comma-
                        separated name=value pairs.
```

**TensorBoard**

```
$ python -m tensorboard.main --logdir=checkpoints/
```
