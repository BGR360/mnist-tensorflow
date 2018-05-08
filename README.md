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
usage: train.py [-h] [-c CONFIG] [-m {DNN}] [--model_dir MODEL_DIR]
                [--data_dir DATA_DIR] [--batch_size BATCH_SIZE] [--shuffle]
                [--train_steps TRAIN_STEPS] [--eval_steps EVAL_STEPS]
                [--eval_interval_secs EVAL_INTERVAL_SECS]
                [--save_checkpoints_secs SAVE_CHECKPOINTS_SECS]
                [--hparams HPARAMS]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Config file path (default: None)
  -m {DNN}, --model {DNN}
                        Which model type to use for classification. (default:
                        DNN)
  --model_dir MODEL_DIR
                        Where to save model checkpoints. (default: /Users/Ben/
                        Documents/CodingProjects/MachineLearning/mnist-
                        tensorflow/checkpoints)
  --data_dir DATA_DIR   Directory containing MNIST .tfrecords files. (default:
                        /Users/Ben/Documents/CodingProjects/MachineLearning/mn
                        ist-tensorflow/dataset/data)
  --batch_size BATCH_SIZE
  --shuffle             Shuffle dataset when iterating through it. (default:
                        False)
  --train_steps TRAIN_STEPS
                        Maximum number of batches to train on. (default: 5000)
  --eval_steps EVAL_STEPS
                        How many batches to run during each evaluation run.
                        (default: 50)
  --eval_interval_secs EVAL_INTERVAL_SECS
                        Minimum interval between evaluation runs. (default:
                        30)
  --save_checkpoints_secs SAVE_CHECKPOINTS_SECS
                        How often to save model checkpoints. (default: 30)
  --hparams HPARAMS     Hyperparameters for the estimator. List of comma-
                        separated name=value pairs. (default: )
```

**TensorBoard**

```
$ python -m tensorboard.main --logdir=checkpoints/
```
