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
