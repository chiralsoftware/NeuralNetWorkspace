#!/bin/bash

echo "Downloading the MINST data"
echo "Based on URLs in:"
echo "https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py"

BASE_URL=https://storage.googleapis.com/cvdf-datasets/mnist/
wget "$BASE_URL"train-images-idx3-ubyte.gz
wget "$BASE_URL"train-labels-idx1-ubyte.gz
wget "$BASE_URL"t10k-images-idx3-ubyte.gz
wget "$BASE_URL"t10k-labels-idx1-ubyte.gz
