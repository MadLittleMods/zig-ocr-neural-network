# Basic OCR example using a neural network against the MNIST dataset

A from scratch neural network implementation in Zig, trained against the MNIST dataset
to recognize handwritten digits.

A lot of the phrasing and concepts are taken from the resources linked in the
[*developer notes*](./dev-notes.md). I'm just trying to piece together all of those
resources into something that works and is understandable to me as I learn.

## Setup

Download and extract the MNIST dataset from http://yann.lecun.com/exdb/mnist/ to a
directory called `data/` in the root of this project. Here is a copy-paste command
you can run:

```sh
# Make a data/ directory
mkdir data/ &&
cd data/ &&
# Download the MNIST dataset
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz &&
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz &&
# Unzip the files
gunzip *.gz
```


## Building and running

Tested with Zig 0.11.0

```sh
$ zig build run
...
```

## Dev notes

See the [*developer notes*](./dev-notes.md) for more information.
