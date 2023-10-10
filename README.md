# Basic OCR example using a neural network against the MNIST dataset

A from scratch neural network implementation in Zig, trained against the MNIST dataset
to recognize handwritten digits.

A lot of the phrasing and concepts are taken from the resources linked in the
[*developer notes*](./dev-notes.md). I'm just trying to piece together all of those
resources into something that works and is understandable to me as I learn. Major kudos
to Sebastian Lague, 3Blue1Brown, and Samson Zhang for their excellent resources.

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

To run a basic graphable dataset example, you can use the `simple_xy_animal_sample_main.zig`:

```sh
$ zig run src/simple_xy_animal_sample_main.zig
```

## Dev notes

See the [*developer notes*](./dev-notes.md) for more information.
