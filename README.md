# Basic OCR example using a neural network against the MNIST dataset

A from scratch neural network implementation in Zig, trained against the MNIST dataset
to recognize handwritten digits.

A lot of the phrasing and concepts are taken from the resources linked in the
[*developer notes*](./dev-notes.md). I'm just trying to piece together all of those
resources into something that works and is understandable to me as I learn. Major kudos
to Sebastian Lague, 3Blue1Brown, and Samson Zhang for their excellent resources. And a
special shoutout to Hans Musgrave ([@hmusgrave](https://github.com/hmusgrave)) for the
immense amount of help to get my head around these concepts as I got stuck through this
process.

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

With the OCR example, on my machine, I can complete 1 epoch of training in ~1 minute
which gets to 94% accuracy and creeps to 97% after a few more epochs (60k training
images, 10k test images):
```sh
$ zig build run-mnist_ocr
```

To run a basic graphable dataset example that we can visualize, you can use the
`simple_xy_animal_sample_main.zig` example which produces an image called
`simple_xy_animal_graph.ppm` every 1,000 epochs showing the classification boundary.

```sh
$ zig build run-simple_xy_animal_sample
```

![](https://github.com/MadLittleMods/zig-ocr-neural-network/assets/558581/128ca52f-0f6f-42ae-8d7e-c557ad943706)


There is also the most barebones XOR example which just trains a neural network to act
like a XOR gate.

```sh
$ zig build run-xor
```


## Testing

```sh
$ zig build test --summary all
```

## Dev notes

See the [*developer notes*](./dev-notes.md) for more information.
