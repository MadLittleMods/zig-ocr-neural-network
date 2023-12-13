# Basic OCR example using a neural network against the MNIST dataset

> [!TIP]
If you are looking for a more complete neural network library that you can
> include in your project, I've created a library based on this project ->
> https://github.com/MadLittleMods/zig-neural-networks

A from scratch neural network implementation in Zig, trained against the MNIST dataset
to recognize handwritten digits.

A lot of the phrasing and concepts are taken from the resources linked in the
[*developer notes*](./dev-notes.md). I'm just trying to piece together all of those
resources into something that works and is understandable to me as I learn. Major kudos
to Sebastian Lague, 3Blue1Brown, The Independent Code (Omar Aflak), and Samson Zhang for
their excellent resources. And a special shoutout to Hans Musgrave
([@hmusgrave](https://github.com/hmusgrave)) for the immense amount of help to get my
head around these concepts as I got stuck through this process.

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
debug: Created normalized data points. Training on 60000 data points, testing on 10000
debug: Here is what the first training data point looks like:
┌──────────┐
│ Label: 5 │
┌────────────────────────────────────────────────────────┐
│                                                        │
│                                                        │
│                                                        │
│                                                        │
│                                                        │
│                        ░░░░░░░░▒▒▓▓▓▓░░▓▓████▒▒        │
│                ░░░░▒▒▓▓▓▓████████████▓▓██████▒▒        │
│              ░░████████████████████▒▒▒▒▒▒░░░░          │
│              ░░██████████████▓▓████                    │
│                ▒▒▓▓▒▒██████░░  ░░▓▓                    │
│                  ░░░░▓▓██▒▒                            │
│                      ▓▓██▓▓░░                          │
│                      ░░▓▓██▒▒                          │
│                        ░░████▓▓▒▒░░                    │
│                          ▒▒██████▒▒░░                  │
│                            ░░▓▓████▓▓░░                │
│                              ░░▒▒████▓▓                │
│                                  ██████▒▒              │
│                            ░░▓▓▓▓██████░░              │
│                        ░░▓▓██████████▓▓                │
│                    ░░▒▒████████████▒▒                  │
│                ░░▒▒████████████▒▒░░                    │
│            ░░▓▓████████████▒▒░░                        │
│        ░░▓▓████████████▓▓░░                            │
│        ▓▓████████▓▓▓▓░░                                │
│                                                        │
│                                                        │
│                                                        │
└────────────────────────────────────────────────────────┘
debug: epoch 0   batch 0             3s -> cost 331.64265899045563, accuracy with 100 test points 0.11
debug: epoch 0   batch 5             4s -> cost 242.16033395427667, accuracy with 100 test points 0.56
debug: epoch 0   batch 10            5s -> cost 155.62913461977217, accuracy with 100 test points 0.7
debug: epoch 0   batch 15            5s -> cost 118.45908401769115, accuracy with 100 test points 0.75
[...]
```

### Other examples

#### Simple animal example 

This is a small dataset that I made up to test the neural network. There are only 2
arbitrary features (x and y) where the labeled data points (fish and goat) occupy
distinct parts of the graph. Since there are only 2 input features (which means 2
dimensions), we can easily graph the neural network's decision/classification boundary.
It's a good way to visualize the neural network and see how it evolves while training.

This example produces an image called `simple_xy_animal_graph.ppm` every 1,000 epochs
showing the decision/classification boundary.

```sh
$ zig build run-simple_xy_animal_sample
```

![](https://github.com/MadLittleMods/zig-ocr-neural-network/assets/558581/128ca52f-0f6f-42ae-8d7e-c557ad943706)


#### Barebones XOR example

There is also a barebones XOR example which just trains a neural network to act like a
XOR ("exclusive or") gate.

```sh
$ zig build run-xor
```

![](https://github.com/MadLittleMods/zig-ocr-neural-network/assets/558581/887e7323-41e7-4fda-aa58-5989dc437f97)


## Testing

```sh
$ zig build test --summary all
```

## Dev notes

See the [*developer notes*](./dev-notes.md) for more information.
