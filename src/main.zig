const std = @import("std");
const mnist_data_utils = @import("mnist/mnist_data_utils.zig");
const neural_networks = @import("neural_networks/neural_networks.zig");
const NeuralNetwork = neural_networks.NeuralNetwork;

const TRAIN_DATA_FILE_PATH = "data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE_PATH = "data/train-labels-idx1-ubyte";
const TEST_DATA_FILE_PATH = "data/t10k-images-idx3-ubyte";
const TEST_LABELS_FILE_PATH = "data/t10k-labels-idx1-ubyte";

// Adjust as necessary. To make the program run faster, you can reduce the number of
// images to train on and test on. To make the program more accurate, you can increase
// the number of images to train on (also try playing with the value of `k` in the model
// which influences K-nearest neighbor algorithm).
const NUMBER_OF_IMAGES_TO_TRAIN_ON = 10000; // (max 60k)
const NUMBER_OF_IMAGES_TO_TEST_ON = 100; // (max 10k)

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    // Read in the MNIST training labels
    const training_labels_data = try mnist_data_utils.readMnistFile(
        mnist_data_utils.MnistLabelFileHeader,
        mnist_data_utils.LabelType,
        TRAIN_LABELS_FILE_PATH,
        "number_of_labels",
        NUMBER_OF_IMAGES_TO_TRAIN_ON,
        allocator,
    );
    defer allocator.free(training_labels_data.items);
    std.log.debug("training labels header {}", .{training_labels_data.header});
    try std.testing.expectEqual(training_labels_data.header.magic_number, 2049);
    try std.testing.expectEqual(training_labels_data.header.number_of_labels, 60000);

    // Read in the MNIST training images
    const training_images_data = try mnist_data_utils.readMnistFile(
        mnist_data_utils.MnistImageFileHeader,
        mnist_data_utils.RawImageData,
        TRAIN_DATA_FILE_PATH,
        "number_of_images",
        NUMBER_OF_IMAGES_TO_TRAIN_ON,
        allocator,
    );
    defer allocator.free(training_images_data.items);
    std.log.debug("training images header {}", .{training_images_data.header});
    try std.testing.expectEqual(training_images_data.header.magic_number, 2051);
    try std.testing.expectEqual(training_images_data.header.number_of_images, 60000);
    try std.testing.expectEqual(training_images_data.header.number_of_rows, 28);
    try std.testing.expectEqual(training_images_data.header.number_of_columns, 28);

    // Read in the MNIST testing labels
    const testing_labels_data = try mnist_data_utils.readMnistFile(
        mnist_data_utils.MnistLabelFileHeader,
        mnist_data_utils.LabelType,
        TEST_LABELS_FILE_PATH,
        "number_of_labels",
        NUMBER_OF_IMAGES_TO_TEST_ON,
        allocator,
    );
    defer allocator.free(testing_labels_data.items);
    std.log.debug("testing labels header {}", .{testing_labels_data.header});
    try std.testing.expectEqual(testing_labels_data.header.magic_number, 2049);
    try std.testing.expectEqual(testing_labels_data.header.number_of_labels, 10000);

    // Read in the MNIST testing images
    const testing_images_data = try mnist_data_utils.readMnistFile(
        mnist_data_utils.MnistImageFileHeader,
        mnist_data_utils.RawImageData,
        TEST_DATA_FILE_PATH,
        "number_of_images",
        NUMBER_OF_IMAGES_TO_TEST_ON,
        allocator,
    );
    defer allocator.free(testing_images_data.items);
    std.log.debug("testing images header {}", .{testing_images_data.header});
    try std.testing.expectEqual(testing_images_data.header.magic_number, 2051);
    try std.testing.expectEqual(testing_images_data.header.number_of_images, 10000);
    try std.testing.expectEqual(testing_images_data.header.number_of_rows, 28);
    try std.testing.expectEqual(testing_images_data.header.number_of_columns, 28);

    const layer_sizes = [_]u32{ 784, 100, 10 };
    var neural_network = NeuralNetwork(layer_sizes[0..]).init(neural_networks.ActivationFunction.Relu, allocator);
    _ = neural_network;
}
