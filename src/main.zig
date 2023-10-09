const std = @import("std");
const mnist_data_utils = @import("mnist/mnist_data_utils.zig");
const neural_networks = @import("neural_networks/neural_networks.zig");
const NeuralNetwork = neural_networks.NeuralNetwork;
const DataPoint = neural_networks.DataPoint;

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

const EPOCHS = 25;
const LEARN_RATE: f64 = 0.1;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    // // Read in the MNIST training labels
    // const training_labels_data = try mnist_data_utils.readMnistFile(
    //     mnist_data_utils.MnistLabelFileHeader,
    //     mnist_data_utils.LabelType,
    //     TRAIN_LABELS_FILE_PATH,
    //     "number_of_labels",
    //     NUMBER_OF_IMAGES_TO_TRAIN_ON,
    //     allocator,
    // );
    // defer allocator.free(training_labels_data.items);
    // std.log.debug("training labels header {}", .{training_labels_data.header});
    // try std.testing.expectEqual(training_labels_data.header.magic_number, 2049);
    // try std.testing.expectEqual(training_labels_data.header.number_of_labels, 60000);

    // // Read in the MNIST training images
    // const training_images_data = try mnist_data_utils.readMnistFile(
    //     mnist_data_utils.MnistImageFileHeader,
    //     mnist_data_utils.RawImageData,
    //     TRAIN_DATA_FILE_PATH,
    //     "number_of_images",
    //     NUMBER_OF_IMAGES_TO_TRAIN_ON,
    //     allocator,
    // );
    // defer allocator.free(training_images_data.items);
    // std.log.debug("training images header {}", .{training_images_data.header});
    // try std.testing.expectEqual(training_images_data.header.magic_number, 2051);
    // try std.testing.expectEqual(training_images_data.header.number_of_images, 60000);
    // try std.testing.expectEqual(training_images_data.header.number_of_rows, 28);
    // try std.testing.expectEqual(training_images_data.header.number_of_columns, 28);

    // // Read in the MNIST testing labels
    // const testing_labels_data = try mnist_data_utils.readMnistFile(
    //     mnist_data_utils.MnistLabelFileHeader,
    //     mnist_data_utils.LabelType,
    //     TEST_LABELS_FILE_PATH,
    //     "number_of_labels",
    //     NUMBER_OF_IMAGES_TO_TEST_ON,
    //     allocator,
    // );
    // defer allocator.free(testing_labels_data.items);
    // std.log.debug("testing labels header {}", .{testing_labels_data.header});
    // try std.testing.expectEqual(testing_labels_data.header.magic_number, 2049);
    // try std.testing.expectEqual(testing_labels_data.header.number_of_labels, 10000);

    // // Read in the MNIST testing images
    // const testing_images_data = try mnist_data_utils.readMnistFile(
    //     mnist_data_utils.MnistImageFileHeader,
    //     mnist_data_utils.RawImageData,
    //     TEST_DATA_FILE_PATH,
    //     "number_of_images",
    //     NUMBER_OF_IMAGES_TO_TEST_ON,
    //     allocator,
    // );
    // defer allocator.free(testing_images_data.items);
    // std.log.debug("testing images header {}", .{testing_images_data.header});
    // try std.testing.expectEqual(testing_images_data.header.magic_number, 2051);
    // try std.testing.expectEqual(testing_images_data.header.number_of_images, 10000);
    // try std.testing.expectEqual(testing_images_data.header.number_of_rows, 28);
    // try std.testing.expectEqual(testing_images_data.header.number_of_columns, 28);

    // const labels = [_]u8{
    //     0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // };

    // const MnistDataPoint = DataPoint(u8, labels);

    // var neural_network = try NeuralNetwork.init(
    //     [_]u32{ 784, 100, 10 },
    //     neural_networks.ActivationFunction.Relu,
    //     allocator,
    // );
    // _ = neural_network;

    const animal_labels = [_][]const u8{ "fish", "goat" };
    const AnimalDataPoint = DataPoint([]const u8, &animal_labels);
    // Graph of animal data points:
    // https://www.desmos.com/calculator/x72k0x9ies
    const animal_training_data_points = [_]AnimalDataPoint{
        AnimalDataPoint.init(&[_]f64{ 0.214, 0.049 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.214, 0.238 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.086, 0.155 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.314, 0.692 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.145, 0.159 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.596, 0.054 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.192, 0.348 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.132, 0.28 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.728, 0.208 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.097, 0.05 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.445, 0.384 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.517, 0.416 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.323, 0.573 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.683, 0.88 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.662, 0.737 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.195, 0.469 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.314, 0.771 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.38, 0.43 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.865, 0.525 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.806, 0.274 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.262, 0.426 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.472, 0.604 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.538, 0.51 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.047, 0.427 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.762, 0.39 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.881, 0.382 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.643, 0.277 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.41, 0.085 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.917, 0.267 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.535, 0.216 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.766, 0.62 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.342, 0.142 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.587, 0.138 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.673, 0.135 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.59, 0.452 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.331, 0.263 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.466, 0.504 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.064, 0.554 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.663, 0.61 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.31, 0.37 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.49, 0.075 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.502, 0.324 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.622, 0.368 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.21, 0.573 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.065, 0.318 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.375, 0.316 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.49, 0.75 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.67, 0.222 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.724, 0.049 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.135, 0.404 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.123, 0.517 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.136, 0.726 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.246, 0.315 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.924, 0.166 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.863, 0.068 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.586, 0.653 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.89, 0.746 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.096, 0.827 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.73, 0.507 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.04, 0.085 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.364, 0.042 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.436, 0.182 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.768, 0.14 }, "goat"),
    };
    const animal_testing_data_points = [_]AnimalDataPoint{
        AnimalDataPoint.init(&[_]f64{ 0.23, 0.14 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.087, 0.236 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.507, 0.142 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.486, 0.383 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.67, 0.076 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.085, 0.402 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.41, 0.257 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.278, 0.273 }, "fish"),
        AnimalDataPoint.init(&[_]f64{ 0.37, 0.674 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.32, 0.43 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.066, 0.628 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.635, 0.527 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.704, 0.305 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.82, 0.137 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.862, 0.305 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.709, 0.679 }, "goat"),
        AnimalDataPoint.init(&[_]f64{ 0.18, 0.527 }, "goat"),
    };
    _ = animal_testing_data_points;

    var neural_network = try NeuralNetwork(AnimalDataPoint).init(
        &[_]u32{ 2, 3, animal_labels.len },
        neural_networks.ActivationFunction.Relu,
        neural_networks.CostFunction.MeanSquaredError,
        allocator,
    );
    const epoch_count = 0;
    while (epoch_count < EPOCHS) : (epoch_count += 1) {
        neural_network.learn(animal_training_data_points, LEARN_RATE);

        const cost = neural_network.cost(animal_training_data_points);
        std.log.debug("epoch {d} -> cost {d}", .{ epoch_count, cost });
    }
}
