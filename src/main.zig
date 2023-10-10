const std = @import("std");
const mnist_data_utils = @import("mnist/mnist_data_utils.zig");
const neural_networks = @import("neural_networks/neural_networks.zig");
const LayerOutputData = @import("neural_networks/layer.zig").LayerOutputData;

// Adjust as necessary. To make the program run faster, you can reduce the number of
// images to train on and test on. To make the program more accurate, you can increase
// the number of images to train on (also try playing with the value of `k` in the model
// which influences K-nearest neighbor algorithm).
const NUM_OF_IMAGES_TO_TRAIN_ON = 10000; // (max 60k)
const NUM_OF_IMAGES_TO_TEST_ON = 100; // (max 10k)

const TRAINING_EPOCHS = 1000;
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

    // Read the MNIST data from the filesystem and normalize it.
    const raw_mnist_data = try mnist_data_utils.getMnistData(allocator, .{
        .num_images_to_train_on = NUM_OF_IMAGES_TO_TRAIN_ON,
        .num_images_to_test_on = NUM_OF_IMAGES_TO_TEST_ON,
    });
    defer raw_mnist_data.deinit(allocator);
    const normalized_raw_training_images = try mnist_data_utils.normalizeMnistRawImageData(
        raw_mnist_data.training_images,
        allocator,
    );
    defer allocator.free(normalized_raw_training_images);
    const normalized_raw_test_images = try mnist_data_utils.normalizeMnistRawImageData(
        raw_mnist_data.testing_images,
        allocator,
    );
    defer allocator.free(normalized_raw_test_images);

    const digit_labels = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const DigitDataPoint = neural_networks.DataPoint(u8, &digit_labels);

    // Convert the normalized MNIST data into `DigitDataPoint` which are compatible with the neural network
    const training_data_points = try allocator.alloc(DigitDataPoint, normalized_raw_training_images.len);
    defer allocator.free(training_data_points);
    for (normalized_raw_training_images, 0..) |raw_image, image_index| {
        training_data_points[image_index] = DigitDataPoint.init(
            &raw_image,
            raw_mnist_data.training_labels[image_index],
        );
    }
    const testing_data_points = try allocator.alloc(DigitDataPoint, normalized_raw_test_images.len);
    defer allocator.free(testing_data_points);
    for (normalized_raw_test_images, 0..) |raw_image, image_index| {
        testing_data_points[image_index] = DigitDataPoint.init(
            &raw_image,
            raw_mnist_data.testing_labels[image_index],
        );
    }

    var neural_network = try neural_networks.NeuralNetwork(DigitDataPoint).init(
        &[_]u32{ 784, 100, digit_labels.len },
        neural_networks.ActivationFunction{ .relu = .{} },
        neural_networks.CostFunction{ .mean_squared_error = .{} },
        allocator,
    );
    defer neural_network.deinit(allocator);

    var current_epoch_iteration_count: usize = 0;
    while (current_epoch_iteration_count < TRAINING_EPOCHS) : (current_epoch_iteration_count += 1) {
        try neural_network.learn(
            training_data_points,
            LEARN_RATE,
            allocator,
        );

        const cost = try neural_network.cost(testing_data_points, allocator);
        const accuracy = try neural_network.getAccuracyAgainstTestingDataPoints(
            testing_data_points,
            allocator,
        );
        std.log.debug("epoch {d} -> cost {d}, acccuracy {d}", .{
            current_epoch_iteration_count,
            cost,
            accuracy,
        });
    }
}
