const std = @import("std");
const mnist_data_utils = @import("mnist/mnist_data_utils.zig");
const neural_networks = @import("neural_networks/neural_networks.zig");
const LayerOutputData = @import("neural_networks/layer.zig").LayerOutputData;
const time_utils = @import("utils/time_utils.zig");

// Adjust as necessary. To make the program run faster, you can reduce the number of
// images to train on and test on. To make the program more accurate, you can increase
// the number of images to train on (also try playing with the value of `k` in the model
// which influences K-nearest neighbor algorithm).
const NUM_OF_IMAGES_TO_TRAIN_ON = 10000; // (max 60k)
const NUM_OF_IMAGES_TO_TEST_ON = 100; // (max 10k)

// The number of times to run through the whole training data set.
const TRAINING_EPOCHS = 1000;
const BATCH_SIZE: u32 = 32;
const LEARN_RATE: f64 = 0.1;
const MOMENTUM = 0.9;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    const start_timestamp_seconds = std.time.timestamp();

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
    std.log.debug("Created normalized data points. Training on {d} data points, testing on {d}", .{
        training_data_points.len,
        testing_data_points.len,
    });

    var neural_network = try neural_networks.NeuralNetwork(DigitDataPoint).init(
        &[_]u32{ 784, 100, digit_labels.len },
        neural_networks.ActivationFunction{
            // .relu = .{},
            .leaky_relu = .{},
            //.sigmoid = .{},
        },
        neural_networks.ActivationFunction{ .soft_max = .{} },
        neural_networks.CostFunction{ .mean_squared_error = .{} },
        allocator,
    );
    defer neural_network.deinit(allocator);

    std.log.debug("initial layer weights {d:.3}", .{neural_network.layers[1].weights});
    std.log.debug("initial layer biases {d:.3}", .{neural_network.layers[1].biases});

    var current_epoch_index: usize = 0;
    while (true
    // current_epoch_index < TRAINING_EPOCHS
    ) : (current_epoch_index += 1) {
        // Split the training data into mini batches so way we can get through learning
        // iterations faster. It does make the learning progress a bit noisy because the
        // cost landscape is a bit different for each batch but it's fast and apparently
        // the noise can even be beneficial in various ways, like for escaping settle
        // points in the cost gradient (ridgelines between two valleys).
        //
        // Instead of "gradient descent" with the full training set, using mini batches
        // is called "stochastic gradient descent".
        var batch_index: u32 = 0;
        while (batch_index < NUM_OF_IMAGES_TO_TRAIN_ON / BATCH_SIZE) : (batch_index += 1) {
            const batch_start_index = batch_index * BATCH_SIZE;
            const batch_end_index = batch_start_index + BATCH_SIZE;
            const training_batch = training_data_points[batch_start_index..batch_end_index];

            // try neural_network.learn_estimate(
            try neural_network.learn(
                training_batch,
                LEARN_RATE,
                MOMENTUM,
                allocator,
            );

            // std.log.debug("layer weights {d:.3}", .{neural_network.layers[1].weights});
            // std.log.debug("layer biases {d:.3}", .{neural_network.layers[1].biases});

            if (current_epoch_index % 1 == 0 and
                current_epoch_index != 0 and
                batch_index == 0)
            {
                const current_timestamp_seconds = std.time.timestamp();
                const runtime_duration_seconds = current_timestamp_seconds - start_timestamp_seconds;
                const duration_string = try time_utils.formatDuration(
                    runtime_duration_seconds * time_utils.ONE_SECOND_MS,
                    allocator,
                );
                defer allocator.free(duration_string);

                const cost = try neural_network.cost_many(testing_data_points, allocator);
                const accuracy = try neural_network.getAccuracyAgainstTestingDataPoints(
                    testing_data_points,
                    allocator,
                );
                std.log.debug("epoch {d: <3} batch {d: <3} {s: >12} -> cost {d}, acccuracy with test points {d}", .{
                    current_epoch_index,
                    batch_index,
                    duration_string,
                    cost,
                    accuracy,
                });
            }
        }
    }
}
