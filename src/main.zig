const std = @import("std");
const shuffle = @import("zshuffle").shuffle;
const mnist_data_utils = @import("mnist/mnist_data_utils.zig");
const mnist_print_utils = @import("mnist/print_utils.zig");
const neural_networks = @import("neural_networks/neural_networks.zig");
const LayerOutputData = @import("neural_networks/layer.zig").LayerOutputData;
const time_utils = @import("utils/time_utils.zig");

// Adjust as necessary. To make the program run faster, you can reduce the number of
// images to train on and test on. To make the program more accurate, you can increase
// the number of images to train on.
const NUM_OF_IMAGES_TO_TRAIN_ON = 60000; // (max 60k)
// The number of test points to use when we do a full cost breakdown after each epoch
const NUM_OF_IMAGES_TO_TEST_ON = 10000; // (max 10k)
// We only use a small portion of test points when calculating cost and accuracy while
// going through the mini-batches in each epoch. This is to make the program run faster.
// The full cost breakdown is done after each epoch.
const NUM_OF_IMAGES_TO_QUICK_TEST_ON = 100; // (max 10k)

// The number of times to run through the whole training data set.
const TRAINING_EPOCHS = 1000;
const BATCH_SIZE: u32 = 1;
const LEARN_RATE: f64 = 0.05;
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

    // XXX: We can use `@intCast(u64, std.time.timestamp())` to get a seed that changes
    // but it's nicer to have a fixed seed so we can reproduce the same results.
    const seed = 123;
    var prng = std.rand.DefaultPrng.init(seed);
    const random_instance = prng.random();

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
    var training_data_points = try allocator.alloc(DigitDataPoint, normalized_raw_training_images.len);
    defer allocator.free(training_data_points);
    for (normalized_raw_training_images, 0..) |*raw_image, image_index| {
        training_data_points[image_index] = DigitDataPoint.init(
            raw_image,
            raw_mnist_data.training_labels[image_index],
        );
    }
    const testing_data_points = try allocator.alloc(DigitDataPoint, normalized_raw_test_images.len);
    defer allocator.free(testing_data_points);
    for (normalized_raw_test_images, 0..) |*raw_image, image_index| {
        testing_data_points[image_index] = DigitDataPoint.init(
            raw_image,
            raw_mnist_data.testing_labels[image_index],
        );
    }
    std.log.debug("Created normalized data points. Training on {d} data points, testing on {d}", .{
        training_data_points.len,
        testing_data_points.len,
    });
    // Show what the first image looks like
    std.log.debug("Here is what the first training data point looks like:", .{});
    const labeled_image_under_test = mnist_data_utils.LabeledImage{
        .label = training_data_points[0].label,
        .image = mnist_data_utils.Image{ .normalized_image = .{
            .pixels = training_data_points[0].inputs[0..(28 * 28)].*,
        } },
    };
    try mnist_print_utils.printLabeledImage(labeled_image_under_test, allocator);

    var neural_network = try neural_networks.NeuralNetwork(DigitDataPoint).init(
        &[_]u32{ 784, 100, digit_labels.len },
        neural_networks.ActivationFunction{
            // .relu = .{},
            // .leaky_relu = .{},
            // .elu = .{},
            .sigmoid = .{},
        },
        neural_networks.ActivationFunction{
            .soft_max = .{},
            // .sigmoid = .{},
        },
        neural_networks.CostFunction{
            .squared_error = .{},
            // .cross_entropy = .{},
        },
        allocator,
    );
    defer neural_network.deinit(allocator);

    var current_epoch_index: usize = 0;
    while (true
    // current_epoch_index < TRAINING_EPOCHS
    ) : (current_epoch_index += 1) {
        // We assume the data is already shuffled so we skip shuffling on the first
        // epoch. Using a pre-shuffled dataset also gives us nice reproducible results
        // during the first epoch when trying to debug things.
        var shuffled_training_data_points = training_data_points;
        if (current_epoch_index > 0) {
            // Shuffle the data after each epoch
            shuffled_training_data_points = try shuffle(random_instance, training_data_points, .{ .allocator = allocator });
        }
        defer {
            if (current_epoch_index > 0) {
                allocator.free(shuffled_training_data_points);
            }
        }

        // Split the training data into mini batches so way we can get through learning
        // iterations faster. It does make the learning progress a bit noisy because the
        // cost landscape is a bit different for each batch but it's fast and apparently
        // the noise can even be beneficial in various ways, like for escaping settle
        // points in the cost gradient (ridgelines between two valleys).
        //
        // Instead of "gradient descent" with the full training set, using mini batches
        // is called "stochastic gradient descent".
        var batch_index: u32 = 0;
        while (batch_index < shuffled_training_data_points.len / BATCH_SIZE) : (batch_index += 1) {
            const batch_start_index = batch_index * BATCH_SIZE;
            const batch_end_index = batch_start_index + BATCH_SIZE;
            const training_batch = shuffled_training_data_points[batch_start_index..batch_end_index];

            try neural_network.learn(
                training_batch,
                LEARN_RATE,
                MOMENTUM,
                allocator,
            );

            if (batch_index % 5 == 0) {
                const current_timestamp_seconds = std.time.timestamp();
                const runtime_duration_seconds = current_timestamp_seconds - start_timestamp_seconds;
                const duration_string = try time_utils.formatDuration(
                    runtime_duration_seconds * time_utils.ONE_SECOND_MS,
                    allocator,
                );
                defer allocator.free(duration_string);

                const cost = try neural_network.cost_many(testing_data_points[0..NUM_OF_IMAGES_TO_QUICK_TEST_ON], allocator);
                const accuracy = try neural_network.getAccuracyAgainstTestingDataPoints(
                    testing_data_points[0..NUM_OF_IMAGES_TO_QUICK_TEST_ON],
                    allocator,
                );
                std.log.debug("epoch {d: <3} batch {d: <3} {s: >12} -> cost {d}, accuracy with {d} test points {d}", .{
                    current_epoch_index,
                    batch_index,
                    duration_string,
                    cost,
                    NUM_OF_IMAGES_TO_QUICK_TEST_ON,
                    accuracy,
                });
            }

            // TODO: remove
            break;
        }

        // Do a full cost break-down with all of the test points after each epoch
        const cost = try neural_network.cost_many(testing_data_points, allocator);
        const accuracy = try neural_network.getAccuracyAgainstTestingDataPoints(
            testing_data_points,
            allocator,
        );
        std.log.debug("epoch end {d: <3} {s: >18} -> cost {d}, accuracy with *ALL* test points {d}", .{
            current_epoch_index,
            "",
            cost,
            accuracy,
        });

        // TODO: remove
        break;
    }
}
