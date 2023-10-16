const std = @import("std");
const shuffle = @import("zshuffle").shuffle;
const neural_networks = @import("neural_networks/neural_networks.zig");
const LayerOutputData = @import("neural_networks/layer.zig").LayerOutputData;
const graphNeuralNetwork = @import("graph_visualization/graph_neural_network.zig").graphNeuralNetwork;
const time_utils = @import("utils/time_utils.zig");

const TRAINING_EPOCHS = 2000;
// TODO: Set this back to 10 after figuring out cost problems
const BATCH_SIZE: u32 = 1;
const LEARN_RATE: f64 = 0.1;
// Since this problem space doesn't have much curvature, momentum tends to hurt us more
// with higher values.
const MOMENTUM = 0.3;

// This is a small testing dataset (to make sure our code is working) with only 2
// arbitrary features (x and y) where the labeled data points (fish and goat) occupy
// distinct parts of the graph. The boundary between the two labels is not a
// straight line (linear relationship) so we need the power of a neural network to
// learn the non-linear relationship.
//
// Since we only have two inputs with this dataset, we could graph the data points
// based on the inputs as (x, y) and colored based on the label. Then we can run the
// neural network over every pixel in the graph to visualize the boundary that the
// networks weights and biases is making. See https://youtu.be/hfMk-kjRv4c?t=311 for
// reference.
const animal_labels = [_][]const u8{
    "fish",
    "goat",
    // TODO: Remove this third label (just testing what goes wrong)
    "TODO: Remove me",
    //"TODO: Remove me2",
};
const AnimalDataPoint = neural_networks.DataPoint([]const u8, &animal_labels);
// Graph of animal data points:
// https://www.desmos.com/calculator/tkfacez5wt
var animal_training_data_points = [_]AnimalDataPoint{
    AnimalDataPoint.init(&[_]f64{ 0.924, 0.166 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.04, 0.085 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.352, 0.373 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.662, 0.737 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.724, 0.049 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.123, 0.517 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.2245, 0.661 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.466, 0.504 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.375, 0.316 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.039, 0.3475 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.28, 0.363 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.342, 0.142 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.517, 0.416 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.108, 0.403 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.728, 0.208 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.214, 0.238 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.865, 0.525 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.645, 0.363 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.436, 0.182 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.41, 0.085 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.146, 0.404 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.09, 0.457 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.663, 0.61 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.445, 0.384 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.588, 0.409 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.49, 0.075 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.679, 0.4455 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.145, 0.159 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.086, 0.155 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.192, 0.348 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.766, 0.62 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.132, 0.28 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.04, 0.403 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.588, 0.353 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.59, 0.452 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.364, 0.042 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.863, 0.068 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.806, 0.274 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.571, 0.49 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.762, 0.39 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.245, 0.388 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.097, 0.05 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.112, 0.339 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.538, 0.51 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.73, 0.507 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.472, 0.604 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.368, 0.506 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.768, 0.14 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.49, 0.75 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.21, 0.573 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.881, 0.382 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.331, 0.263 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.6515, 0.213 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.155, 0.721 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.89, 0.746 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.613, 0.265 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.442, 0.449 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.064, 0.554 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.314, 0.771 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.673, 0.135 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.535, 0.216 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.047, 0.267 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.502, 0.324 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.096, 0.827 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.586, 0.653 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.214, 0.049 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.683, 0.88 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.246, 0.315 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.264, 0.512 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.39, 0.414 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.323, 0.573 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.593, 0.307 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.314, 0.692 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.817, 0.456 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.596, 0.054 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.192, 0.403 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.195, 0.469 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.587, 0.138 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.315, 0.338 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.917, 0.267 }, "goat"),
};
const animal_testing_data_points = [_]AnimalDataPoint{
    AnimalDataPoint.init(&[_]f64{ 0.23, 0.14 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.087, 0.236 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.507, 0.142 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.503, 0.403 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.67, 0.076 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.074, 0.34 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.41, 0.257 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.278, 0.273 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.5065, 0.373 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.5065, 0.272 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.551, 0.173 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.636, 0.128 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.2, 0.33 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.409, 0.345 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.358, 0.284 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.098, 0.102 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.442, 0.058 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.368, 0.167 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.459, 0.3235 }, "fish"),
    AnimalDataPoint.init(&[_]f64{ 0.37, 0.674 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.32, 0.43 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.066, 0.628 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.635, 0.527 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.704, 0.305 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.82, 0.137 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.862, 0.305 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.709, 0.679 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.18, 0.527 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.072, 0.405 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.218, 0.408 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.303, 0.357 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.425, 0.443 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.554, 0.505 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.659, 0.251 }, "goat"),
    AnimalDataPoint.init(&[_]f64{ 0.597, 0.386 }, "goat"),
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        switch (gpa.deinit()) {
            .ok => {},
            .leak => std.log.err("GPA allocator: Memory leak detected", .{}),
        }
    }

    // TODO: We can use `@intCast(u64, std.time.timestamp())` to get a seed that changes
    const seed = 123;
    var prng = std.rand.DefaultPrng.init(seed);
    const random_instance = prng.random();

    const start_timestamp_seconds = std.time.timestamp();

    var neural_network = try neural_networks.NeuralNetwork(AnimalDataPoint).init(
        &[_]u32{
            2,
            // TODO: set this back to 10, 10
            //10, 10,
            animal_labels.len,
        },
        neural_networks.ActivationFunction{
            // .relu = .{},
            .leaky_relu = .{},
            //.sigmoid = .{},
        },
        neural_networks.ActivationFunction{
            .soft_max = .{},
            //.sigmoid = .{},
        },
        neural_networks.CostFunction{
            .squared_error = .{},
            //.cross_entropy = .{},
        },
        allocator,
    );
    defer neural_network.deinit(allocator);

    const test_layer_index = neural_network.layers.len - 1;
    std.log.debug("initial layer weights {d:.3}", .{neural_network.layers[test_layer_index].weights});
    std.log.debug("initial layer biases {d:.3}", .{neural_network.layers[test_layer_index].biases});

    var current_epoch_iteration_count: usize = 0;
    while (true
    //current_epoch_iteration_count < TRAINING_EPOCHS
    ) : (current_epoch_iteration_count += 1) {
        // Shuffle the data after each epoch
        shuffle(random_instance, &animal_training_data_points, .{});

        // Split the training data into mini batches so way we can get through learning
        // iterations faster. It does make the learning progress a bit noisy because the
        // cost landscape is a bit different for each batch but it's fast and apparently
        // the noise can even be beneficial in various ways, like for escaping settle
        // points in the cost gradient (ridgelines between two valleys).
        //
        // Instead of "gradient descent" with the full training set, using mini batches
        // is called "stochastic gradient descent".
        var batch_index: u32 = 0;
        while (batch_index < animal_training_data_points.len / BATCH_SIZE) : (batch_index += 1) {
            // TODO: Shuffle the data after each epoch
            const batch_start_index = batch_index * BATCH_SIZE;
            const batch_end_index = batch_start_index + BATCH_SIZE;
            const training_batch = animal_training_data_points[batch_start_index..batch_end_index];

            try neural_network.learn(
                training_batch,
                LEARN_RATE,
                MOMENTUM,
                allocator,
            );

            // if (current_epoch_iteration_count % 10 == 0 and
            //     current_epoch_iteration_count != 0 and
            //     batch_index == 0)
            // {
            const current_timestamp_seconds = std.time.timestamp();
            const runtime_duration_seconds = current_timestamp_seconds - start_timestamp_seconds;
            // TODO: Looks like we could use `std.fmt.fmtDuration(ns: u64)` instead of our custom thing
            const duration_string = try time_utils.formatDuration(
                runtime_duration_seconds * time_utils.ONE_SECOND_MS,
                allocator,
            );
            defer allocator.free(duration_string);

            const cost = try neural_network.cost_many(&animal_testing_data_points, allocator);
            const accuracy = try neural_network.getAccuracyAgainstTestingDataPoints(
                &animal_testing_data_points,
                allocator,
            );
            std.log.debug("epoch {d: <5} batch {d: <2} {s: >12} -> cost {d}, acccuracy with testing points {d}", .{
                current_epoch_iteration_count,
                batch_index,
                duration_string,
                cost,
                accuracy,
            });
            // }
        }

        // Graph how the neural network is learning over time.
        if (current_epoch_iteration_count % 10000 == 0 and current_epoch_iteration_count != 0) {
            try graphNeuralNetwork(
                AnimalDataPoint,
                &neural_network,
                &animal_training_data_points,
                &animal_testing_data_points,
                allocator,
            );
        }
    }

    // Graph how the neural network looks at the end of training.
    try graphNeuralNetwork(
        AnimalDataPoint,
        &neural_network,
        &animal_training_data_points,
        &animal_testing_data_points,
        allocator,
    );
}
