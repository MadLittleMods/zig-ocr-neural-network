const std = @import("std");
const neural_networks = @import("neural_networks/neural_networks.zig");
const LayerOutputData = @import("neural_networks/layer.zig").LayerOutputData;

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
    const animal_labels = [_][]const u8{ "fish", "goat" };
    const AnimalDataPoint = neural_networks.DataPoint([]const u8, &animal_labels);
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

    var neural_network = try neural_networks.NeuralNetwork(AnimalDataPoint).init(
        &[_]u32{ 2, 3, 3, animal_labels.len },
        neural_networks.ActivationFunction{ .relu = .{} },
        neural_networks.ActivationFunction{ .soft_max = .{} },
        neural_networks.CostFunction{ .mean_squared_error = .{} },
        allocator,
    );
    defer neural_network.deinit(allocator);

    var current_epoch_iteration_count: usize = 0;
    while (current_epoch_iteration_count < TRAINING_EPOCHS) : (current_epoch_iteration_count += 1) {
        try neural_network.learn(
            &animal_training_data_points,
            LEARN_RATE,
            allocator,
        );

        const cost = try neural_network.cost(&animal_training_data_points, allocator);
        const accuracy = try neural_network.getAccuracyAgainstTestingDataPoints(
            &animal_testing_data_points,
            allocator,
        );
        std.log.debug("epoch {d} -> cost {d}, acccuracy {d}", .{
            current_epoch_iteration_count,
            cost,
            accuracy,
        });
    }
}
