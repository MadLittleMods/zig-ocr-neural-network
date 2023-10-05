const std = @import("std");
const ActivationFunction = @import("activation_functions.zig").ActivationFunction;

pub fn Layer(comptime num_nodes_in: usize, comptime num_nodes_in_this_layer: usize) type {
    return struct {
        const Self = @This();
        num_nodes_in: usize,
        num_nodes_in_this_layer: usize,
        // Weights for each incoming connection. Each node in this layer has a weighted
        // connection to each node in the previous layer.
        //
        // The weights are stored in row-major order where each row is the incoming
        // connection weights for a single node in this layer.
        weights: [num_nodes_in_this_layer * num_nodes_in]f64,
        // Bias for each node in the layer
        biases: [num_nodes_in_this_layer]f64,
        costGradientWeights: [num_nodes_in_this_layer * num_nodes_in]f64,
        costGradientBiases: [num_nodes_in_this_layer]f64,

        activation_function: ActivationFunction,

        /// Create the layer
        pub fn init(activation_function: ActivationFunction, allocator: std.mem.Allocator) !Self {
            _ = allocator;
            var prng = std.rand.DefaultPrng.init(123);

            // Initialize the weights and biases to random values
            var weights = [num_nodes_in_this_layer * num_nodes_in]f64;
            for (weights) |*weight| {
                weight.* = prng.random().floatNorm(f64) * 0.2;
            }
            var biases: [num_nodes_in_this_layer]f64 = .{};

            return Self{
                .num_nodes_in = num_nodes_in,
                .num_nodes_in_this_layer = num_nodes_in_this_layer,
                .weights = weights,
                .biases = biases,
                .activation_function = activation_function,
            };
        }

        // pub fn nodeCount(self: *Self) usize {
        //     return self.num_nodes_in_this_layer;
        // }

        /// Calculate the output of the layer.
        ///
        /// The output of a node in this layer is the bias plus the weighted sum of all
        /// of the incoming connections after they have been passed through the
        /// activation function.
        pub fn calculateOutputs(self: *Self, inputs: [num_nodes_in]f64, allocator: std.mem.Allocator) [num_nodes_in_this_layer]f64 {
            var outputs = try allocator.alloc(f64, num_nodes_in_this_layer);
            // Calculate the weighted inputs for each node in this layer
            for (0..self.num_nodes_in_this_layer) |node_index| {
                // Calculate the weighted input for this node
                var weighted_input_sum: f64 = 0.0;
                for (self.num_nodes_in) |node_in_index| {
                    weighted_input_sum += inputs[node_in_index] * self.weights[(node_index * num_nodes_in) + node_in_index];
                }
                outputs[node_index] = self.activation_function.activate(self.biases[node_index] + weighted_input_sum);
            }
            return outputs;
        }

        pub const updateGradients(node_values: [num_nodes_in_this_layer]f64) void {
            for (0..self.num_nodes_in_this_layer) |node_index| {
                for (self.num_nodes_in) |node_in_index| {
                    // Evaluate the partial derivative: cost / weight of the current connection
                    const derivative_cost_with_regards_to_weight = inputs[node_in_index] * node_values[node_index];
                    // The costGradientW array stores these partial derivatives for each weight.
                    // Note: The derivative is being added to the array here because ultimately we want
                    // to calculuate the average gradient across all the data in the training batch
                    self.costGradientWeights[(node_index * num_nodes_in) + node_in_index] += derivative_cost_with_regards_to_weight;
                }

                // Evaluate the partial derivative: cost / bias of the current node
                // TODO: Why `1 *`?
                const derivative_cost_with_regards_to_bias = 1 * node_values[node_index];
                self.costGradientBiases[node_index] += derivative_cost_with_regards_to_bias;
            }
        }

        /// Update the weights and biases based on the cost gradients (gradient descent)
        pub fn applyGradients(self: *Self, learnRate: f64) void {
            for (0..self.num_nodes_in_this_layer) |node_index| {
                self.biases[node_index] -= learnRate * self.costGradientBiases[node_index];
                for (self.num_nodes_in) |node_in_index| {
                    self.weights[(node_index * num_nodes_in) + node_in_index] -= learnRate * self.costGradientWeights[(node_index * num_nodes_in) + node_in_index];
                }
            }
        }
    };
}
