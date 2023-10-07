const std = @import("std");
const ActivationFunction = @import("activation_functions.zig").ActivationFunction;

pub const Layer = struct {
    const Self = @This();
    num_nodes_in: usize,
    num_nodes_in_this_layer: usize,
    // Weights for each incoming connection. Each node in this layer has a weighted
    // connection to each node in the previous layer (num_nodes_in * num_nodes_in_this_layer).
    //
    // The weights are stored in row-major order where each row is the incoming
    // connection weights for a single node in this layer.
    weights: []f64,
    // Bias for each node in the layer (num_nodes_in_this_layer)
    biases: []f64,
    costGradientWeights: []f64,
    costGradientBiases: []f64,

    activation_function: ActivationFunction,

    /// Create the layer
    pub fn init(
        num_nodes_in: usize,
        num_nodes_in_this_layer: usize,
        activation_function: ActivationFunction,
        allocator: std.mem.Allocator,
    ) !Self {
        var prng = std.rand.DefaultPrng.init(123);

        // Initialize the weights
        var weights: []f64 = try allocator.alloc(f64, num_nodes_in * num_nodes_in_this_layer);
        // Initialize the weights of the network to random values
        for (weights) |*weight| {
            // Get a random value between -1 and +1
            const random_value = prng.random().floatNorm(f64);
            // Now to choose a good weight initialization scheme. The "best" heuristic
            // often depends on the specific activiation function being used. We want to
            // avoid the vanishing/exploding gradient problem.
            //
            // Xavier initialization takes a set of random values sampled uniformly from
            // a range proportional to the size of the number of nodes in the previous
            // layer. Specifically multiplying the random value by `1 / sqrt(num_nodes_in)`.
            //
            // "He initialization" is similar to Xavier initialization, but multiplies
            // the random value by `1 / sqrt(2 / num_nodes_in)`. This modification is
            // suggested when using the ReLU activation function to achieve a "properly
            // scaled uniform distribution for initialization".
            weight.* = random_value / @sqrt(2 / num_nodes_in);

            // Note: there are many different ways of trying to chose a good range for
            // the random weights, and these depend on facors such as the activation
            // function being used. and howthe inputs to the network have been
            // scaled/normalized (ideally our input data should be scaled to the range
            // [0, 1]).
            //
            // For example, when using the sigmoid activation function, we don't want
            // the weighted inputs to be too large, as otherwise the slope of the
            // function will be very close to zero, resulting in the gradient descent
            // algorithm learning very slowly (or not at all).
        }
        var biases: []f64 = try allocator.alloc(f64, num_nodes_in_this_layer);
        for (biases) |*bias| {
            // Specifically for the ReLU activation function, the *Deep Learning* (Ian
            // Goodfellow) book suggests:
            // > it can be a good practice to set all elements of [the bias] to a small,
            // > positive value, such as 0.1. This makes it very likely that the rectified
            // > linear units will be initially active for most inputs in the training set
            // > and allow the derivatives to pass through.
            bias.* = 0.1;
        }

        var costGradientWeights: []f64 = try allocator.alloc(f64, num_nodes_in * num_nodes_in_this_layer);
        var costGradientBiases: []f64 = try allocator.alloc(f64, num_nodes_in_this_layer);

        return Self{
            .num_nodes_in = num_nodes_in,
            .num_nodes_in_this_layer = num_nodes_in_this_layer,
            .weights = weights,
            .biases = biases,
            .costGradientWeights = costGradientWeights,
            .costGradientBiases = costGradientBiases,
            .activation_function = activation_function,
        };
    }

    // pub fn nodeCount(self: *Self) usize {
    //     return self.num_nodes_in_this_layer;
    // }

    /// Helper to access the weight for a specific connection since
    /// the weights are stored in a flat array.
    fn getWeight(self: *Self, node_index: usize, node_in_index: usize) f64 {
        return self.weights[(node_index * self.num_nodes_in) + node_in_index];
    }

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
                weighted_input_sum += inputs[node_in_index] * self.getWeight(node_index, node_in_index);
            }
            outputs[node_index] = self.activation_function.activate(self.biases[node_index] + weighted_input_sum);
        }
        return outputs;
    }

    pub fn updateCostGradients(self: *Self, node_values: []f64) !void {
        if (node_values.len != self.num_nodes_in_this_layer) {
            std.log.err("updateGradients() was called with {d} node_values but we expect it to match num_nodes_in_this_layer={d}", .{
                node_values.len,
                self.num_nodes_in_this_layer,
            });

            return error.NodeCountMismatch;
        }

        for (0..self.num_nodes_in_this_layer) |node_index| {
            for (self.num_nodes_in) |node_in_index| {
                // Evaluate the partial derivative of cost with respect to the weight of the current connection
                const derivative_cost_with_regards_to_weight = inputs[node_in_index] * node_values[node_index];
                // The costGradientW array stores these partial derivatives for each weight.
                // Note: The derivative is being added to the array here because ultimately we want
                // to calculuate the average gradient across all the data in the training batch
                self.costGradientWeights[(node_index * num_nodes_in) + node_in_index] += derivative_cost_with_regards_to_weight;
            }

            // Evaluate the partial derivative of cost with respect to bias of the current node.
            // TODO: Why `1 *`?
            const derivative_cost_with_regards_to_bias = 1 * node_values[node_index];
            self.costGradientBiases[node_index] += derivative_cost_with_regards_to_bias;
        }
    }

    /// Update the weights and biases based on the cost gradients (gradient descent).
    /// Also resets the gradients back to zero.
    pub fn applyCostGradients(self: *Self, learnRate: f64) void {
        for (self.weights, 0..) |*weight, weight_index| {
            weight.* -= learnRate * self.costGradientWeights[weight_index];
            self.costGradientWeights[weight_index] = 0;
        }

        for (self.biases, 0..) |*bias, bias_index| {
            bias.* -= learnRate * self.costGradientBiases[bias_index];
            self.costGradientBiases[bias_index] = 0;
        }
    }
};
