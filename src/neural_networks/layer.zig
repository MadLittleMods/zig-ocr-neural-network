const std = @import("std");
const ActivationFunction = @import("activation_functions.zig").ActivationFunction;

pub const Layer = struct {
    const Self = @This();
    num_input_nodes: usize,
    num_output_nodes: usize,
    // Weights for each incoming connection. Each node in this layer has a weighted
    // connection to each node in the previous layer (num_input_nodes * num_output_nodes).
    //
    // The weights are stored in row-major order where each row is the incoming
    // connection weights for a single node in this layer.
    weights: []f64,
    // Bias for each node in the layer (num_output_nodes)
    biases: []f64,
    costGradientWeights: []f64,
    costGradientBiases: []f64,

    activation_function: ActivationFunction,

    /// Create the layer
    pub fn init(
        num_input_nodes: usize,
        num_output_nodes: usize,
        activation_function: ActivationFunction,
        allocator: std.mem.Allocator,
    ) !Self {
        // Initialize the weights
        var weights: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
        var biases: []f64 = try allocator.alloc(f64, num_output_nodes);
        Layer.initializeWeightsAndBiases(weights, biases, num_input_nodes);

        var costGradientWeights: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
        var costGradientBiases: []f64 = try allocator.alloc(f64, num_output_nodes);

        return Self{
            .num_input_nodes = num_input_nodes,
            .num_output_nodes = num_output_nodes,
            .weights = weights,
            .biases = biases,
            .costGradientWeights = costGradientWeights,
            .costGradientBiases = costGradientBiases,
            .activation_function = activation_function,
        };
    }

    fn initializeWeightsAndBiases(
        weights: []f64,
        biases: []f64,
        num_input_nodes: usize,
    ) void {
        var prng = std.rand.DefaultPrng.init(123);

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
            // layer. Specifically multiplying the random value by `1 / sqrt(num_input_nodes)`.
            //
            // "He initialization" is similar to Xavier initialization, but multiplies
            // the random value by `1 / sqrt(2 / num_input_nodes)`. This modification is
            // suggested when using the ReLU activation function to achieve a "properly
            // scaled uniform distribution for initialization".
            weight.* = random_value / @sqrt(2 / num_input_nodes);

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

        for (biases) |*bias| {
            // Specifically for the ReLU activation function, the *Deep Learning* (Ian
            // Goodfellow) book suggests:
            // > it can be a good practice to set all elements of [the bias] to a small,
            // > positive value, such as 0.1. This makes it very likely that the rectified
            // > linear units will be initially active for most inputs in the training set
            // > and allow the derivatives to pass through.
            bias.* = 0.1;
        }
    }

    // pub fn nodeCount(self: *Self) usize {
    //     return self.num_output_nodes;
    // }

    /// Helper to access the weight for a specific connection since
    /// the weights are stored in a flat array.
    fn getWeight(self: *Self, node_index: usize, node_in_index: usize) f64 {
        return self.weights[(node_index * self.num_input_nodes) + node_in_index];
    }

    /// Calculate the output of the layer.
    ///
    /// The output of a node in this layer is the weighted sum of all
    /// of the incoming connections after they have been passed through the
    /// activation function plus a bias value.
    pub fn calculateOutputs(self: *Self, inputs: [num_input_nodes]f64, allocator: std.mem.Allocator) [num_output_nodes]f64 {
        var outputs = try allocator.alloc(f64, num_output_nodes);
        // Calculate the weighted inputs for each node in this layer
        for (0..self.num_output_nodes) |node_index| {
            // Calculate the weighted input for this node
            var weighted_input_sum: f64 = 0.0;
            for (self.num_input_nodes) |node_in_index| {
                weighted_input_sum += inputs[node_in_index] * self.getWeight(node_index, node_in_index);
            }
            outputs[node_index] = self.activation_function.activate(weighted_input_sum + self.biases[node_index]);
        }
        return outputs;
    }

    pub fn updateCostGradients(
        self: *Self,
        // "Node values" is just the name given to the result of (da_2/dz_2 * dc/da_2)
        // since we can re-use that work when calculating the partial derivative of cost
        // with respect to both weight and bias.
        node_values: []f64,
    ) !void {
        if (node_values.len != self.num_output_nodes) {
            std.log.err("updateGradients() was called with {d} node_values but we expect it to match the same num_output_nodes={d}", .{
                node_values.len,
                self.num_output_nodes,
            });

            return error.NodeCountMismatch;
        }

        for (0..self.num_output_nodes) |node_index| {
            for (self.num_input_nodes) |node_in_index| {
                // Evaluate the partial derivative of cost with respect to the weight of the current connection
                const derivative_cost_with_regards_to_weight = inputs[node_in_index] * node_values[node_index];
                // The costGradientWeights array stores these partial derivatives for each weight.
                // Note: The derivative is being added to the array here because ultimately we want
                // to calculuate the average gradient across all the data in the training batch
                self.costGradientWeights[(node_index * num_input_nodes) + node_in_index] += derivative_cost_with_regards_to_weight;
            }

            // This is `1` because no matter how much the bias changes, the weighted input will
            // change by the same amount. z_2 = a_1 * w_2 + b_2, so dz_2/db_2 = 1.
            const derivative_weighted_input_with_respect_to_bias = 1;

            // Evaluate the partial derivative of cost with respect to bias of the current node.
            const derivative_cost_with_regards_to_bias = derivative_weighted_inputs_with_respect_to_biases * node_values[node_index];
            self.costGradientBiases[node_index] += derivative_cost_with_regards_to_bias;
        }
    }

    /// Update the weights and biases based on the cost gradients (gradient descent).
    /// Also resets the gradients back to zero.
    pub fn applyCostGradients(self: *Self, learnRate: f64) void {
        for (self.weights, 0..) |*weight, weight_index| {
            weight.* -= learnRate * self.costGradientWeights[weight_index];
            // Reset the gradient back to zero now that we've applied it
            self.costGradientWeights[weight_index] = 0;
        }

        for (self.biases, 0..) |*bias, bias_index| {
            bias.* -= learnRate * self.costGradientBiases[bias_index];
            // Reset the gradient back to zero now that we've applied it
            self.costGradientBiases[bias_index] = 0;
        }
    }
};
