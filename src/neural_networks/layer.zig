const std = @import("std");
const ActivationFunction = @import("activation_functions.zig").ActivationFunction;
const CostFunction = @import("cost_functions.zig").CostFunction;

/// Used to keep track of what a layer took in as input and what we outputted last time we
/// calculated the output of this layer. This is used for backpropagation.
pub const LayerOutputData = struct {
    /// The inputs to the layer that produce the following outputs.
    /// Size: num_input_nodes
    inputs: []const f64,
    /// The weighted input sum for each node in the layer. This is the sum of all the
    /// incoming connections to the node after they have been multiplied by their
    /// respective weights (plus a bias).
    /// Size: num_output_nodes
    weighted_input_sums: []f64,
    /// The output of the layer after passing the weighted input sums through the
    /// activation function.
    /// Size: num_output_nodes
    outputs: []f64,
};

pub const Layer = struct {
    const Self = @This();
    num_input_nodes: usize,
    num_output_nodes: usize,
    // Weights for each incoming connection. Each node in this layer has a weighted
    // connection to each node in the previous layer (num_input_nodes * num_output_nodes).
    //
    // The weights are stored in row-major order where each row is the incoming
    // connection weights for a single node in this layer.
    // Size: num_output_nodes * num_input_nodes
    weights: []f64,
    // Bias for each node in the layer (num_output_nodes)
    // Size: num_output_nodes
    biases: []f64,
    // Store the cost gradients for each weight and bias. These are used to update
    // the weights and biases after each training batch.
    //
    // The partial derivative of the cost function with respect to the weight of the
    // current connection.
    //
    // Size: num_output_nodes * num_input_nodes
    cost_gradient_weights: []f64,
    // The partial derivative of the cost function with respect to the bias of the
    // current node.
    //
    // Size: num_output_nodes
    cost_gradient_biases: []f64,

    // Used for adding momentum to gradient descent. Stores the change in weight/bias
    // from the previous learning iteration.
    // Size: num_output_nodes * num_input_nodes
    weight_velocities: []f64,
    // Size: num_output_nodes
    bias_velocities: []f64,

    activation_function: ActivationFunction,
    cost_function: ?CostFunction = null,

    // TODO: Maybe we call this `layer_forward_propagation_data` instead?
    layer_output_data: LayerOutputData,

    /// Create the layer
    pub fn init(
        num_input_nodes: usize,
        num_output_nodes: usize,
        activation_function: ActivationFunction,
        allocator: std.mem.Allocator,
        options: struct {
            cost_function: ?CostFunction = null,
        },
    ) !Self {
        // Initialize the weights
        var weights: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
        var biases: []f64 = try allocator.alloc(f64, num_output_nodes);
        Layer.initializeWeightsAndBiases(
            weights,
            biases,
            num_input_nodes,
            activation_function,
        );

        var cost_gradient_weights: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
        @memset(cost_gradient_weights, 0);
        var cost_gradient_biases: []f64 = try allocator.alloc(f64, num_output_nodes);
        @memset(cost_gradient_biases, 0);

        var weight_velocities: []f64 = try allocator.alloc(f64, num_input_nodes * num_output_nodes);
        @memset(weight_velocities, 0);
        var bias_velocities: []f64 = try allocator.alloc(f64, num_output_nodes);
        @memset(bias_velocities, 0);

        var weighted_input_sums = try allocator.alloc(f64, num_output_nodes);
        // We don't need to initialize @memset(weighted_input_sums, 0) because we will
        // calculate the weighted input sum for each node in the layer before we use
        // them.

        return Self{
            .num_input_nodes = num_input_nodes,
            .num_output_nodes = num_output_nodes,
            .weights = weights,
            .biases = biases,
            .cost_gradient_weights = cost_gradient_weights,
            .cost_gradient_biases = cost_gradient_biases,
            .weight_velocities = weight_velocities,
            .bias_velocities = bias_velocities,
            .activation_function = activation_function,
            .cost_function = options.cost_function,
            .layer_output_data = .{
                .inputs = undefined,
                .weighted_input_sums = weighted_input_sums,
                .outputs = undefined,
            },
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        allocator.free(self.biases);
        allocator.free(self.cost_gradient_weights);
        allocator.free(self.cost_gradient_biases);
        allocator.free(self.weight_velocities);
        allocator.free(self.bias_velocities);
        allocator.free(self.layer_output_data.weighted_input_sums);
    }

    fn initializeWeightsAndBiases(
        weights: []f64,
        biases: []f64,
        num_input_nodes: usize,
        activation_function: ActivationFunction,
    ) void {
        // TODO: We can use `@intCast(u64, std.time.timestamp())` to get a seed that changes
        const seed = 123;
        var prng = std.rand.DefaultPrng.init(seed);
        const random_instance = prng.random();

        // Initialize the weights of the network to random values
        for (weights) |*weight| {
            // Get a random value with a range `stddev = 1` centered around `mean = 0`.
            // When using a normal distribution like this, the odds are most likely that
            // your number will fall in the [-3, +3] range.
            //
            // > To use different parameters, use: floatNorm(...) * desiredStddev + desiredMean.
            const normal_random_value = random_instance.floatNorm(f64);
            // Now to choose a good weight initialization scheme. The "best" heuristic
            // often depends on the specific activiation function being used. We want to
            // avoid the vanishing/exploding gradient problem.
            //
            // Xavier initialization takes a set of random values sampled uniformly from
            // a range proportional to the size of the number of nodes in the previous
            // layer (fan-in). Specifically multiplying the normal random value by
            // `stddev = sqrt(1 / fan_in)`.
            //
            // "He initialization" is similar to Xavier initialization, but multiplies
            // the normal random value by `stddev = sqrt(2 / fan_in)`. This modification
            // is suggested when using the ReLU activation function to achieve a
            // "properly scaled uniform distribution for initialization".
            //
            // References:
            //  - https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
            //  - https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
            const desired_mean = 0;
            switch (activation_function) {
                ActivationFunction.relu => {
                    // He initialization
                    const desired_standard_deviation = @sqrt(2.0 / @as(f64, @floatFromInt(num_input_nodes)));
                    weight.* = normal_random_value * desired_standard_deviation + desired_mean;
                },
                else => {
                    // Xavier initialization
                    const desired_standard_deviation = @sqrt(1.0 / @as(f64, @floatFromInt(num_input_nodes)));
                    weight.* = normal_random_value * desired_standard_deviation + desired_mean;
                },
            }

            // Note: there are many different ways of trying to chose a good range for
            // the random weights, and these depend on facors such as the activation
            // function being used. and how the inputs to the network have been
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
    pub fn getWeight(self: *Self, node_index: usize, node_in_index: usize) f64 {
        const weight_index = self.getFlatWeightIndex(node_index, node_in_index);
        return self.weights[weight_index];
    }

    pub fn getFlatWeightIndex(self: *Self, node_index: usize, node_in_index: usize) usize {
        return (node_index * self.num_input_nodes) + node_in_index;
    }

    /// Calculate the output of the layer (forward propagation).
    ///
    /// The output of a node in this layer is the weighted sum of all
    /// of the incoming connections after they have been passed through the
    /// activation function plus a bias value.
    pub fn calculateOutputs(self: *Self, inputs: []const f64, allocator: std.mem.Allocator) ![]f64 {
        if (inputs.len != self.num_input_nodes) {
            std.log.err("calculateOutputs() was called with {d} inputs but we expect it to match the same num_input_nodes={d}", .{
                inputs.len,
                self.num_input_nodes,
            });

            return error.ExpectedOutputCountMismatch;
        }

        self.layer_output_data.inputs = inputs;

        // Calculate the weighted inputs for each node in this layer
        for (0..self.num_output_nodes) |node_index| {
            // Calculate the weighted input for this node
            var weighted_input_sum: f64 = self.biases[node_index];
            for (0..self.num_input_nodes) |node_in_index| {
                weighted_input_sum += inputs[node_in_index] * self.getWeight(node_index, node_in_index);
            }
            self.layer_output_data.weighted_input_sums[node_index] = weighted_input_sum;
        }

        // Apply activation function
        var outputs = try allocator.alloc(f64, self.num_output_nodes);
        for (0..self.num_output_nodes) |node_index| {
            // Then calculate the activation of the node
            outputs[node_index] = self.activation_function.activate(
                self.layer_output_data.weighted_input_sums,
                node_index,
            );
        }

        self.layer_output_data.outputs = outputs;

        return outputs;
    }

    /// Calculate the "shareable_node_derivatives" for this output layer
    /// TODO: Explain "shareable_node_derivatives"
    pub fn calculateOutputLayerShareableNodeDerivatives(
        self: *Self,
        expected_outputs: []const f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        if (expected_outputs.len != self.num_output_nodes) {
            std.log.err("calculateOutputLayerShareableNodeDerivatives() was called with {d} expected_outputs but we expect it to match the same num_output_nodes={d}", .{
                expected_outputs,
                self.num_output_nodes,
            });

            return error.ExpectedOutputCountMismatch;
        }

        // The following comments are made from the perspective of layer 2 in our
        // ridicously simple neural network that has just 3 nodes connected by 2
        // weights (see dev-notes.md for more details).
        var shareable_node_derivatives: []f64 = try allocator.alloc(f64, self.num_output_nodes);

        for (0..self.num_output_nodes) |node_index| {
            // Evaluate the partial derivative of activation for the current node with respect to its weighted input
            // da_2/dz_2 = activation_function.derivative(z_2)
            const activation_derivative = self.activation_function.derivative(
                self.layer_output_data.weighted_input_sums,
                node_index,
            );
            // Evaluate the partial derivative of cost for the current node with respect to its activation
            // dc/da_2 = cost_function.derivative(a_2, expected_output)
            if (self.cost_function) |cost_function| {
                const cost_derivative = cost_function.individual_derivative(
                    self.layer_output_data.outputs[node_index],
                    expected_outputs[node_index],
                );
                shareable_node_derivatives[node_index] = activation_derivative * cost_derivative;
            } else {
                @panic(
                    \\Cannot call `calculateOutputLayerShareableNodeDerivatives(...)`
                    \\without a `cost_function` set. Make sure to set a `cost_function`
                    \\for the output layer.
                );
            }
        }

        return shareable_node_derivatives;
    }

    /// Calculate the "shareable_node_derivatives" for this hidden layer
    /// TODO: Explain "shareable_node_derivatives"
    pub fn calculateHiddenLayerShareableNodeDerivatives(
        self: *Self,
        next_layer: *Self,
        next_layer_shareable_node_derivatives: []const f64,
        allocator: std.mem.Allocator,
    ) ![]f64 {
        // The following comments are made from the perspective of layer 1 in our
        // ridicously simple neural network that has just 3 nodes connected by 2
        // weights (see dev-notes.md for more details).
        var shareable_node_derivatives: []f64 = try allocator.alloc(f64, self.num_output_nodes);
        for (0..self.num_output_nodes) |node_index| {
            var shareable_node_derivative: f64 = 0.0;
            for (next_layer_shareable_node_derivatives, 0..) |next_layer_shareable_node_derivative, next_layer_node_index| {
                // Evaluate the partial derivative of the weighted input for this layer with respect to the input of the previous<TODO>
                // dz_2/da_1 = w_2
                const derivative_weighted_input_wrt_activation = next_layer.getWeight(
                    next_layer_node_index,
                    node_index,
                );
                shareable_node_derivative += derivative_weighted_input_wrt_activation * next_layer_shareable_node_derivative;
            }
            // Evaluate the partial derivative of activation for the current node with respect to its weighted input
            // da_1/dz_1 = activation_function.derivative(z_1)
            shareable_node_derivative *= self.activation_function.derivative(
                self.layer_output_data.weighted_input_sums,
                node_index,
            );
            shareable_node_derivatives[node_index] = shareable_node_derivative;
        }
        return shareable_node_derivatives;
    }

    pub fn updateCostGradients(
        self: *Self,
        // "shareable_node_derivatives" is just the name given to a set of derivatives
        // calculated for the type of layer respectively (see
        // `calculateOutputLayerShareableNodeDerivatives(...)` and
        // `calculateHiddenLayerShareableNodeDerivatives(...)` above). Since those
        // "shareable_node_derivatives" are shared in the equations for calculating the
        // partial derivative of cost with respect to both weight and bias we can re-use
        // that work.
        // Size: num_output_nodes
        shareable_node_derivatives: []const f64,
    ) !void {
        if (shareable_node_derivatives.len != self.num_output_nodes) {
            std.log.err("updateGradients() was called with {d} shareable_node_derivatives but we expect it to match the same num_output_nodes={d}", .{
                shareable_node_derivatives.len,
                self.num_output_nodes,
            });

            return error.NodeCountMismatch;
        }

        for (0..self.num_output_nodes) |node_index| {
            for (0..self.num_input_nodes) |node_in_index| {
                // dz_2/dw_2 = a_1
                const derivative_weighted_input_wrt_weight = self.layer_output_data.inputs[node_in_index];
                // Evaluate the partial derivative of cost with respect to the weight of the current connection
                // dc/dw_2 = dz_2/dw_2 * shareable_node_derivatives[node_index]
                const derivative_cost_wrt_weight = derivative_weighted_input_wrt_weight * shareable_node_derivatives[node_index];
                // The cost_gradient_weights array stores these partial derivatives for each weight.
                // Note: The derivative is being added to the array here because ultimately we want
                // to calculuate the average gradient across all the data in the training batch
                self.cost_gradient_weights[self.getFlatWeightIndex(node_index, node_in_index)] += derivative_cost_wrt_weight;
            }

            // This is `1` because no matter how much the bias changes, the weighted input will
            // change by the same amount. z_2 = a_1 * w_2 + b_2, so dz_2/db_2 = 1.
            const derivative_weighted_input_wrt_bias = 1;

            // Evaluate the partial derivative of cost with respect to bias of the current node.
            // dc/db_2 = dz_2/db_2 * shareable_node_derivatives[node_index]
            const derivative_cost_wrt_bias = derivative_weighted_input_wrt_bias * shareable_node_derivatives[node_index];
            self.cost_gradient_biases[node_index] += derivative_cost_wrt_bias;
        }
    }

    /// Update the weights and biases based on the cost gradients (gradient descent).
    /// Also resets the gradients back to zero.
    pub fn applyCostGradients(
        self: *Self,
        learn_rate: f64,
        // The momentum to apply to gradient descent. This is a value between 0 and 1
        // and often has a value close to 1.0, such as 0.8, 0.9, or 0.99. A momentum of
        // 0.0 is the same as gradient descent without momentum.
        //
        // Momentum is used to help the gradient descent algorithm keep the learning
        // process going in the right direction between different batches. It does this
        // by adding a fraction of the previous weight change to the current weight
        // change. Essentially, if it was moving before, it will keep moving in the same
        // direction. It's most useful in situations where the cost surface has lots of
        // curvature (changes a lot) ("highly non-spherical") or when the cost surface
        // "flat or nearly flat, e.g. zero gradient. The momentum allows the search to
        // progress in the same direction as before the flat spot and helpfully cross
        // the flat region."
        // (https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/)
        //
        // > The momentum algorithm accumulates an exponentially decaying moving average
        // > of past gradients and continues to move in their direction.
        // >
        // > -- *Deep Learning* book page 296 (Ian Goodfellow)
        momentum: f64,
    ) void {
        for (self.weights, 0..) |*weight, weight_index| {
            const velocity = (learn_rate * self.cost_gradient_weights[weight_index]) +
                (momentum * self.weight_velocities[weight_index]);
            // Store the velocity for use in the next iteration
            self.weight_velocities[weight_index] = velocity;

            // Update the weight
            weight.* -= velocity;
            // Reset the gradient back to zero now that we've applied it
            self.cost_gradient_weights[weight_index] = 0;
        }

        for (self.biases, 0..) |*bias, bias_index| {
            const velocity = (learn_rate * self.cost_gradient_biases[bias_index]) +
                (momentum * self.bias_velocities[bias_index]);
            // Store the velocity for use in the next iteration
            self.bias_velocities[bias_index] = velocity;

            // Update the bias
            bias.* -= velocity;
            // Reset the gradient back to zero now that we've applied it
            self.cost_gradient_biases[bias_index] = 0;
        }
    }
};
