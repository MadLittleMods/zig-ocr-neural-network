const std = @import("std");
const Layer = @import("layer.zig").Layer;
const LayerOutputData = @import("layer.zig").LayerOutputData;

pub const ActivationFunction = @import("activation_functions.zig").ActivationFunction;
pub const CostFunction = @import("cost_functions.zig").CostFunction;
pub const DataPoint = @import("data_point.zig").DataPoint;

pub fn NeuralNetwork(comptime DataPointType: type) type {
    return struct {
        const Self = @This();

        cost_function: CostFunction,
        layers: []Layer,

        pub fn init(
            layer_sizes: []const u32,
            activation_function: ActivationFunction,
            output_layer_activation_function: ActivationFunction,
            cost_function: CostFunction,
            allocator: std.mem.Allocator,
        ) !Self {
            var layers = try allocator.alloc(Layer, layer_sizes.len - 1);
            const output_layer_index = layers.len - 1;
            for (layers, 0..) |*layer, layer_index| {
                layer.* = try Layer.init(
                    layer_sizes[layer_index],
                    layer_sizes[layer_index + 1],
                    if (layer_index == output_layer_index) output_layer_activation_function else activation_function,
                    allocator,
                    .{
                        // We just set the cost function for all layers even though it's
                        // only needed for the last output layer.
                        .cost_function = cost_function,
                    },
                );
            }

            if (layers[output_layer_index].cost_function) |_| {
                // no-op
            } else {
                std.log.err("NeuralNetwork.init(...): The cost function for the output layer (the last layer in the neural network) must be specified", .{});
                return error.CostFunctionNotSpecifiedForOutputLayer;
            }

            return Self{
                .cost_function = cost_function,
                .layers = layers,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (self.layers) |*layer| {
                layer.deinit(allocator);
            }

            allocator.free(self.layers);
        }

        // Run the input values through the network to calculate the output values
        pub fn calculateOutputs(
            self: *Self,
            inputs: []const f64,
            allocator: std.mem.Allocator,
        ) ![]const f64 {
            var inputs_to_next_layer = inputs;
            for (self.layers) |*layer| {
                inputs_to_next_layer = try layer.calculateOutputs(inputs_to_next_layer, allocator);
            }

            return inputs_to_next_layer;
        }

        pub fn freeAfterCalculateOutputs(self: *Self, allocator: std.mem.Allocator) void {
            // We only need to free the inputs for the hidden layers because the output
            // of one layer is the input to the next layer. We do need to clean up the
            // output of the output layer though (see below).
            for (self.layers, 0..) |*layer, layer_index| {
                // Avoid freeing the initial `inputs` that someone passed in to this function.
                if (layer_index > 0) {
                    allocator.free(layer.layer_output_data.inputs);
                }

                // We don't need to free the `weighted_input_sums` because it's shared
                // across runs and cleaned up in `layer.deinit()`.
                // allocator.free(layer.layer_output_data.weighted_input_sums);
            }

            // Clean up the output of the output layer
            const output_layer_index = self.layers.len - 1;
            const output_layer = self.layers[output_layer_index];
            allocator.free(output_layer.layer_output_data.outputs);
        }

        // Run the input values through the network and calculate which output node has
        // the highest value
        pub fn classify(
            self: *Self,
            inputs: []const f64,
            allocator: std.mem.Allocator,
        ) !DataPointType.LabelType {
            var outputs = try self.calculateOutputs(inputs, allocator);
            defer self.freeAfterCalculateOutputs(allocator);

            var max_output = outputs[0];
            var max_output_index: usize = 0;
            for (outputs, 0..) |output, index| {
                if (output > max_output) {
                    max_output = output;
                    max_output_index = index;
                }
            }

            return DataPointType.oneHotIndexToLabel(max_output_index);
        }

        pub fn getAccuracyAgainstTestingDataPoints(
            self: *Self,
            testing_data_points: []const DataPointType,
            allocator: std.mem.Allocator,
        ) !f64 {
            var correct_count: f64 = 0;
            for (testing_data_points) |testing_data_point| {
                const result = try self.classify(testing_data_point.inputs, allocator);
                if (DataPointType.checkLabelsEqual(result, testing_data_point.label)) {
                    correct_count += 1;
                }
            }

            return correct_count / @as(f64, @floatFromInt(testing_data_points.len));
        }

        /// Calculate the total cost of the network for a single data point
        pub fn cost_individual(
            self: *Self,
            data_point: DataPointType,
            allocator: std.mem.Allocator,
        ) !f64 {
            var outputs = try self.calculateOutputs(data_point.inputs, allocator);
            defer self.freeAfterCalculateOutputs(allocator);

            return self.cost_function.cost(outputs, &data_point.expected_outputs);
        }

        /// Calculate the total cost of the network for a batch of data points
        pub fn cost_many(
            self: *Self,
            data_points: []const DataPointType,
            allocator: std.mem.Allocator,
        ) !f64 {
            var total_cost: f64 = 0.0;
            for (data_points) |data_point| {
                const cost_of_data_point = try self.cost_individual(data_point, allocator);
                // std.log.debug("cost_of_data_point: {d}", .{cost_of_data_point});
                total_cost += cost_of_data_point;
            }
            return total_cost;
        }

        /// Calculate the average cost of the network for a batch of data points
        pub fn average_cost(
            self: *Self,
            data_points: []const DataPointType,
            allocator: std.mem.Allocator,
        ) !f64 {
            const total_cost = try self.many_cost(data_points, allocator);
            return total_cost / @as(f64, @floatFromInt(data_points.len));
        }

        /// Run a single iteration of gradient descent.
        /// We use gradient descent to minimize the cost function.
        pub fn learn(
            self: *Self,
            training_data_batch: []const DataPointType,
            learn_rate: f64,
            allocator: std.mem.Allocator,
        ) !void {
            // Use the backpropagation algorithm to calculate the gradient of the cost function
            // (with respect to the network's weights and biases). This is done for each data point,
            // and the gradients are added together.
            for (training_data_batch) |data_point| {
                try self.updateCostGradients(data_point, allocator);
            }

            // TODO: Update to only gradient check at the beginning of training and
            // randomly/sparingly during training with a small batch
            //
            // Gradient checking to make sure our back propagration algorithm is working correctly
            const should_gradient_check = true;
            if (should_gradient_check) {
                try self.sanityCheckCostGradients(training_data_batch, allocator);
            }

            // Gradient descent step: update all weights and biases in the network
            for (self.layers) |*layer| {
                layer.applyCostGradients(
                    // Because we summed the gradients from all of the training data points,
                    // we need to average out all of gradients that we added together. Since
                    // we end up multiplying the gradient values by the learnRate, we can
                    // just divide it by the number of training data points to get the
                    // average gradient.
                    learn_rate / @as(f64, @floatFromInt(training_data_batch.len)),
                );
            }
        }

        /// Gradient checking to make sure our back propagration algorithm is working correctly.
        ///
        /// Check to make sure cost gradients for weights generated from backpropagation
        /// and biases match the estimated cost gradients which are easier to trust (less
        /// moving pieces). We check to make sure that the actual and esimated gradients
        /// match or are a consistent multiple of each other. (TODO: Is a multiple of each other fine?)
        fn sanityCheckCostGradients(
            self: *Self,
            training_data_batch: []const DataPointType,
            allocator: std.mem.Allocator,
        ) !void {
            var test_layer = self.layers[self.layers.len - 1];
            // std.log.debug("Weight: test_layer costGradientWeights {d:.6}", .{test_layer.costGradientWeights});
            // std.log.debug("Bias: test_layer costGradientBiases {d:.6}", .{test_layer.costGradientBiases});

            const estimated_cost_gradients = try self.estimateCostGradientsForLayer(
                &test_layer,
                training_data_batch,
                allocator,
            );
            defer allocator.free(estimated_cost_gradients.costGradientWeights);
            defer allocator.free(estimated_cost_gradients.costGradientBiases);

            // We use `0.0002` instead of `0.0001` because imagine the ratio is truely
            // `2` but due to floating point precision, we find the first number with
            // `1.9999` ratio and the second number with `2.0001` ratio. The
            // difference between the two ratios is `0.0002` which is greater than
            // `0.0001` so we would incorrectly think that the ratio is wrong.
            const ratio_threshold: f64 = 0.0002;
            var found_estimated_to_actual_cost_gradient_ratio: f64 = 0;

            var has_flawed_weight_cost_gradient: bool = false;
            for (estimated_cost_gradients.costGradientWeights, 0..) |estimated_weight_cost, cost_index| {
                const actual_weight_cost = test_layer.costGradientWeights[cost_index];

                // We have this check to watch out for divide by zero
                if (estimated_weight_cost != 0 and actual_weight_cost != 0) {
                    const weight_ratio = estimated_weight_cost / actual_weight_cost;
                    // Set if it's not already set
                    if (found_estimated_to_actual_cost_gradient_ratio == 0) {
                        found_estimated_to_actual_cost_gradient_ratio = weight_ratio;
                    }

                    const ratio_from_found_difference = @fabs(weight_ratio - found_estimated_to_actual_cost_gradient_ratio);
                    const ratio_from_matching_difference = @fabs(weight_ratio - 1);
                    if (
                    // If the ratio is too different from the ratio we found in the
                    // first non-zero weight, then that's suspect since we would expect
                    // the ratio to be the same for all weights.
                    ratio_from_found_difference > ratio_threshold and
                        // We can also sanity check that the ratio is close to 1 since
                        // that means the estimated and actual cost gradients are
                        // roughly the same.
                        ratio_from_matching_difference > ratio_threshold)
                    {
                        has_flawed_weight_cost_gradient = true;
                        std.log.warn("Weight: (ratio: {d:.3}) estimated_weight_cost {d} is too different from our cost gradient calculated by our actual back propagation algorithm {d}", .{
                            weight_ratio,
                            estimated_weight_cost,
                            actual_weight_cost,
                        });
                    }
                }
            }

            var has_flawed_bias_cost_gradient: bool = false;
            for (estimated_cost_gradients.costGradientBiases, 0..) |estimated_bias_cost, cost_index| {
                const actual_bias_cost = test_layer.costGradientBiases[cost_index];

                // We have this check to watch out for divide by zero
                if (estimated_bias_cost != 0 and actual_bias_cost != 0) {
                    const bias_ratio = estimated_bias_cost / actual_bias_cost;
                    // Set if it's not already set
                    if (found_estimated_to_actual_cost_gradient_ratio == 0) {
                        found_estimated_to_actual_cost_gradient_ratio = bias_ratio;
                    }

                    const ratio_from_found_difference = @fabs(bias_ratio - found_estimated_to_actual_cost_gradient_ratio);
                    const ratio_from_matching_difference = @fabs(bias_ratio - 1);
                    if (
                    // If the ratio is too different from the ratio we found in the
                    // first non-zero weight, then that's suspect since we would expect
                    // the ratio to be the same for all weights.
                    ratio_from_found_difference > ratio_threshold and
                        // We can also sanity check that the ratio is close to 1 since
                        // that means the estimated and actual cost gradients are
                        // roughly the same.
                        ratio_from_matching_difference > ratio_threshold)
                    {
                        has_flawed_bias_cost_gradient = true;
                        std.log.warn("Bias: (ratio: {d:.3}) estimated_bias_cost {d} is too different from our cost gradient calculated by our actual back propagation algorithm {d}", .{
                            bias_ratio,
                            estimated_bias_cost,
                            actual_bias_cost,
                        });
                    }
                }
            }

            if (found_estimated_to_actual_cost_gradient_ratio == 0) {
                std.log.err("Unable to find (estimated / actual) cost gradient ratio because the " ++
                    "cost gradients had zeros everywhere (at least there was never a spot " ++
                    "where both had a non-zero number). Maybe check for vanishing gradient " ++
                    "problem as well." ++
                    "\n    Estimated weight gradient: {d:.6}" ++
                    "\n       Actual weight gradient: {d:.6}" ++
                    "\n    Estimated bias gradient: {d:.6}" ++
                    "\n       Actual bias gradient: {d:.6}", .{
                    estimated_cost_gradients.costGradientWeights,
                    test_layer.costGradientWeights,
                    estimated_cost_gradients.costGradientBiases,
                    test_layer.costGradientBiases,
                });
                return error.UnableToFindEstimatedToActualWeightRatio;
            } else if (found_estimated_to_actual_cost_gradient_ratio != 1) {
                // This is just a warning because I don't think it affects the direction
                std.log.warn("The (estimated / actual) cost gradient ratio is {d} " ++
                    "(should be ~1 which means the estimated and actual match) " ++
                    "which means our actual calculated cost is some multiple of the true weight gradient. " ++
                    "This doesn't affect the direction of the gradient (or probably accuracy of the descent " ++
                    "step) but may indicate some slight problem." ++
                    "\n    Estimated weight gradient: {d:.6}" ++
                    "\n       Actual weight gradient: {d:.6}" ++
                    "\n    Estimated bias gradient: {d:.6}" ++
                    "\n       Actual bias gradient: {d:.6}", .{
                    found_estimated_to_actual_cost_gradient_ratio,
                    estimated_cost_gradients.costGradientWeights,
                    test_layer.costGradientWeights,
                    estimated_cost_gradients.costGradientBiases,
                    test_layer.costGradientBiases,
                });
            }

            if (has_flawed_weight_cost_gradient or has_flawed_bias_cost_gradient) {
                std.log.err("The actual cost gradient does not match our estimated cost gradient " ++
                    "(flawed weight cost gradient: {}, flawed bias cost gradient: {})" ++
                    "\n    Estimated weight gradient: {d:.6}" ++
                    "\n       Actual weight gradient: {d:.6}" ++
                    "\n    Estimated bias gradient: {d:.6}" ++
                    "\n       Actual bias gradient: {d:.6}", .{
                    has_flawed_weight_cost_gradient,
                    has_flawed_bias_cost_gradient,
                    estimated_cost_gradients.costGradientWeights,
                    test_layer.costGradientWeights,
                    estimated_cost_gradients.costGradientBiases,
                    test_layer.costGradientBiases,
                });
                return error.FlawedCostGradients;
            }
        }

        /// Used for gradient checking
        ///
        /// This is extremely slow because we have to run the cost function for every weight
        /// and bias in the network against all of the training data points.
        ///
        /// Resources:
        ///  - https://cs231n.github.io/neural-networks-3/#gradcheck
        ///  - This looks like the naive way (finite-difference) to estimate the slope that
        ///    Sebastian Lague started off with in his video,
        ///    https://youtu.be/hfMk-kjRv4c?si=iQohVzk-oFtYldQK&t=937
        fn estimateCostGradientsForLayer(
            self: *Self,
            layer: *Layer,
            training_data_batch: []const DataPointType,
            allocator: std.mem.Allocator,
        ) !struct {
            costGradientWeights: []f64,
            costGradientBiases: []f64,
        } {
            var costGradientWeights: []f64 = try allocator.alloc(f64, layer.num_input_nodes * layer.num_output_nodes);
            var costGradientBiases: []f64 = try allocator.alloc(f64, layer.num_output_nodes);

            // We want h to be small but not too small to cause float point precision problems.
            const h: f64 = 0.0001;
            const original_cost = try self.cost_many(training_data_batch, allocator);
            _ = original_cost;

            // Calculate the cost gradient for the current weights.
            // We're using the the centered difference formula for better accuracy: (f(x + h) - f(x - h)) / 2h
            // The normal finite difference formula has less accuracy: (f(x + h) - f(x)) / h
            for (0..layer.num_output_nodes) |node_index| {
                for (0..layer.num_input_nodes) |node_in_index| {
                    // Make a small nudge the weight in the positive direction (+ h)
                    layer.weights[layer.getFlatWeightIndex(node_index, node_in_index)] += h;
                    // Check how much that nudge causes the cost to change
                    const cost1 = try self.cost_many(training_data_batch, allocator);

                    // Make a small nudge the weight in the negative direction (- h). We
                    // `- 2h` because we nudged the weight in the positive direction by
                    // `h` just above and want to get back original_value first so we
                    // minus h, and then minus h again to get to (- h).
                    layer.weights[layer.getFlatWeightIndex(node_index, node_in_index)] -= 2 * h;
                    // Check how much that nudge causes the cost to change
                    const cost2 = try self.cost_many(training_data_batch, allocator);
                    // Find how much the cost changed between the two nudges
                    const delta_cost = cost1 - cost2;

                    // Reset the weight back to its original value
                    layer.weights[layer.getFlatWeightIndex(node_index, node_in_index)] += h;

                    // Calculate the gradient: change in cost / change in weight (which is h)
                    costGradientWeights[layer.getFlatWeightIndex(node_index, node_in_index)] = delta_cost / (2 * h);
                }
            }

            // Calculate the cost gradient for the current biases
            for (0..layer.num_output_nodes) |node_index| {
                // Make a small nudge the bias (+ h)
                layer.biases[node_index] += h;
                // Check how much that nudge causes the cost to change
                const cost1 = try self.cost_many(training_data_batch, allocator);

                // Make a small nudge the bias in the negative direction (- h). We
                // `- 2h` because we nudged the bias in the positive direction by
                // `h` just above and want to get back original_value first so we
                // minus h, and then minus h again to get to (- h).
                layer.biases[node_index] -= 2 * h;
                // Check how much that nudge causes the cost to change
                const cost2 = try self.cost_many(training_data_batch, allocator);
                // Find how much the cost changed between the two nudges
                const delta_cost = cost1 - cost2;

                // Reset the bias back to its original value
                layer.biases[node_index] += h;

                // Calculate the gradient: change in cost / change in bias (which is h)
                costGradientBiases[node_index] = delta_cost / (2 * h);
            }

            return .{
                .costGradientWeights = costGradientWeights,
                .costGradientBiases = costGradientBiases,
            };
        }

        fn updateCostGradients(
            self: *Self,
            data_point: DataPointType,
            allocator: std.mem.Allocator,
        ) !void {
            // Feed data through the network to calculate outputs. Save all
            // inputs/weighted_inputs/activations along the way to use for
            // backpropagation (`layer.layer_output_data`).
            _ = try self.calculateOutputs(data_point.inputs, allocator);
            defer self.freeAfterCalculateOutputs(allocator);

            // ---- Backpropagation ----
            // Update gradients of the output layer
            const output_layer_index = self.layers.len - 1;
            var output_layer = self.layers[output_layer_index];
            var output_layer_shareable_node_derivatives = try output_layer.calculateOutputLayerShareableNodeDerivatives(
                &data_point.expected_outputs,
                allocator,
            );
            try output_layer.updateCostGradients(
                output_layer_shareable_node_derivatives,
            );

            // Loop backwards through all of the hidden layers and update their gradients
            var shareable_node_derivatives = output_layer_shareable_node_derivatives;
            const num_hidden_layers = self.layers.len - 1;
            for (0..num_hidden_layers) |forward_hidden_layer_index| {
                const backward_hidden_layer_index = num_hidden_layers - forward_hidden_layer_index - 1;
                var hidden_layer = self.layers[backward_hidden_layer_index];

                // Free the shareable_node_derivatives from the last iteration at the
                // end of the block after we're done using it in the next hidden layer.
                const shareable_node_derivatives_to_free = shareable_node_derivatives;
                defer allocator.free(shareable_node_derivatives_to_free);

                shareable_node_derivatives = try hidden_layer.calculateHiddenLayerShareableNodeDerivatives(
                    &self.layers[backward_hidden_layer_index + 1],
                    shareable_node_derivatives,
                    allocator,
                );
                try hidden_layer.updateCostGradients(
                    shareable_node_derivatives,
                );
            }
            // Free the last iteration of the loop
            defer allocator.free(shareable_node_derivatives);
        }
    };
}
