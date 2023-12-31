const std = @import("std");
const Layer = @import("layer.zig").Layer;
const LayerOutputData = @import("layer.zig").LayerOutputData;

pub const ActivationFunction = @import("activation_functions.zig").ActivationFunction;
pub const CostFunction = @import("cost_functions.zig").CostFunction;
pub const DataPoint = @import("data_point.zig").DataPoint;

// Here are the thresholds of error we should be tolerating:
//
// >  - relative error > 1e-2 usually means the gradient is probably wrong
// >  - 1e-2 > relative error > 1e-4 should make you feel uncomfortable
// >  - 1e-4 > relative error is usually okay for objectives with kinks.
// >    But if there are no kinks (e.g. use of tanh nonlinearities and softmax),
// >    then 1e-4 is too high.
// >  - 1e-7 and less you should be happy.
// >
// > -- https://cs231n.github.io/neural-networks-3/#gradcheck
fn calculateRelativeError(a: f64, b: f64) f64 {
    // We have this check to watch out for divide by zero
    if (a != 0 and b != 0) {
        // > Notice that normally the relative error formula only includes one of
        // > the two terms (either one), but I prefer to max (or add) both to make
        // > it symmetric and to prevent dividing by zero in the case where one of
        // > the two is zero (which can often happen, especially with ReLUs)
        // >
        // > -- https://cs231n.github.io/neural-networks-3/#gradcheck
        //
        // |a - b| / max(|a|, |b|)
        const relative_error = @fabs(a - b) / @max(@fabs(a), @fabs(b));
        return relative_error;
    }

    return 0;
}

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
            for (testing_data_points) |*testing_data_point| {
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
            data_point: *const DataPointType,
            allocator: std.mem.Allocator,
        ) !f64 {
            var outputs = try self.calculateOutputs(data_point.inputs, allocator);
            defer self.freeAfterCalculateOutputs(allocator);

            return self.cost_function.vector_cost(outputs, &data_point.expected_outputs);
        }

        /// Calculate the total cost of the network for a batch of data points
        pub fn cost_many(
            self: *Self,
            data_points: []const DataPointType,
            allocator: std.mem.Allocator,
        ) !f64 {
            var total_cost: f64 = 0.0;
            for (data_points) |*data_point| {
                const cost_of_data_point = try self.cost_individual(data_point, allocator);
                // std.log.debug("cost_of_data_point: {d}", .{cost_of_data_point});
                total_cost += cost_of_data_point;
            }
            return total_cost;
        }

        /// Calculate the average cost of the network for a batch of data points
        pub fn cost_average(
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
            /// See the comment in `Layer.updateCostGradients()` for more info
            momentum: f64,
            allocator: std.mem.Allocator,
        ) !void {
            // Use the backpropagation algorithm to calculate the gradient of the cost function
            // (with respect to the network's weights and biases). This is done for each data point,
            // and the gradients are added together.
            for (training_data_batch) |*data_point| {
                try self.updateCostGradients(data_point, allocator);
            }

            // TODO: Update to only gradient check at the beginning of training and
            // randomly/sparingly during training with a small batch
            //
            // Gradient checking to make sure our back propagration algorithm is working correctly
            const should_gradient_check = false;
            if (should_gradient_check) {
                try self.sanityCheckCostGradients(training_data_batch, allocator);
            }

            // Gradient descent step: update all weights and biases in the network
            for (self.layers) |*layer| {
                layer.applyCostGradients(
                    // Because we summed the gradients from all of the training data points,
                    // we need to average out all of gradients that we added together. Since
                    // we end up multiplying the gradient values by the learn_rate, we can
                    // just divide it by the number of training data points to get the
                    // average gradient.
                    learn_rate / @as(f64, @floatFromInt(training_data_batch.len)),
                    momentum,
                );
            }
        }

        /// Gradient checking to make sure our back propagration algorithm is working correctly.
        ///
        /// Check to make sure cost gradients for weights generated from backpropagation
        /// and biases match the estimated cost gradients which are easier to trust (less
        /// moving pieces). We check to make sure that the actual and esimated gradients
        /// match or are a consistent multiple of each other.
        fn sanityCheckCostGradients(
            self: *Self,
            training_data_batch: []const DataPointType,
            allocator: std.mem.Allocator,
        ) !void {
            var test_layer = self.layers[self.layers.len - 1];
            // std.log.debug("Weight: test_layer cost_gradient_weights {d:.6}", .{test_layer.cost_gradient_weights});
            // std.log.debug("Bias: test_layer cost_gradient_biases {d:.6}", .{test_layer.cost_gradient_biases});

            // Also known as the "numerical gradient" as opposed to the actual
            // "analytical gradient"
            const estimated_cost_gradients = try self.estimateCostGradientsForLayer(
                &test_layer,
                training_data_batch,
                allocator,
            );
            defer allocator.free(estimated_cost_gradients.cost_gradient_weights);
            defer allocator.free(estimated_cost_gradients.cost_gradient_biases);

            const gradients_to_compare = [_]struct { gradient_name: []const u8, actual_gradient: []f64, estimated_gradient: []f64 }{
                .{
                    .gradient_name = "weight",
                    .actual_gradient = test_layer.cost_gradient_weights,
                    .estimated_gradient = estimated_cost_gradients.cost_gradient_weights,
                },
                .{
                    .gradient_name = "bias",
                    .actual_gradient = test_layer.cost_gradient_biases,
                    .estimated_gradient = estimated_cost_gradients.cost_gradient_biases,
                },
            };

            var found_relative_error: f64 = 0;
            var has_uneven_cost_gradient: bool = false;
            var was_relative_error_too_high: bool = false;
            for (gradients_to_compare) |gradient_to_compare| {
                // Calculate the relative error between the values in the estimated and
                // actual cost gradients. We want to make sure the relative error is not
                // too high.
                // =========================================================================
                for (gradient_to_compare.actual_gradient, gradient_to_compare.estimated_gradient, 0..) |a_value, b_value, gradient_index| {
                    const relative_error = calculateRelativeError(a_value, b_value);
                    // Set if it's not already set
                    if (found_relative_error == 0) {
                        found_relative_error = relative_error;
                    }

                    // Here are the thresholds of error we should be tolerating:
                    //
                    // >  - relative error > 1e-2 usually means the gradient is probably wrong
                    // >  - 1e-2 > relative error > 1e-4 should make you feel uncomfortable
                    // >  - 1e-4 > relative error is usually okay for objectives with kinks.
                    // >    But if there are no kinks (e.g. use of tanh nonlinearities and softmax),
                    // >    then 1e-4 is too high.
                    // >  - 1e-7 and less you should be happy.
                    // >
                    // > -- https://cs231n.github.io/neural-networks-3/#gradcheck
                    if (relative_error > 1e-2) {
                        std.log.err("Relative error for index {d} in {s} gradient was too high ({d}).", .{
                            gradient_index,
                            gradient_to_compare.gradient_name,
                            relative_error,
                        });
                        was_relative_error_too_high = true;
                    } else if (relative_error > 1e-4) {
                        // > Note that it is possible to know if a kink was crossed in the
                        // > evaluation of the loss. This can be done by keeping track of
                        // > the identities of all "winners" in a function of form max(x,y);
                        // > That is, was x or y higher during the forward pass. If the
                        // > identity of at least one winner changes when evaluating f(x+h)
                        // > and then f(x−h), then a kink was crossed and the numerical
                        // > gradient will not be exact.
                        // >
                        // > -- https://cs231n.github.io/neural-networks-3/#gradcheck
                        std.log.warn("Relative error for index {d} in {s} gradient is pretty high " ++
                            "but if there was a kink in the objective, this level of error ({d}) is acceptable when " ++
                            "crossing one of those kinks.", .{
                            gradient_index,
                            gradient_to_compare.gradient_name,
                            relative_error,
                        });
                    }

                    if (
                    // Compare the error to the first non-zero error we found. If the
                    // error is too different then that's suspect since we would expect
                    // the error to be the same for all weights/biases.
                    std.math.approxEqAbs(f64, relative_error, found_relative_error, 1e-4) and
                        // We can also sanity check whether the error is close to 0
                        // since that means the estimated and actual cost gradients are
                        // roughly the same.
                        relative_error > 1e-4)
                    {
                        has_uneven_cost_gradient = true;
                    }
                }
            }

            if (found_relative_error == 0) {
                std.log.err("Unable to find relative error because the " ++
                    "cost gradients had zeros everywhere (at least there was never a spot " ++
                    "where both had a non-zero number). Maybe check for a vanishing gradient " ++
                    "problem." ++
                    "\n    Estimated weight gradient: {d:.6}" ++
                    "\n       Actual weight gradient: {d:.6}" ++
                    "\n    Estimated bias gradient: {d:.6}" ++
                    "\n       Actual bias gradient: {d:.6}", .{
                    estimated_cost_gradients.cost_gradient_weights,
                    test_layer.cost_gradient_weights,
                    estimated_cost_gradients.cost_gradient_biases,
                    test_layer.cost_gradient_biases,
                });
                return error.UnableToFindRelativeErrorOfEstimatedToActualGradient;
            } else if (found_relative_error > 1e-4) {
                const uneven_error_message = "The relative error is the different across the entire gradient which " ++
                    "means the gradient is pointing in a totally different direction than it should. " ++
                    "Our backpropagation algorithm is probably wrong.";

                const even_error_message = "The relative error is the same across the entire gradient so even though " ++
                    "the actual value is different than the estimated value, it doesn't affect the direction " ++
                    "of the gradient or accuracy of the gradient descent step but may indicate some " ++
                    "slight problem.";

                // This is just a warning because I don't think it affects the
                // direction. If that assumption is wrong, then this should be an error.
                std.log.warn("The first relative error we found is {d} " ++
                    "(should be ~0 which indicates the estimated and actual gradients match) " ++
                    "which means our actual cost gradient values are some multiple of the estimated weight gradient. " ++
                    "{s}" ++
                    "\n    Estimated weight gradient: {d:.6}" ++
                    "\n       Actual weight gradient: {d:.6}" ++
                    "\n    Estimated bias gradient: {d:.6}" ++
                    "\n       Actual bias gradient: {d:.6}", .{
                    found_relative_error,
                    if (has_uneven_cost_gradient) uneven_error_message else even_error_message,
                    estimated_cost_gradients.cost_gradient_weights,
                    test_layer.cost_gradient_weights,
                    estimated_cost_gradients.cost_gradient_biases,
                    test_layer.cost_gradient_biases,
                });
            }

            // We only consider it an error when the error was too high and the error
            // was inconsisent across the gradient which means we're just stepping in
            // a completely wrong direction.
            if (was_relative_error_too_high and has_uneven_cost_gradient) {
                std.log.err("Relative error in cost gradients was too high meaning that " ++
                    "some values in the estimated vs actual cost gradients were too different " ++
                    "which means our backpropagation algorithm is probably wrong and we're " ++
                    "probably stepping in an arbitrarily wrong direction. " ++
                    "\n    Estimated weight gradient: {d:.6}" ++
                    "\n       Actual weight gradient: {d:.6}" ++
                    "\n    Estimated bias gradient: {d:.6}" ++
                    "\n       Actual bias gradient: {d:.6}", .{
                    estimated_cost_gradients.cost_gradient_weights,
                    test_layer.cost_gradient_weights,
                    estimated_cost_gradients.cost_gradient_biases,
                    test_layer.cost_gradient_biases,
                });
                return error.RelativeErrorTooHigh;
            }
        }

        /// Used for gradient checking to calculate the esimated gradient to compare
        /// against (also known as the "numerical gradient" as opposed to the actual
        /// "analytical gradient").
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
            cost_gradient_weights: []f64,
            cost_gradient_biases: []f64,
        } {
            var cost_gradient_weights: []f64 = try allocator.alloc(f64, layer.num_input_nodes * layer.num_output_nodes);
            var cost_gradient_biases: []f64 = try allocator.alloc(f64, layer.num_output_nodes);

            // We want h to be small but not too small to cause float point precision problems.
            const h: f64 = 0.0001;

            // Calculate the cost gradient for the current weights.
            // We're using the the centered difference formula for better accuracy: (f(x + h) - f(x - h)) / 2h
            // The normal finite difference formula has less accuracy: (f(x + h) - f(x)) / h
            //
            // We need to be aware of kinks in the objective introduced by activation
            // functions like ReLU. Imagine our weight is just below 0 on the x-axis and
            // we nudge the weight just above 0, we would estimate some value when the
            // actual value is 0 (with ReLU(x), any x <= 0 will result in 0). Ideally,
            // we should have some leniency during the gradient check as it's expected
            // that our estimated gradient will not match our actual gradient exactly
            // when we hit a kink.
            for (0..layer.num_output_nodes) |node_index| {
                for (0..layer.num_input_nodes) |node_in_index| {
                    const weight_index = layer.getFlatWeightIndex(node_index, node_in_index);

                    // Make a small nudge to the weight in the positive direction (+ h)
                    layer.weights[weight_index] += h;
                    // Check how much that nudge causes the cost to change
                    const cost1 = try self.cost_many(training_data_batch, allocator);

                    // Make a small nudge to the weight in the negative direction (- h). We
                    // `- 2h` because we nudged the weight in the positive direction by
                    // `h` just above and want to get back original_value first so we
                    // minus h, and then minus h again to get to (- h).
                    layer.weights[weight_index] -= 2 * h;
                    // Check how much that nudge causes the cost to change
                    const cost2 = try self.cost_many(training_data_batch, allocator);
                    // Find how much the cost changed between the two nudges
                    const delta_cost = cost1 - cost2;

                    // Reset the weight back to its original value
                    layer.weights[weight_index] += h;

                    // Calculate the gradient: change in cost / change in weight (which is 2h)
                    cost_gradient_weights[weight_index] = delta_cost / (2 * h);
                }
            }

            // Calculate the cost gradient for the current biases
            for (0..layer.num_output_nodes) |node_index| {
                // Make a small nudge to the bias (+ h)
                layer.biases[node_index] += h;
                // Check how much that nudge causes the cost to change
                const cost1 = try self.cost_many(training_data_batch, allocator);

                // Make a small nudge to the bias in the negative direction (- h). We
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

                // Calculate the gradient: change in cost / change in bias (which is 2h)
                cost_gradient_biases[node_index] = delta_cost / (2 * h);
            }

            return .{
                .cost_gradient_weights = cost_gradient_weights,
                .cost_gradient_biases = cost_gradient_biases,
            };
        }

        fn updateCostGradients(
            self: *Self,
            data_point: *const DataPointType,
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
