const std = @import("std");
const Layer = @import("layer.zig").Layer;
const LayerOutputData = @import("layer.zig").LayerOutputData;

pub const ActivationFunction = @import("activation_functions.zig").ActivationFunction;
pub const CostFunction = @import("cost_functions.zig").CostFunction;
pub const DataPoint = @import("data_point.zig").DataPoint;

pub fn NeuralNetwork(comptime DataPointType: type) type {
    return struct {
        const Self = @This();

        activation_function: ActivationFunction,
        cost_function: CostFunction,
        layers: []Layer,

        pub fn init(
            layer_sizes: []const u32,
            activation_function: ActivationFunction,
            cost_function: CostFunction,
            allocator: std.mem.Allocator,
        ) !Self {
            var layers = try allocator.alloc(Layer, layer_sizes.len - 1);
            for (layers, 0..) |*layer, layer_index| {
                layer.* = try Layer.init(
                    layer_sizes[layer_index],
                    layer_sizes[layer_index + 1],
                    activation_function,
                    allocator,
                    .{
                        // We just set the cost function for all layers even though it's
                        // only needed for the last output layer.
                        .cost_function = cost_function,
                    },
                );
            }

            if (layers[layers.len - 1].cost_function) |_| {
                // no-op
            } else {
                std.log.err("NeuralNetwork.init(...): The cost function for the output layer (the last layer in the neural network) must be specified", .{});
                return error.CostFunctionNotSpecifiedForOutputLayer;
            }

            return Self{
                .activation_function = activation_function,
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
        pub fn calculateOutputs(self: *Self, inputs: []const f64) ![]f64 {
            for (self.layers) |*layer| {
                inputs = try layer.calculateOutputs(inputs);
            }

            return inputs;
        }

        // Run the input values through the network and calculate which output node has
        // the highest value
        pub fn classify(self: *Self, inputs: []f64) !u32 {
            var outputs = try self.calculateOutputs(inputs);
            var max_output = outputs[0];
            var max_output_index = 0;
            for (outputs, 0..) |output, index| {
                if (output > max_output) {
                    max_output = output;
                    max_output_index = index;
                }
            }
            return max_output_index;
        }

        /// Calculate the cost of the network for a single data point
        pub fn cost_individual_data_point(self: *Self, data_point: DataPointType) !f64 {
            var outputs = try self.calculateOutputs(data_point.inputs);
            return self.cost_function.many_cost(outputs, &data_point.expected_outputs);
        }

        /// Calculate the average cost of the network for a batch of data points
        pub fn cost(self: *Self, data_points: []const DataPointType) !f64 {
            var total_cost: f64 = 0.0;
            for (data_points) |data_point| {
                total_cost += try self.cost_individual_data_point(data_point);
            }
            return total_cost / @as(f64, @floatFromInt(data_points.len));
        }

        /// Run a single iteration of gradient descent (using the finite-difference method).
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

        fn updateCostGradients(
            self: *Self,
            data_point: DataPointType,
            allocator: std.mem.Allocator,
        ) !void {
            // Feed data through the network to calculate outputs. Save all
            // inputs/weighted_inputs/activations along the way to use for
            // backpropagation.
            var inputs_to_next_layer = data_point.inputs;
            for (self.layers) |*layer| {
                inputs_to_next_layer = try layer.calculateOutputs(inputs_to_next_layer, allocator);
            }

            // ---- Backpropagation ----
            // Update gradients of the output layer
            const output_layer_index = self.layers.len - 1;
            var output_layer = self.layers[output_layer_index];
            var shareable_node_derivatives = output_layer.calculateOutputLayerShareableNodeDerivatives(
                &data_point.expected_outputs,
            );
            try output_layer.updateCostGradients(
                shareable_node_derivatives,
            );

            // Loop backwards through all of the hidden layers and update their gradients
            var hidden_layer_index = output_layer_index - 1;
            while (hidden_layer_index >= 0) : (hidden_layer_index -= 1) {
                var hidden_layer = self.layers[hidden_layer_index];
                shareable_node_derivatives = hidden_layer.calculateHiddenLayerShareableNodeDerivatives(
                    self.layers[hidden_layer_index + 1],
                    shareable_node_derivatives,
                );
                try hidden_layer.updateCostGradients(
                    shareable_node_derivatives,
                );
            }
        }
    };
}
