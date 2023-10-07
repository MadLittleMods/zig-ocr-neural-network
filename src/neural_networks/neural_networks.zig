const std = @import("std");
const Layer = @import("layer.zig").Layer;

pub const ActivationFunction = @import("activation_functions.zig").ActivationFunction;

pub fn NeuralNetwork(comptime layer_sizes: []const u32) type {
    return struct {
        const Self = @This();

        activation_function: ActivationFunction,
        layers: []Layer,

        pub fn init(activation_function: ActivationFunction, allocator: std.mem.Allocator) Self {
            var layers = try allocator.alloc(Layer, layer_sizes.len - 1);
            for (layers, 0..) |*layer, layer_index| {
                layer.* = try Layer.init(
                    layer_sizes[layer_index],
                    layer_sizes[layer_index + 1],
                    activation_function,
                    allocator,
                );
            }

            return Self{
                .activation_function = activation_function,
                .layers = layers,
            };
        }

        // Run the input values through the network to calculate the output values
        pub fn calculateOutputs(self: *Self, inputs: [layer_sizes[0]]f64) [layer_sizes[layer_sizes.len - 1]]f64 {
            for (self.layers) |layer| {
                inputs = layer.calculateOutputs(inputs);
            }

            return inputs;
        }

        // Run the input values through the network and calculate which output node has
        // the highest value
        pub fn classify(self: *Self, inputs: [layer_sizes[0]]f64) u32 {
            var outputs = self.calculateOutputs(inputs);
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

        pub fn cost(self: *Self, data_point: DataPoint) f64 {
            _ = self;
            // TODO
            return 0.0;
        }

        /// Run a single iteration of gradient descent (using the finite-difference method)
        /// We use gradient descent to minimize the cost function.
        pub fn Learn(self: *Self, training_batch: []DataPoint, learn_rate: f64) void {
            const h: f64 = 0.0001;
            const original_cost = self.cost(training_data);

            // TODO
        }

        fn UpdateAllGradients(self: *Self, data_point: DataPoint) void {

            // TODO
            var output_layer = self.layers[self.layers.len - 1];
            output_layer.CalculateOutputLayerNodeValues(data_point.expected_outputs);
        }
    };
}
