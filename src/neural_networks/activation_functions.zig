// Activation functions allows a layer to have a non-linear affect on the output so they
// can bend the boundary around the data. Without this, the network would only be able
// to separate data with a straight line.

// TODO: In the future, we could add Sigmoid, Tanh, SiLU, Softmax, LeakyReLU, etc to try out

// ReLU
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
pub const Relu = struct {
    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        return @max(0.0, input);

        // Or in other words:
        //
        // if (input > 0.0) {
        //     return input;
        // }
        // return 0.0;
    }

    pub fn derivative(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        if (input > 0.0) {
            return 1.0;
        }
        return 0.0;
    }
};

// Sigmoid
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
pub const Sigmoid = struct {
    const Self = @This();

    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        return 1.0 / (1.0 + @exp(-input));
    }

    pub fn derivative(self: @This(), inputs: []const f64, input_index: usize) f64 {
        const activation_value = self.activate(inputs, input_index);
        return activation_value * (1.0 - activation_value);
    }
};

// SoftMax
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
pub const SoftMax = struct {
    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        var exp_sum: f64 = 0.0;
        for (inputs) |input| {
            exp_sum += @exp(input);
        }

        const exp_input = @exp(inputs[input_index]);

        return exp_input / exp_sum;
    }

    pub fn derivative(_: @This(), inputs: []const f64, input_index: usize) f64 {
        var exp_sum: f64 = 0.0;
        for (inputs) |input| {
            exp_sum += @exp(input);
        }

        const exp_input = @exp(inputs[input_index]);

        return (exp_input * exp_sum - exp_input * exp_input) / (exp_sum * exp_sum);
    }
};

pub const ActivationFunction = union(enum) {
    relu: Relu,
    sigmoid: Sigmoid,
    soft_max: SoftMax,

    pub fn activate(self: @This(), inputs: []const f64, input_index: usize) f64 {
        return switch (self) {
            inline else => |case| case.activate(inputs, input_index),
        };
    }

    /// A derivative is just the slope of the activation function at a given point.
    pub fn derivative(self: @This(), inputs: []const f64, input_index: usize) f64 {
        return switch (self) {
            inline else => |case| case.derivative(inputs, input_index),
        };
    }
};
