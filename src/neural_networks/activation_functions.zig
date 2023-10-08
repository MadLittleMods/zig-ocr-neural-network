// Activation functions allows a layer to have a non-linear affect on the output so they
// can bend the boundary around the data. Without this, the network would only be able
// to separate data with a straight line.

// TODO: In the future, we could add Sigmoid, Tanh, SiLU, Softmax, LeakyReLU, etc to try out

// ReLU
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
pub const Relu = struct {
    pub fn activate(input: f64) f64 {
        return @max(0.0, input);

        // Or in other words:
        //
        // if (input > 0.0) {
        //     return input;
        // }
        // return 0.0;
    }

    pub fn derivative(input: f64) f64 {
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

    pub fn activate(input: f64) f64 {
        return 1.0 / (1.0 + @exp(-input));
    }

    pub fn derivative(input: f64) f64 {
        const activation_value = Self.activate(input);
        return activation_value * (1.0 - activation_value);
    }
};

pub const ActivationFunction = union(enum) {
    Relu: Relu,
    Sigmoid: Sigmoid,

    pub fn activate(self: @This(), input: f64) f64 {
        return switch (self) {
            inline else => |case| case.activate(input),
        };
    }

    /// A derivative is just the slope of the activation function at a given point.
    pub fn derivative(self: @This(), input: f64) f64 {
        return switch (self) {
            inline else => |case| case.derivative(input),
        };
    }
};
