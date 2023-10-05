// Activation functions allows a layer to have a non-linear affect on the output so they
// can bend the boundary around the data. Without this, the network would only be able
// to separate data with a straight line.

// TODO: In the future, we could add Sigmoid, Tanh, SiLU, Softmax, etc to try out

// ReLU
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
pub const Relu = struct {
    pub fn activate(input: f64) f64 {
        return @max(0.0, input);

        // Or in other words:
        //
        // if (x > 0.0) {
        //     return x;
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

pub const ActivationFunction = union(enum) {
    Relu: Relu,

    pub fn activate(self: @This(), input: f64) f64 {
        return switch (self) {
            inline else => |case| case.activate(input),
        };
    }
    pub fn derivative(self: @This(), input: f64) f64 {
        return switch (self) {
            inline else => |case| case.derivative(input),
        };
    }
};
