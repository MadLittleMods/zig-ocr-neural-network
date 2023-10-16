const std = @import("std");

// Activation functions allows a layer to have a non-linear affect on the output so they
// can bend the boundary around the data. Without this, the network would only be able
// to separate data with a straight line.

// TODO: In the future, we could add Sigmoid, Tanh, SiLU, Softmax, LeakyReLU, etc to try out

// ReLU
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
//
// ReLU should probably only be used on hidden layers (TODO: Why?)
//
// > The operation of ReLU is closer to the way our biological neurons work. (TODO: Why?)
// >
// > ReLU is non-linear and has the advantage of not having any backpropagation errors
// > unlike the sigmoid function
//
// https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
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

// LeakyReLU
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
//
// Like ReLU except when x < 0, LeakyReLU will have a small negative slope instead of a
// hard zero. This slope is typically a small value like 0.01.
//
// > [This solves the Dying ReLU problem] we see in ReLU [...] where some ReLU Neurons
// > essentially die for all inputs and remain inactive no matter what input is
// > supplied, here no gradient flows and if large number of dead neurons are there in a
// > Neural Network it’s performance is affected, this can be corrected by making use of
// > what is called Leaky ReLU.
// >
// > -- https://himanshuxd.medium.com/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e
pub const LeakyRelu = struct {
    pub fn activate(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        if (input > 0.0) {
            return input;
        }
        return 0.1 * input;
    }

    pub fn derivative(_: @This(), inputs: []const f64, input_index: usize) f64 {
        const input = inputs[input_index];
        if (input > 0.0) {
            return 1.0;
        }
        return 0.1;
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

// SoftMax squishes the output between [0, 1] (TODO: inclusive or exclusive?) and all
// the resulting elements add up to 1. So in terms of this usage, this function will
// tell you what percentage that the given value at the `input_index` makes up the total
// sum of all the values in the array.
//
// TODO: Visualize this (ASCII art)
// TODO: Why would someone use this one?
// https://machinelearningmastery.com/softmax-activation-function-with-python/
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

/// Estimate the slope of the activation function at the given input using the
/// ActivationFunction's `activate` function. We can use this to compare against the
/// ActivationFunction's `derivative` function to make sure it's correct.
///
/// We're using the the centered difference formula for better accuracy: (f(x + h) - f(x - h)) / 2h
/// The normal finite difference formula has less accuracy: (f(x + h) - f(x)) / h
fn estimateSlopeOfActivationFunction(
    activation_function: ActivationFunction,
    inputs: []const f64,
    input_index: usize,
) !f64 {
    var mutable_inputs = try std.testing.allocator.alloc(f64, inputs.len);
    defer std.testing.allocator.free(mutable_inputs);
    @memcpy(mutable_inputs, inputs);

    // We want h to be small but not too small to cause float point precision problems.
    const h = 0.0001;

    // Make a small nudge the input in the positive direction (+ h)
    mutable_inputs[input_index] += h;
    // Check how much that nudge causes the result to change
    const result1 = activation_function.activate(mutable_inputs, input_index);

    // Make a small nudge the weight in the negative direction (- h). We
    // `- 2h` because we nudged the weight in the positive direction by
    // `h` just above and want to get back original_value first so we
    // minus h, and then minus h again to get to (- h).
    mutable_inputs[input_index] -= 2 * h;
    // Check how much that nudge causes the cost to change
    const result2 = activation_function.activate(mutable_inputs, input_index);
    // Find how much the cost changed between the two nudges
    const delta_result = result1 - result2;

    // Reset the input back to its original value
    mutable_inputs[input_index] += h;

    // Calculate the gradient: change in activation / change in input (which is 2h)
    const estimated_slope = delta_result / (2 * h);

    return estimated_slope;
}

const ActivationTestCase = struct {
    activation_function: ActivationFunction,
    inputs: []const f64,
    input_index: usize,
};

// Cross-check the activate function against the derivative function to make sure they
// relate and match up to each other.
test "Slope check activation functions" {
    var test_cases = [_]ActivationTestCase{
        .{
            .activation_function = ActivationFunction{ .relu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .relu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        // ReLU is not differentiable at 0.0 so our estimatation would run into a kink
        // and make the estimated slope innaccurate. So inaccurate that comparing the
        // derivative function against the estimated slope would fail.
        // .{
        //     .activation_function = ActivationFunction{ .relu = .{} },
        //     .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
        //     .input_index = 2,
        // },
        .{
            .activation_function = ActivationFunction{ .leaky_relu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .leaky_relu = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        // LeakyReLU is not differentiable at 0.0 so our estimatation would run into a kink
        // and make the estimated slope innaccurate. So inaccurate that comparing the
        // derivative function against the estimated slope would fail.
        // .{
        //     .activation_function = ActivationFunction{ .leaky_relu = .{} },
        //     .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
        //     .input_index = 2,
        // },
        .{
            .activation_function = ActivationFunction{ .sigmoid = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .sigmoid = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .sigmoid = .{} },
            .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .soft_max = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .soft_max = .{} },
            .inputs = &[_]f64{ 0.1, 0.2, 0.3 },
            .input_index = 2,
        },
        .{
            .activation_function = ActivationFunction{ .soft_max = .{} },
            .inputs = &[_]f64{ -0.2, 0.1, 0.0, 0.1, 0.2 },
            .input_index = 2,
        },
    };

    for (test_cases) |test_case| {
        var activation_function = test_case.activation_function;
        var inputs = test_case.inputs;
        const input_index = test_case.input_index;

        // Estimate the slope of the activation function at the given input
        const estimated_slope = try estimateSlopeOfActivationFunction(
            activation_function,
            inputs,
            input_index,
        );
        // A derivative is just the slope of the given function. So the slope returned
        // by the derivative function should be the same as the slope we estimated.
        const actual_slope = activation_function.derivative(inputs, input_index);

        // Check to make sure the actual slope is within a certain threshold of the
        // estimated slope
        const threshold = 0.0001;
        if (@fabs(estimated_slope - actual_slope) > threshold) {
            std.debug.print("{s}: Expected actual slope {d} to be within {d} of the estimated slope: {d} (which we assume to ~correct)\n", .{
                activation_function.getName(),
                actual_slope,
                threshold,
                estimated_slope,
            });
            return error.FaultySlope;
        }
    }
}

pub const ActivationFunction = union(enum) {
    relu: Relu,
    leaky_relu: LeakyRelu,
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

    pub fn getName(self: @This()) []const u8 {
        return switch (self) {
            .relu => "ReLU",
            .leaky_relu => "LeakyReLU",
            .sigmoid => "Sigmoid",
            .soft_max => "SoftMax",
        };
    }
};
