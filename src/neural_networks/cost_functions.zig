// Cost functions are also known as loss functions.

// TODO: In the future, we could add CrossEntropy, negative log likelihood,
// MeanAbsoluteError, RootMeanSquaredError, etc.

pub const MeanSquaredError = struct {
    // We want to calculate the total cost (not the average cost).
    pub fn cost(_: @This(), actual_outputs: []const f64, expected_outputs: []const f64) f64 {
        var cost_sum: f64 = 0;
        for (actual_outputs, expected_outputs) |actual_output, expected_output| {
            const error_difference = actual_output - expected_output;
            cost_sum += (error_difference * error_difference);
        }

        // > We multiply our MSE cost function by 1/2 so that when we take the derivative,
        // > the 2s cancel out. Multiplying the cost function by a scalar does not affect
        // > the location of its minimum, so we can get away with this.
        // >
        // > Alternatively, you could think of this as folding the 2 into the learning
        // > rate. It makes sense to leave the 1/m term, though, because we want the same
        // > learning rate (alpha) to work for different training set sizes (m).
        // >
        // > https://mccormickml.com/2014/03/04/gradient-descent-derivation/#one-half-mean-squared-error
        const one_half_mean_squared_error = 0.5 * cost_sum;

        return one_half_mean_squared_error;
    }

    pub fn derivative(_: @This(), actual_output: f64, expected_output: f64) f64 {
        return actual_output - expected_output;
    }
};

pub const CostFunction = union(enum) {
    mean_squared_error: MeanSquaredError,

    pub fn cost(self: @This(), actual_outputs: []const f64, expected_outputs: []const f64) f64 {
        return switch (self) {
            inline else => |case| case.cost(actual_outputs, expected_outputs),
        };
    }
    pub fn derivative(self: @This(), actual_output: f64, expected_output: f64) f64 {
        return switch (self) {
            inline else => |case| case.derivative(actual_output, expected_output),
        };
    }
};
