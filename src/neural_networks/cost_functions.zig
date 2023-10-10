// Cost functions are also known as loss functions.

// TODO: In the future, we could add CrossEntropy, negative log likelihood,
// MeanAbsoluteError, RootMeanSquaredError, etc.

pub const MeanSquaredError = struct {
    pub fn many_cost(self: @This(), actual_outputs: []const f64, expected_outputs: []const f64) f64 {
        var cost_sum: f64 = 0;
        for (actual_outputs, expected_outputs) |actual_output, expected_output| {
            cost_sum += self.individual_cost(actual_output, expected_output);
        }

        // Return the average cost (sum / number of outputs)
        return cost_sum / @as(f64, @floatFromInt(actual_outputs.len));
    }

    pub fn individual_cost(_: @This(), actual_output: f64, expected_output: f64) f64 {
        const error_difference = actual_output - expected_output;
        return (error_difference * error_difference) / 2;
    }

    pub fn individual_derivative(_: @This(), actual_output: f64, expected_output: f64) f64 {
        return actual_output - expected_output;
    }
};

pub const CostFunction = union(enum) {
    mean_squared_error: MeanSquaredError,

    pub fn many_cost(self: @This(), actual_outputs: []const f64, expected_outputs: []const f64) f64 {
        return switch (self) {
            inline else => |case| case.many_cost(actual_outputs, expected_outputs),
        };
    }
    pub fn individual_cost(self: @This(), actual_output: f64, expected_output: f64) f64 {
        return switch (self) {
            inline else => |case| case.individual_cost(actual_output, expected_output),
        };
    }
    pub fn individual_derivative(self: @This(), actual_output: f64, expected_output: f64) f64 {
        return switch (self) {
            inline else => |case| case.individual_derivative(actual_output, expected_output),
        };
    }
};
