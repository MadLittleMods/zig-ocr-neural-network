// Cost functions are also known as loss functions.

// TODO: In the future, we could add CrossEntropy

pub const MeanSquaredError = struct {
    pub fn cost(actual_outputs: []f64, expected_outputs: []f64) f64 {
        var cost_sum: f64 = 0;
        for (actual_outputs, expected_outputs) |actual_output, expected_output| {
            const error_difference = actual_output - expected_output;
            cost_sum += (error_difference) ** 2;
        }

        // TODO: Verify this is correct
        return cost_sum / @as(u64, actual_outputs.len);
    }

    pub fn derivative(actual_output: f64, expected_output: f64) f64 {
        // TODO: Verify this is correct
        return actual_output - expected_output;
    }
};

pub const CostFunction = union(enum) {
    MeanSquaredError: MeanSquaredError,

    pub fn cost(self: @This(), actual_outputs: []f64, expected_outputs: []f64) f64 {
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
