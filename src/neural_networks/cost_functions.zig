const std = @import("std");

// Cost functions are also known as loss functions.

// TODO: In the future, we could add negative log likelihood, MeanAbsoluteError,
// RootMeanSquaredError, etc.

// Used for regression problems where the output data comes from a normal/gaussian
// distribution (TODO: What does this mean?). Does not penalize misclassifications as
// much as it could/should for binary/multi-class classification problems. Although this
// answer says that it doesn't matter, https://stats.stackexchange.com/a/568253/360344.
// Useful when TODO: What?
//
// > The MSE function is non-convex for binary classification. Thus, if a binary
// > classification model is trained with MSE Cost function, it is not guaranteed to
// > minimize the Cost function. Also, using MSE as a cost function assumes the Gaussian
// > distribution which is not the case for binary classification.
// >
// > -- https://stats.stackexchange.com/questions/46413/can-the-mean-squared-error-be-used-for-classification/449436#449436
//
//
//
// TODO: Is this Sum of Squared Errors (SSE) or Mean Squared Error (MSE)?
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

    // TODO: Derivative of what with respect to what?
    pub fn derivative(_: @This(), actual_output: f64, expected_output: f64) f64 {
        return actual_output - expected_output;
    }
};

// Cross-Entropy is also referred to as Logarithmic loss. Used for binary or multi-class
// classification problems where the output data comes from a bernoulli distribution
// which just means we have buckets/categories with expected probabilities.
//
// https://machinelearningmastery.com/cross-entropy-for-machine-learning/
pub const CrossEntropy = struct {
    // We want to calculate the total cost (not the average cost).
    pub fn cost(
        _: @This(),
        actual_outputs: []const f64,
        /// Note: `expected_outputs` are expected to all be either 0 or 1 (probably using one-hot encoding).
        expected_outputs: []const f64,
    ) f64 {
        var cost_sum: f64 = 0;
        for (actual_outputs, expected_outputs) |actual_output, expected_output| {
            var v: f64 = 0;
            if (expected_output == 1.0) {
                v += -1 * @log(actual_output);
            } else {
                v += -1 * @log(1 - actual_output);
            }

            cost_sum += if (std.math.isNan(v)) 0 else v;
        }

        return cost_sum;
    }

    pub fn derivative(_: @This(), actual_output: f64, expected_output: f64) f64 {
        // The function is undefined at 0 and 1, so we return 0 because TODO: Why?
        if (actual_output == 0.0 or actual_output == 1.0) {
            return 0.0;
        }

        return (-1 * actual_output + expected_output) / (actual_output * (actual_output - 1));
    }
};

pub const CostFunction = union(enum) {
    mean_squared_error: MeanSquaredError,
    cross_entropy: CrossEntropy,

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
