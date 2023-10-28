const std = @import("std");

pub fn DataPoint(
    /// The type of the label. This can be an integer, float, or string (`[]const u8`).
    comptime InputLabelType: type,
    /// The possible labels for a data point.
    comptime labels: []const InputLabelType,
) type {
    return struct {
        const Self = @This();

        pub const LabelType = InputLabelType;
        pub const label_list = labels;

        inputs: []const f64,
        expected_outputs: [labels.len]f64,
        label: InputLabelType,

        pub fn init(inputs: []const f64, label: InputLabelType) Self {
            return .{
                .inputs = inputs,
                .expected_outputs = oneHotEncodeLabel(label),
                .label = label,
            };
        }

        fn oneHotEncodeLabel(label: InputLabelType) [labels.len]f64 {
            var one_hot = std.mem.zeroes([labels.len]f64);
            for (labels, 0..) |comparison_label, label_index| {
                const is_label_matching = checkLabelsEqual(comparison_label, label);
                if (is_label_matching) {
                    one_hot[label_index] = 1.0;
                } else {
                    one_hot[label_index] = 0.0;
                }
            }
            return one_hot;
        }

        pub fn oneHotIndexToLabel(one_hot_index: usize) InputLabelType {
            return labels[one_hot_index];
        }

        pub fn labelToOneHotIndex(label: InputLabelType) !usize {
            for (labels, 0..) |comparison_label, label_index| {
                const is_label_matching = checkLabelsEqual(comparison_label, label);
                if (is_label_matching) {
                    return label_index;
                }
            }

            switch (@typeInfo(InputLabelType)) {
                .Int, .Float => {
                    std.log.err("Unable to find label {d} in label list {any}", .{ label, labels });
                },
                .Pointer => |ptr_info| {
                    if (!ptr_info.is_const or ptr_info.size != .Slice or ptr_info.child != u8) {
                        @compileError("unsupported type");
                    }

                    // We found the label to be a string (`[]const u8`)
                    std.log.err("Unable to find label {s} in label list {any}", .{ label, labels });
                },
                else => @compileError("unsupported type"),
            }
            return error.LabelNotFound;
        }

        // This is just complicated logic to handle numbers or strings as labels
        pub fn checkLabelsEqual(a: InputLabelType, b: InputLabelType) bool {
            const is_label_matching = switch (@typeInfo(InputLabelType)) {
                .Int, .Float => a == b,
                .Pointer => |ptr_info| blk: {
                    if (!ptr_info.is_const or ptr_info.size != .Slice or ptr_info.child != u8) {
                        @compileError("unsupported type");
                    }

                    // Compare strings (`[]const u8`)
                    break :blk std.mem.eql(u8, a, b);
                },
                else => @compileError("unsupported type"),
            };

            return is_label_matching;
        }
    };
}
