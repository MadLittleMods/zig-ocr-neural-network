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

        // This is just complicated logic to handle numbers or strings as labels
        pub fn checkLabelsEqual(a: InputLabelType, b: InputLabelType) bool {
            const is_label_matching = switch (@typeInfo(InputLabelType)) {
                .Int, .Float => a == b,
                .Pointer => |ptr_info| blk: {
                    if (!ptr_info.is_const or ptr_info.size != .Slice or ptr_info.child != u8) {
                        @compileError("unsupported type");
                    }

                    break :blk std.mem.eql(u8, a, b);
                },
                else => @compileError("unsupported type"),
            };

            return is_label_matching;
        }
    };
}
