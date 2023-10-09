const std = @import("std");

pub fn DataPoint(
    /// The type of the label. This can be an integer, float, or string (`[]const u8`).
    comptime LabelType: type,
    /// The possible labels for a data point.
    comptime labels: []const LabelType,
) type {
    return struct {
        const Self = @This();

        inputs: []const f64,
        expected_outputs: [labels.len]f64,
        label: LabelType,

        pub fn init(inputs: []const f64, label: LabelType) Self {
            return .{
                .inputs = inputs,
                .expected_outputs = oneHotEncodeLabel(label),
                .label = label,
            };
        }

        fn oneHotEncodeLabel(label: LabelType) [labels.len]f64 {
            var one_hot = std.mem.zeroes([labels.len]f64);
            for (labels, 0..) |comparison_label, label_index| {
                const is_label_matching = switch (@typeInfo(LabelType)) {
                    .Int, .Float => comparison_label == label,
                    .Pointer => |ptr_info| blk: {
                        if (!ptr_info.is_const or ptr_info.size != .Slice or ptr_info.child != u8) {
                            @compileError("unsupported type");
                        }

                        break :blk std.mem.eql(u8, comparison_label, label);
                    },
                    else => @compileError("unsupported type"),
                };

                if (is_label_matching) {
                    one_hot[label_index] = 1.0;
                } else {
                    one_hot[label_index] = 0.0;
                }
            }
            return one_hot;
        }
    };
}
