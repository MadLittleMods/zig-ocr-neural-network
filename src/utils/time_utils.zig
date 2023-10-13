const std = @import("std");

pub const ONE_SECOND_MS = 1000;
pub const ONE_MINUTE_MS = 60 * ONE_SECOND_MS;
pub const ONE_HOUR_MS = 60 * ONE_MINUTE_MS;
pub const ONE_DAY_MS = 24 * ONE_HOUR_MS;

const TimeBreakdown = struct {
    d: i64,
    h: i64,
    m: i64,
    s: i64,
    ms: i64,
};

pub fn formatDuration(input_ms: i64, allocator: std.mem.Allocator) ![]const u8 {
    // Get the absolute value of the input
    const ms = if (input_ms > 0) input_ms else -1 * input_ms;

    const time_breakdown = TimeBreakdown{
        .d = @divFloor(ms, ONE_DAY_MS),
        .h = @mod(@divFloor(ms, ONE_HOUR_MS), 24),
        .m = @mod(@divFloor(ms, ONE_MINUTE_MS), 60),
        .s = @mod(@divFloor(ms, ONE_SECOND_MS), 60),
        .ms = @mod(@divFloor(ms, ONE_SECOND_MS), 1000),
    };

    const fields = std.meta.fields(@TypeOf(time_breakdown));
    var time_parts = try std.ArrayList([]const u8).initCapacity(allocator, fields.len);
    inline for (fields) |field| {
        const value = @field(time_breakdown, field.name);
        if (value > 0) {
            const time_part = try std.fmt.allocPrint(allocator, "{d}{s}", .{ value, field.name });
            try time_parts.append(time_part);
        }
    }

    var duration_string = try std.mem.join(allocator, "", time_parts.items);

    for (time_parts.items) |time_part| {
        allocator.free(time_part);
    }

    return duration_string;
}
