pub const activation_functions = @import("neural_networks/activation_functions.zig");
pub const cost_functions = @import("neural_networks/cost_functions.zig");

test {
    // https://ziglang.org/documentation/master/#Nested-Container-Tests
    @import("std").testing.refAllDecls(@This());
}
