const std = @import("std");

// TODO: We can use `@intCast(u64, std.time.timestamp())` to get a seed that changes
const seed = 123;
var prng = std.rand.DefaultPrng.init(seed);
const random_instance = prng.random();

/// Create a a list of lookups in the data initialized sequentially [0, 1, 2, ...]
pub fn createLookupIndices(num_items: usize, allocator: std.mem.Allocator) ![]usize {
    var lookup_indices = try allocator.alloc(usize, num_items);
    for (0..lookup_indices.len) |index| {
        lookup_indices[index] = index;
    }

    return lookup_indices;
}

/// Use after every training epoch to shuffle the indices so that we don't always
/// train on the same data in the same order.
pub fn shuffleLookupIndicesInPlace(lookup_indices: []usize) void {
    random_instance.shuffle(usize, lookup_indices);
}

/// Instead of shuffling the data itself, we just shuffle a list of indices into the
/// data and then use them to assemble a batch.
pub fn assembleShuffledBatch(
    comptime DataPointType: type,
    data_points: []const DataPointType,
    lookup_indices: []usize,
    batch_index: u32,
    batch_size: u32,
    allocator: std.mem.Allocator,
) ![]DataPointType {
    const batch = try allocator.alloc(DataPointType, batch_size);

    for (0..batch.len) |batch_item_index| {
        const data_point_index = lookup_indices[batch_index * batch_size + batch_item_index];
        batch[batch_item_index] = data_points[data_point_index];
    }

    return batch;
}

test "createLookupIndices" {
    const allocator = std.heap.page_allocator;
    const lookup_indices = try createLookupIndices(3, allocator);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, lookup_indices);
}

test "shuffleLookupIndicesInPlace" {
    const allocator = std.testing.allocator;
    var lookup_indices = try createLookupIndices(10, allocator);
    defer allocator.free(lookup_indices);

    shuffleLookupIndicesInPlace(lookup_indices);

    // Assert that the indices are shuffled by checking that they are no longer in the
    // same order that they were intially (sequentially).
    //
    // XXX: It's entirely possible for this test to fail on different platforms (even
    // though the random seed is fixed) because Zigs shuffle doesn't yield consistent
    // results across different platforms and it might so happen that our "random"
    // result is the same as the initial order. We've hedged this a little by using 10
    // items here so it should be very unlikely.
    //
    // This assertion is also a bit annoying since it outputs logs to the console when
    // the inner `std.testing.expectEqualSlices` assertion fails even though we expect
    // it to fail.
    try std.testing.expectError(
        error.TestExpectedEqual,
        std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, lookup_indices),
    );
}

test "assembleShuffledBatch" {
    const allocator = std.testing.allocator;
    const data_points = [_][]const u8{ "foo", "bar", "baz" };
    var lookup_indices = [_]usize{ 1, 0, 2 };
    const shuffled_batch = try assembleShuffledBatch(
        []const u8,
        &data_points,
        &lookup_indices,
        0,
        3,
        allocator,
    );
    defer allocator.free(shuffled_batch);

    try std.testing.expectEqualSlices(
        []const u8,
        &[_][]const u8{ "bar", "foo", "baz" },
        shuffled_batch,
    );
}
