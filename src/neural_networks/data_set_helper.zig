const std = @import("std");

// TODO: We can use `@intCast(u64, std.time.timestamp())` to get a seed that changes
const seed = 123;
var prng = std.rand.DefaultPrng.init(seed);
const random_instance = prng.random();

/// Create a a list of lookups in the data
pub fn createLookupIndices(num_items: usize, allocator: std.mem.Allocator) ![]usize {
    var lookup_indices = try allocator.alloc(usize, num_items);
    for (0..lookup_indices.len) |index| {
        lookup_indices[index] = index;
    }

    return lookup_indices;
}

/// Use after every training epoch
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
