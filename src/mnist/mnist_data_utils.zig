const std = @import("std");
const bigEndianStructDeserializer = @import("big_endian_struct_deserializer.zig").bigEndianStructDeserializer;

pub const MnistLabelFileHeader = extern struct {
    /// The magic number is used to identify the file type (doesn't really matter to us)
    magic_number: u32,
    number_of_labels: u32,
};

pub const MnistImageFileHeader = extern struct {
    /// The magic number is used to identify the file type (doesn't really matter to us).
    magic_number: u32,
    number_of_images: u32,
    number_of_rows: u32,
    number_of_columns: u32,
};

pub const RawImageData = [28 * 28]u8;

pub const Image = extern struct {
    width: u8 = 28,
    height: u8 = 28,
    pixels: RawImageData,
};

pub const LabelType = u8;
pub const LabeledImage = extern struct {
    label: LabelType,
    image: Image,
};

pub fn MnistData(comptime HeaderType: type, comptime ItemType: type) type {
    return struct {
        header: HeaderType,
        items: []const ItemType,
    };
}

// This method works against the standard MNIST dataset files, which can be downloaded from:
// http://yann.lecun.com/exdb/mnist/
pub fn readMnistFile(
    comptime HeaderType: type,
    comptime ItemType: type,
    file_path: []const u8,
    comptime numberOfItemsFieldName: []const u8,
    number_of_items_to_read: u32,
    allocator: std.mem.Allocator,
) !MnistData(HeaderType, ItemType) {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    // Use a buffered reader for better performance (less syscalls)
    // (https://zig.news/kristoff/how-to-add-buffering-to-a-writer-reader-in-zig-7jd)
    var buffered_reader = std.io.bufferedReader(file.reader());
    var file_reader = buffered_reader.reader();

    const header = try file_reader.readStructBig(HeaderType);

    // Make sure we don't try to allocate more images than there are in the file
    if (number_of_items_to_read > @field(header, numberOfItemsFieldName)) {
        std.log.err("Trying to read more items than there are in the file {} > {}", .{
            number_of_items_to_read,
            @field(header, numberOfItemsFieldName),
        });
        return error.UnableToReadMoreItemsThanInFile;
    }

    const image_data_array = try allocator.alloc(ItemType, number_of_items_to_read);
    const deserializer = bigEndianStructDeserializer(file_reader);
    for (0..number_of_items_to_read) |image_index| {
        const image = try deserializer.read(ItemType);
        image_data_array[image_index] = image;
    }

    return .{
        .header = header,
        .items = image_data_array[0..],
    };
}
