const std = @import("std");
const neural_networks = @import("../neural_networks/neural_networks.zig");
const createPortablePixMap = @import("create_portable_pix_map.zig").createPortablePixMap;

/// Create a graph of the neural network's decision boundary (we can only visualize
/// this because there are only 2 inputs to the neural network which we can map to
/// the 2d image).
pub fn graphNeuralNetwork(
    comptime DataPointType: type,
    neural_network: *neural_networks.NeuralNetwork(DataPointType),
    training_data_points: []const DataPointType,
    test_data_points: []const DataPointType,
    allocator: std.mem.Allocator,
) !void {
    const width: u32 = 400;
    const height: u32 = 400;
    var pixels: []u24 = try allocator.alloc(u24, width * height);
    defer allocator.free(pixels);

    // For every pixel in the graph, run the neural network and color the pixel based on
    // the output of the neural network.
    for (0..height) |height_index| {
        for (0..width) |width_index| {
            // Normalize the pixel coordinates to be between 0 and 1
            const x = @as(f64, @floatFromInt(width_index)) / @as(f64, @floatFromInt(width));
            const y = @as(f64, @floatFromInt(
                // Flip the Y axis so that the origin (0, 0) in our graph is at the bottom left of the image
                height - height_index - 1,
            )) / @as(f64, @floatFromInt(height));
            const predicted_label = try neural_network.classify(&[_]f64{ x, y }, allocator);

            var pixel_color: u24 = 0x000000;
            if (std.mem.eql(u8, predicted_label, "fish")) {
                pixel_color = 0x4444aa;
            } else if (std.mem.eql(u8, predicted_label, "goat")) {
                pixel_color = 0xaa4444;
            } else {
                @panic("Unknown label");
            }
            pixels[height_index * width + width_index] = pixel_color;
        }
    }

    // Draw a ball for every training point
    for (training_data_points) |*data_point| {
        const label = data_point.label;
        var pixel_color: u24 = 0x000000;
        if (std.mem.eql(u8, label, "fish")) {
            pixel_color = 0x6666ff;
        } else if (std.mem.eql(u8, label, "goat")) {
            pixel_color = 0xff6666;
        } else {
            @panic("Unknown label");
        }

        // Draw the border/shadow of the ball
        drawBallOnPixelCanvasForDataPoint(
            DataPointType,
            .{
                .pixels = pixels,
                .width = width,
                .height = height,
            },
            data_point,
            10,
            0x111111,
        );

        // Draw the colored part of the ball
        drawBallOnPixelCanvasForDataPoint(
            DataPointType,
            .{
                .pixels = pixels,
                .width = width,
                .height = height,
            },
            data_point,
            8,
            pixel_color,
        );
    }

    // Draw a ball for every test point
    for (test_data_points) |*data_point| {
        const label = data_point.label;
        var pixel_color: u24 = 0x000000;
        if (std.mem.eql(u8, label, "fish")) {
            pixel_color = 0xcc33cc;
        } else if (std.mem.eql(u8, label, "goat")) {
            pixel_color = 0xcccc33;
        } else {
            @panic("Unknown label");
        }

        // Draw the border/shadow of the ball
        drawBallOnPixelCanvasForDataPoint(
            DataPointType,
            .{
                .pixels = pixels,
                .width = width,
                .height = height,
            },
            data_point,
            8,
            0x111111,
        );

        // Draw the colored part of the ball
        drawBallOnPixelCanvasForDataPoint(
            DataPointType,
            .{
                .pixels = pixels,
                .width = width,
                .height = height,
            },
            data_point,
            6,
            pixel_color,
        );
    }

    const ppm_file_contents = try createPortablePixMap(pixels, width, height, allocator);
    defer allocator.free(ppm_file_contents);
    const file = try std.fs.cwd().createFile("simple_xy_animal_graph.ppm", .{});
    defer file.close();

    try file.writeAll(ppm_file_contents);
}

fn drawBallOnPixelCanvasForDataPoint(
    comptime DataPointType: type,
    pixel_canvas: struct {
        pixels: []u24,
        width: u32,
        height: u32,
    },
    data_point: *const DataPointType,
    ball_size: u32,
    draw_color: u24,
) void {
    const x_continuous = data_point.inputs[0] * @as(f64, @floatFromInt(pixel_canvas.width));
    const y_continuous = (
    // Flip the Y axis so that the origin (0, 0) in our graph is at the bottom left of the image
        1 - data_point.inputs[1]) * @as(f64, @floatFromInt(pixel_canvas.height));
    const x: u32 = @intFromFloat(x_continuous);
    const y: u32 = @intFromFloat(y_continuous);

    const ball_start_x = x - (ball_size / 2);
    const ball_start_y = y - (ball_size / 2);
    for (ball_start_x..ball_start_x + ball_size) |target_x| {
        for (ball_start_y..ball_start_y + ball_size) |target_y| {
            pixel_canvas.pixels[target_y * pixel_canvas.width + target_x] = draw_color;
        }
    }
}
