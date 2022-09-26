#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

#include "helpers.h"


/*
 * Utility functions.
 */



/// <summary>
/// Bilinearly sample ImageFloat or ImageRGB using relative image coordinates [x,y].
/// Note that the coordinate start and end points align with the entire image and not just
/// the pixel centers (0.5x pixel difference).
/// </summary>
/// <typeparam name="T">template type, can be ImageFloat or ImageRGB</typeparam>
/// <param name="image">input image</param>
/// <param name="rel_pos">x,y position</param>
/// <returns>interpolated pixel value (float or glm::vec3)</returns>
template <typename T>
inline T sampleBilinear(const Image<T>& image, const glm::vec2& rel_pos)
{
    // Write a code that bilinearly interpolates values from a generic image (can contain either float or glm::vec3).
    // The rel_pose input represents the (x,y) relative coordinates of the sampled point where:
    //   [0,0] = The left top corner of the left top (=first) pixel.
    //   [1,1] = The right bottom corner of the right bottom (=last) pixel.
    //   [0,1] = The left bottom corner of the left bottom pixel.
    // 
    // Take into account the size of individual pixel and the fact, that the "value" of pixel is conceptually stored in its center.
    //      => For rel_pos between centers of pixels, the method bilinearly interpolates between 4 nearest pixel.
    //      => For rel_pos corresponding to a center of a pixel, the method needs to return an exact value of that pixel. That is not an exception but direct result of the interpolation formula.
    // 
    // Therefore, steps are as follows:
    //     1. Compute absolute position of the sampled point within a pixel grid.
    //     2. Determine 4 nearest pixels.
    //     3. Bilinearly interpolate their values based on the position of the sampling point between them.
    // 
    // Note: The method is templated by parameter "T". This will be either float or glm::vec3 depending on whether the method
    // is called with ImageFloat or ImageRGB. Use either "T" or "auto" to define your variables and use glm::functions to handle both types.
    // Example:
    //    auto value = image.data[0] * 3; // both float and glm:vec3 support baisc operators
    //    T rounded_value = glm::round(image.data[0]); // glm::round will handle both glm::vec3 and float. 
    // Use glm API for further reference: https://glm.g-truc.net/0.9.9/api/a00241.html
    // 
    
    //
    //    YOUR CODE GOES HERE
    //

    return image.data[0]; // <-- Change this.
}


/*
  Core functions.
*/

/// <summary>
/// Applies the bilateral filter on the given disparity image.
/// Ignored pixels that are marked as invalid.
/// </summary>
/// <param name="disparity">The image to be filtered.</param>
/// <param name="guide">The image guide used for calculating the tonal distances between pixel values.</param>
/// <param name="radius">The kernel radius.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageFloat, the filtered disparity.</returns>
ImageFloat jointBilateralFilter(const ImageFloat& disparity, const ImageRGB& guide, const int radius, const float guide_sigma)
{
    // We assume both images have matching dimensions.
    assert(disparity.width == guide.width && disparity.height == guide.height);

    // Rule of thumb for gaussian's std dev. Fills the radius but does not cut off too much.
    const float sigma = radius / 3.2f;

    // Empty output image.
    auto result = ImageFloat(disparity.width, disparity.height);


    //
    // Implement a bilateral filter of the disparity image guided by the guide.
    // Ignore all contributing pixels where disparity == INVALID_VALUE.
    //
    // 1. Iterate over all output pixels.
    // 2. For each output pixel, visit all neighboring pixels (including itself) in [-radius;radius]^2 neighborhood. 
    //    That is the same as during convolution with a (radius*2+1)x(radius*2+1) filter.
    // 3. If a neighbor is outside of the image or its disparity == INVALID_VALUE, skip the pixel (move to the next one).
    // 4. For each neighbor compute its weight as w_i = gauss(radius, sigma) * gauss(diff_value, guide_sigma)
    //      where
    //          * radius is the Eucledian distance between the center and current pixel in pixels (L2 norm).
    //          * diff_value is difference of guide image pixel values for the center and current pixel.
    //          * gauss(x, sigma) is a Normal distribution pdf function for x=x and std.dev.= sigma which is available for use.
    // 5. Compute weighted mean of all (unskipped) neighboring pixel disparities and assign it to the output.
    // 
    // Refer to the first Lecture or recommended study books for more info on Bilateral filtering.
    // 
    // Notes:
    //   * If a pixel has no neighbor (all were skipped), assign INVALID_VALUE to the output.  
    //   * One point awarded for a correct OpenMP parallelization.

    //
    //    YOUR CODE GOES HERE
    //

    auto example = gauss(0.5f, 1.2f); // This is just example of computing Normal pdf for x=0.5 and std.dev=1.2.


    // Return filtered disparity.
    return result;
}

/// <summary>
/// In-place normalizes and an ImageFloat image to be between 0 and 1.
/// All values marked as invalid will stay marked as invalid.
/// </summary>
/// <param name="scalar_image"></param>
/// <returns></returns>
void normalizeValidValues(ImageFloat& scalar_image)
{
    //
    // Find minimum and maximum among the VALID image values.
    // Linearly rescale the VALID image values to the [0,1] range (in-place).
    // The INVALID values remain INVALID (they are ignored).
    // 
    // Note #1: Pixel is INVALID as long as value == INVALID_VALUE.
    // Note #2: This modified the input image in-place => no "return".
    //
    
    //
    //    YOUR CODE GOES HERE
    //
}

/// <summary>
/// Converts a disparity image to a normalized depth image.
/// Ignores invalid disparity values.
/// </summary>
/// <param name="disparity">disparity in arbitrary units</param>
/// <returns>linear depth scaled from 0 to 1</returns>
ImageFloat disparityToNormalizedDepth(const ImageFloat& disparity)
{
    auto depth = ImageFloat(disparity.width, disparity.height);

    //
    // Convert disparity to a depth with unknown scale:
    //    depth_unscaled = 1.0 / disparity
    // Keep invalid pixels marked as invalid (disparity == INVALID_VALUE).
    //
        
    //
    //    YOUR CODE GOES HERE
    //

    // Rescales valid depth values to [0,1] range.
    normalizeValidValues(depth);

    return depth;
}

/// <summary>
/// Convert linear normalized depth to target pixel disparity.
/// Invalid pixels 
/// </summary>
/// <param name="depth">Normalized depth image ([0,1])</param>
/// <param name="iod_mm">Inter-ocular distance in mm.</param>
/// <param name="px_size_mm">Pixel size in mm.</param>
/// <param name="screen_distance_mm">Screen distance from eyes in mm.</param>
/// <param name="near_plane_mm">Near plane distance from eyes in mm.</param>
/// <param name="far_plane_mm">Far plane distance from eyes in mm.</param>
/// <returns>screen disparity in pixels</returns>
ImageFloat normalizedDepthToDisparity(
    const ImageFloat& depth, const float iod_mm,
    const float px_size_mm, const float screen_distance_mm,
    const float near_plane_mm, const float far_plane_mm)
{
    auto px_disparity = ImageFloat(depth.width, depth.height);

    //
    // Based on physical dimensions, distance, resolution (and hence pixel size) of the screen,
    // as well as physiologically determined distance between viewers pupil (IOD or IPD),
    // compute stereoscopic pixel disparities that will make the display appear at a correct depth
    // represented by linear interpolation between the near and far plane based on the depth input image.
    // 
    // Refer to Lecture 4 for formulas.
    // 
    // Example:
    //    screen distance = 600 mm, near_plane_mm = 550, far_plane == 650, depth = 0.1  
    //         => the pixel should appear 55+0.1(65-55) = 56 cm away from the user
    //         => That is 4 cm in front of the screen.
    //         => That means the pixel disparity will be a negative number ("crossed disparity").
    // 
    // Note:
    //    * All distances are measured orthogonal ot the screen and are assumed constant across the screen (ignores the eccentricity variance).
    //    * Invalid pixels (depth==INVALID_VALUE) are to be marked invalid on the output as well.
    //
    
    //
    //    YOUR CODE GOES HERE
    //

    return px_disparity; // returns disparity measured in pixels
}

/// <summary>
/// Creates a warping grid for an image of specified height and weight.
/// It produces vertex buffer which stores 2D positions of pixel corners,
/// and index buffer which defines triangles by triplets of indices into
/// the vertex buffer (the three vertices form a triangle).
/// 
/// </summary>
/// <param name="width">Image width.</param>
/// <param name="height">Image height.</param>
/// <returns>Mesh, containing a vertex buffer and triangle index buffer.</returns>
Mesh createWarpingGrid(const int width, const int height)
{

    // Build vertex buffer.
    auto num_vertices = (width + 1) * (height + 1);
    auto vertices = std::vector<glm::vec2>(num_vertices);

    //
    // Fill the vertex buffer (vertices) with 2D coordinate of the pixel corners.
    // Expected output coordinates are:
    //   [0,0] for the left top corner of the left top (=first) pixel.
    //   [1,1] for the right bottom corner of the right bottom (=last) pixel.
    //   [0,1] for the left bottom corner of the left bottom pixel.
    // 
    // The order in memory is to be the same as for images (row after row).
    // 
    
    //
    //    YOUR CODE GOES HERE
    //

    // Build index buffer.
    auto num_pixels = width * height;
    auto num_triangles = num_pixels * 2;
    auto triangles = std::vector<glm::ivec3>(num_triangles);

    //
    // Fill the index buffer (triangles) with indices pointing to the vertex buffer.
    // Each element of the "triangles" is an integer triplet (glm::ivec3). 
    // It represents a triangle by selecting 3 vertices from the vertex buffer defining its corners in
    // a clockwise manner.
    // We need to fill the index buffer in the same order as pixels are stored in memory (that is row by row)
    // and for each pixel we should generate two triangles that together cover the are of a pixel as follows:
    // 
    //   A ------- B
    //   |  *      |
    //   |    *    |
    //   |      *  |
    //   D ------- C
    // 
    // Where A,B,C,D are the CORNERS of the respective pixel.
    // 
    // For each such pixel, we add two triangles: 
    //     glm::ivec3(A,B,C) and glm::ivec3(A,C,D)  (in this exact order)
    // where A,B,C,D are indices into the vertex buffer.
    // 
    // The result should be a grid that fills an entire image and replaces each pixel with two small triangles.
    //
    
    //
    //    YOUR CODE GOES HERE
    //
    // Combine the vertex and index buffers into a mesh.
    return Mesh { std::move(vertices), std::move(triangles) };
}

/// <summary>
/// Warps a grid based on the given disparity and scaling_factor.
/// </summary>
/// <param name="grid">The original mesh.</param>
/// <param name="disparity">Disparity for each PIXEL.</param>
/// <param name="scaling_factor">Global scaling factor for the disparity.</param>
/// <returns>Mesh, the warped grid.</returns>
Mesh warpGrid(Mesh& grid, const ImageFloat& disparity, const float scaling_factor)
{
    // Create a copy of the input mesh (all values are copied).
    auto new_grid = Mesh { grid.vertices, grid.triangles };

    const float EDGE_EPSILON = 1e-5f;

    //
    // The goal is to modify the x coordinate of the grid vertices based on 
    // the scaled pixel disparity corresponding to the original location of the vertex buffer:
    // 
    // new_grid.vertex.x = grid.vertex.x + scaling_factor * sampled_disparity
    // 
    // where sampled_disparity is a bilinearly interpolated value from the disparity image
    // which we can easily obtained using our other function "sampleBilinear":
    //     sampled_disparity = sampleBilinear(disparity, grid.vertex)
    // 
    // IMPORTANT - in order to keep the grid attached to the image "frame",
    // we must not move the border vertices => do not move vertices
    // that are within EDGE_EPSILON from the image boundary in either x or y
    // direction.
    //

    // Here is an example use of our bilinear interpolation.
    auto interpolated_value = sampleBilinear(disparity, glm::vec2(0.5f, 0.5f));
    // For a even-sized image it SHOULD return the mean of the center 4 pixels,
    // for an odd-sized image it SHOULD return the central pixel.
    // I recommend you to test that.

    //
    //    YOUR CODE GOES HERE
    //



    return new_grid;
}

/// <summary>
/// Forward-warps an image based on the given disparity and warp_factor.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="disparity">Disparity of the source image in pixels.</param>
/// <param name="warp_factor">Multiplier of the disparity.</param>
/// <returns>ImageRGB, the forward-warped image.</returns>
ImageRGB fowardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const ImageFloat& disparity, const float warp_factor)
{
    // The dimensions of src image, src depth and disparity maps all match.
    assert(src_image.width == disparity.width && src_image.height == disparity.height);
    assert(src_image.width == disparity.width && src_depth.height == src_depth.height);
    
    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), std::numeric_limits<float>::max());

    // 
    // The goal is to forward warp the image pixels using the disparity displacement provided in 
    // the disparity map along with additional scaling factor. Furthermore, we
    // use depth information to resolve conflicts when multiple pixels attempt
    // to write a single output pixel. 
    //
    // 1. For every input pixel, compute where it should be warped to 
    //    based on the associated disparity and warp_factor. 
    //    Use standard rounding rules to obtain integer position (ie., 0.5 rounds up).
    //      x' = round(x + disparity * warp_factor)
    //      y' = y
    // 
    // 2. Check the destination depth at the [x',y'] location and compare it with the 
    //    depth of the currently warped pixel (ie., depth[yi,xi]).
    // 
    // 3. If the currently warped pixel has a smaller depth than the previous value in the output depth buffer,
    //    overwrite the output buffer (otherwise do nothing). This means writing both to the destination image
    //    and to the destination depth map.
    // 
    // 4. Return the final destination image.
    // 
    // Note: One point is awarded for a correct and efficient parallel solution using OpenMP.
    //
    
    //
    //    YOUR CODE GOES HERE
    //


    // Return the warped image.
    return dst_image;

}

/// <summary>
/// Backward-warps an image using a warped mesh.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="src_grid">Source grid.</param>
/// <param name="dst_grid">The warped grid.</param>
/// <returns>ImageRGB, the backward-warped image.</returns>
ImageRGB backwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, Mesh& src_grid, Mesh& dst_grid)
{
    // The dimensions of src image and depth match.
    assert(src_image.width == src_depth.width && src_image.height == src_depth.height);
    // We assume that both grids have the same size and also the same order (ie., there is 1:1 triangle pairing).
    // This implies that the content of index buffers of both meshes are exactly equal (we do not test it here).
    assert(src_grid.triangles.size() == dst_grid.triangles.size());

    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), 1e20f);

    //
    // This method implements mesh-based warping by rasterizing
    // each triangle of the destination grid and sampling
    // the source texture by looking up corresponding 
    // position in the source grid using barycentric coordinates.
    // 
    // 1. For every triangle in the warped grid (dst_grid), 
    //    determine X and Y ranges of destination pixel coordinates
    //    whose **centers** lie in the bounding box of the triangle.
    //    Note: Look back to createWarpingGrid() for definition
    //          of the grid vertex coordinates.
    // 
    // 2. Enumerate all candidate pixels within the bounding box.
    //    Skip pixels whose centers actually do not lie inside the triangle.
    //    Use the provided isPointInsideTriangle(pt_dst, vert_a, vert_b, vert_c)
    //    method to do the test.
    // 
    // 3. Compute barycentric coordinates of the pixel center in the triangle
    //    using the provided method bc = barycentricCoordinates(pt_dst, vert_a, vert_b, vert_c).
    // 
    // 4. Find the corresponding triangle in the original mesh. It has the same triangle index.
    // 
    // 5. Reproject the pixel center to the source grid by using the already computed barycentric
    //    coordinates:
    //        pt_src = (vert_a_src, vert_b_src, vert_c_src) * bc
    // 
    // 6. Bilinearly sample both the source depth map and the source image values at the pt_src
    //    position. 
    //    Hint: Use the sampleBilinear() method implemented earlier.
    // 
    // 7. Compare the depth in the source and destination depth values and implement the depth-test
    //    like in fowardWarpImage().
    //    Hint: The destination map can be accessed directly without interpolation because we are
    //          computing for an exact pixel coordinate.
    // 
    // 8. If the depth test passes, write out the destination image (dst_image) and depth (dst_depth) values. 
    //    Again, it is the same logic as in fowardWarpImage().
    // 
    // 9. Return the warped image (dst_image).
    //
    
    // Example of testing point [0.1, 0.2] is inside a triangle.
    bool is_point_inside = isPointInsideTriangle(glm::vec2(0.1, 0.2), glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0, 1));

    // Example of computing barycentric coordinates of a point [0.1, 0.2] inside a triangle.
    glm::vec2 bc = barycentricCoordinates(glm::vec2(0.1, 0.2), glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(0, 1));

    //
    //    YOUR CODE GOES HERE
    //

    // Return the warped image.
    return dst_image;
}

/// <summary>
/// Returns an anaglyph image.
/// </summary>
/// <param name="image_left">left RGB image</param>
/// <param name="image_right">right RGB image</param>
/// <param name="saturation">color saturation to apply</param>
/// <returns>ImageRGB, the anaglyph image.</returns>
ImageRGB createAnaglyph(const ImageRGB& image_left, const ImageRGB& image_right, const float saturation)
{
    // An empty image for the resulting anaglyph.
    auto anaglyph = ImageRGB(image_left.width, image_left.height);

    // 
    // Convert stereoscopic pair into a single anaglyph stereoscopic image
    // for viewing in red-cyan anaglyph glasses.
    // We additionally scale saturation of the image to make the image
    // more "grayscale" since colors are problematic in analglyph image
    // and increase crosstalk (ghosting).
    // 
    // For both left and rigt image:
    // 1. Convert RGB to HSV color space using the provided rgbToHsv() function.
    // 2. Scale the saturation (stored in the second (=Y) component of the vec3) by the "saturation" param.
    // 3. Convert back to RGB using the hsvToRgb().
    // 
    // Combine the two images such that:
    //    * output.red = left.red
    //    * output.green = right.green
    //    * output.blue = right.blue.
    //

    // Example: RGB->HSV->RGB should be approx identity.
    auto rgb_orig = glm::vec3(0.2, 0.6, 0.4);
    auto rgb_should_be_same = hsvToRgb(rgbToHsv(rgb_orig)); // expect rgb == rgb_2 (up to numerical precision)

    //
    //    YOUR CODE GOES HERE
    //

    // Returns a single analgyph image.
    return anaglyph;
}
