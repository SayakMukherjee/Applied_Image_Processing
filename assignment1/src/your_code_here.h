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

template<typename T>
int getImageOffset(const Image<T>& image, int x, int y)
{
    // Return offset of the pixel at column x and row y in memory such that 
    // the pixel can be accessed by image.data[offset].
    // 
    // Note, that the image is stored in row-first order, 
    // ie. is the order of [x,y] pixels is [0,0],[1,0],[2,0]...[0,1],[1,1][2,1],...
    //
    // Image size can be accessed using image.width and image.height.
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/

    return y * image.width + x ;
}



glm::vec2 getRGBImageMinMax(const ImageRGB& image) {

    auto min_val = 0.0f;
    auto max_val = 0.0f;

    // Write a code that will return minimum value (min of all color channels and pixels) and maximum value as a glm::vec2(min,max).
    
    // Note: Parallelize the code using OpenMP directives for full points.
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/

    auto num_pixels = image.width * image.height;


    // Loop for maximum value
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; i++) {

        if (image.data[i].x > max_val || image.data[i].y > max_val || image.data[i].z > max_val) {

            #pragma omp critical
            if (image.data[i].x > max_val) {

                max_val = std::max(max_val, image.data[i].x);
            }

            #pragma omp critical
            if (image.data[i].y > max_val) {

                max_val = std::max(max_val, image.data[i].y);
            }

            #pragma omp critical
            if (image.data[i].z > max_val) {

                max_val = std::max(max_val, image.data[i].z);
            }
        }
    }

    // Set minimum as maximum value found
    min_val = max_val;

    // Loop for minimum value
    #pragma omp parallel for
    for (int i = 0; i < num_pixels; i++) {

        if (image.data[i].x < min_val || image.data[i].y < min_val || image.data[i].z < min_val) {

            #pragma omp critical
            if (image.data[i].x < min_val) {

                min_val = std::min(min_val, image.data[i].x);
            }

            #pragma omp critical
            if (image.data[i].y < min_val) {

                min_val = std::min(min_val, image.data[i].y);
            }

            #pragma omp critical
            if (image.data[i].z < min_val) {

                min_val = std::min(min_val, image.data[i].z);
            }
        }

    }

    // Return min and max value as x and y components of a vector.
    return glm::vec2(min_val, max_val);
}


ImageRGB normalizeRGBImage(const ImageRGB& image)
{
    // Create an empty image of the same size as input.
    auto result = ImageRGB(image.width, image.height);

    // Find min and max values.
    glm::vec2 min_max = getRGBImageMinMax(image);

    // Fill the result with normalized image values (ie, fit the image to [0,1] range).    
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/

    auto num_pixels = image.width * image.height;

    //std::cout << "Min value: " + std::to_string(min_max[0]) << std::endl;
    //std::cout << "Max value: " + std::to_string(min_max[1]) << std::endl;

    for (int i = 0; i < num_pixels; i++) {

        result.data[i] = (image.data[i] - min_max[0]) / (min_max[1] - min_max[0]);
        
    }

    return result;
}

ImageRGB applyGamma(const ImageRGB& image, const float gamma)
{
    // Create an empty image of the same size as input.
    auto result = ImageRGB(image.width, image.height);

    // Fill the result with gamma mapped pixel values (result = image^gamma).    
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/

    auto num_pixels = image.width * image.height;

    for (int i = 0; i < num_pixels; i++) {

        result.data[i] = glm::vec3(std::pow(image.data[i].x, gamma), std::pow(image.data[i].y, gamma), std::pow(image.data[i].z, gamma));

    }

    return result;
}

/*
    Main algorithm.
*/

/// <summary>
/// Compute luminance from a linear RGB image.
/// </summary>
/// <param name="rgb">A linear RGB image</param>
/// <returns>log-luminance</returns>
ImageFloat rgbToLuminance(const ImageRGB& rgb)
{
    // RGB to luminance weights defined in ITU R-REC-BT.601 in the R,G,B order.
    const auto WEIGHTS_RGB_TO_LUM = glm::vec3(0.299f, 0.587f, 0.114f);
    // An empty luminance image.
    auto luminance = ImageFloat(rgb.width, rgb.height);
    // Fill the image by logarithmic luminace.
    // Luminance is a linear combination of the red, green and blue channels using the weights above.

    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/

    auto num_pixels = rgb.width * rgb.height;

    for (int i = 0; i < num_pixels; i++) {
        luminance.data[i] = (rgb.data[i].x * WEIGHTS_RGB_TO_LUM[0]) + (rgb.data[i].y * WEIGHTS_RGB_TO_LUM[1]) + (rgb.data[i].z * WEIGHTS_RGB_TO_LUM[2]);
    }

    return luminance;
}




/// <summary>
/// Compute X and Y gradients of an image.
/// </summary>
/// <param name="H">H = log luminance</param>
/// <returns>grad H</returns>
ImageGradient getGradients(const ImageFloat& H)
{
    // Empty fields for X and Y gradients with the same sze as the image.
    auto grad_x = ImageFloat(H.width, H.height);
    auto grad_y = ImageFloat(H.width, H.height);

    for (auto y = 0; y < H.height; y++) {
        for (auto x = 0; x < H.width; x++) {
            // Compute X and Y gradients using right-sided forward differences:
            //      H = grad I = (I(x+1,y) - I(x, y), I(x,y+1) - I(x, y)
            // Assume Neumann boundary conditions (ie. gradient is zero when part of the formula falls out of the image size):
            //          => I(x+1,y) - I(x, y) is zero if either of the two pixels is outside of the image.
            // See the grad H equation of Section 5 in the paper for details.
    
            /*******
             * TODO: YOUR CODE GOES HERE!!!
             ******/


            grad_x.data[getImageOffset(grad_x, x, y)] = 0.0f; // TODO: Change this!
            grad_y.data[getImageOffset(grad_y, x, y)] = 0.0f; // TODO: Change this!
        }
    }

    // Return both gradients in an ImageGradient struct.
    return ImageGradient(grad_x, grad_y);
}

/// <summary>
/// Computes gradient attenuation using the formula for phi_k in Sec. 4 of the paper.
/// </summary>
/// <param name="grad_H">gradients of H</param>
/// <param name="alpha_rel">alpha relative to mean grad norm</param>
/// <param name="beta">beta</param>
/// <returns>attenuation coefficients phi</returns>
ImageFloat getGradientAttenuation(const ImageGradient& grad_H, const float alpha_rel = 0.1f, const float beta = 0.35f)
{
    // EPSILON to add to the gradient norm to prevent division by zero.
    const float EPSILON = 1e-3f;
    // An empty gradient attenuation map phi.
    auto phi = ImageFloat(grad_H.x.width, grad_H.x.height);

    // Compute gradient attenuation using the formula for phi_k in Sec. 4 of the paper.
    // Step 1: Compute L2 norms of each XY gradient as grad_norm = sqrt(dx**2+dy**2)
    
    // Step 2: Compute mean norm of all gradients.
    float mean_grad = 0.0f;
    
    // Step 3: Compute alpha = alpha_rel * mean_grad
    float alpha = alpha_rel * mean_grad;
    
    // Step 4: Fill gradient attenuation field phi_k:
    //  phi_k = alpha / (grad_norm + eps) * ((grad_norm + eps) / alpha)^beta
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/

    return phi;
}

/// <summary>
/// Computes attenauted divergence div_G from attenuated gradients H as described in Sec. 5 of the paper.
/// </summary>
/// <param name="grad_H">(original) attenuated gradients of H</param>
/// <param name="phi">gradient attenuations phi</param>
/// <returns>div G</returns>
ImageFloat getAttenuatedDivergence(ImageGradient& grad_H, const ImageFloat& phi) {

    // An empty divergence field with the same size as the input.
    auto div_G = ImageFloat(phi.width, phi.height);
    // Compute attenauted divergence div_G from attenuated gradients H as described in Sec. 5 of the paper.
    // 1. Compute attentuated gradients G:
    //     G = H * phi
    // 2. Compute divergence of G using formula in Sec. 5 of the paper:
    //    div G = (G_x(x,y) - G_x(x-1,y)) + (G_y(x,y) - G_y(x,y-1))
    //
    // Use Neumann boundary conditions (like in getGradients) 
    //  => (G_x(x,y) - G_x(x-1,y)) is zero if either of the two pixels is outside of the image.
    
    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/


    return div_G;
}

/// <summary>
/// Solves poisson equation in form grad^2 I = div G.
/// </summary>
/// <param name="divergence_G">div G</param>
/// <param name="num_iters">number of iterations</param>
/// <returns>luminance I</returns>
ImageFloat solvePoisson(const ImageFloat& divergence_G, const int num_iters = 2000)
{
    // Empty solution.
    auto I = ImageFloat(divergence_G.width, divergence_G.height);
    std::fill(I.data.begin(), I.data.end(), 0.0f);

    // Another solution for the alteranting updates.
    auto I_next = ImageFloat(divergence_G.width, divergence_G.height);

    // Iterative solver.
    for (auto iter = 0; iter < num_iters; iter++)
    {
        if (iter % 500 == 0) {
            // Print progress info every 500 iteartions.
            std::cout << "[" << iter << "/" << num_iters << "] Solving Poisson equation..." << std::endl;
        }

        // Implement one step of the iterative Euler solver:
        //      I_next = ((I[x-1, y] + I[x+1, y] + I[x, y-1] + I[x, y+1]) - div_G[x,y]) / 4
        
        // For I[i,j] values outside of the image bounds, clamp the coordinates to the nearest valid pixel:
        //     Eg:  I(-1,2) => I(0,2)

        // Note: Parallelize the code using OpenMP directives for full points.
    
        /*******
         * TODO: YOUR CODE GOES HERE!!!
         ******/

        // Swaps the current and next solution so that the next iteration
        // uses the new solution as input and the previous solution as output.
        std::swap(I, I_next);
    }

    // After the last "swap", I is the latest solution.
    return I;
}

ImageRGB rescaleRgbByLuminance(const ImageRGB& original_rgb, const ImageFloat& original_luminance, const ImageFloat& new_luminance, const float saturation = 0.5f)
{
    // EPSILON for thresholding the divisior.
    const float EPSILON = 1e-7f;
    // An empty RGB image for the result.
    auto result = ImageRGB(original_rgb.width, original_rgb.height);

    // Return the original_rgb rescaled to match the new luminance as in Sec. 5 of the paper:
    //
    //      result = (original_rgb / max(original_luminance, epsilon))^saturation * new_luminance

    /*******
     * TODO: YOUR CODE GOES HERE!!!
     ******/


    return result;
}