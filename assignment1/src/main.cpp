#include "your_code_here.h"

#include <glm/vec3.hpp>
//#include <glm/detail/func_vector_relational.hpp>

static const std::filesystem::path dataDirPath { DATA_DIR };
static const std::filesystem::path outDirPath { OUTPUT_DIR };

int test() {

    for (const auto& entry : std::filesystem::directory_iterator(outDirPath)) {

        auto filename = entry.path().stem().string() + ".png";

        auto expected = ImageRGB(dataDirPath / "expected-outputs" / filename);
        auto results = ImageRGB(outDirPath / filename);

        auto exp_num_pixels = expected.width * expected.height;
        auto res_num_pixels = results.width * results.height;

        if (exp_num_pixels != res_num_pixels) {
            std::cout << "Size mismatch: " + filename << std::endl;
            continue;
        }

        for (int i = 0; i < exp_num_pixels; i++) {

            if (!glm::all(glm::equal(expected.data[i], results.data[i]))) {
                std::cout << "Validation Failed: " + filename << std::endl;
                break;
            }
        }


    }
    
    return 0;
}

/// <summary>
/// Main method. Runs default tests. Feel free to modify it, add more tests and experiments,
/// change the input images etc. The file is not part of the solution. All solutions have to 
/// implemented in "your_code_here.h".
/// </summary>
/// <returns>0</returns>
int main()
{
    std::chrono::steady_clock::time_point time_start, time_end;
    printOpenMPStatus();
    
    // 0. Load inputs from files. https://www.cs.huji.ac.il/~danix/hdr/pages/memorial.html
    auto image = ImageRGB(dataDirPath / "memorial2_half.hdr");
    image.writeToFile(outDirPath / "0_src.png");

    // 1. Normalize the image range to [0,1].
    auto image_normed = normalizeRGBImage(image);
    image_normed.writeToFile(outDirPath / "1_normalized.png");

    // 2. Apply gamma curve.
    auto image_gamma = applyGamma(image_normed, 1 / 2.2f);
    image_gamma.writeToFile(outDirPath / "2_gamma.png");

    // 2b. Apply gamma to the original image.
    auto gamma_orig = applyGamma(image, 1 / 2.2f);
    gamma_orig.writeToFile(outDirPath / "2_gamma_orig.png");

    // 3. Get luminance.
    auto luminance = rgbToLuminance(image);
    luminance.writeToFile(outDirPath / "3a_luminance.png");
    auto H = lnImage(luminance);
    H.writeToFile(outDirPath / "3b_log_luminance_H.png");

    //// 4. Compute luminance gradients \nabla H (Sec. 5).
    auto gradients = getGradients(H);
    gradientsToRgb(gradients).writeToFile(outDirPath / "4_gradients_H.png");

    // 5. Compute the gradient attenuation \phi (Sec. 4).
    auto grad_atten = getGradientAttenuation(gradients);
    grad_atten.writeToFile(outDirPath / "5_attenuation_phi.png");

    // 6. Compute the attentuated divergence (Sec. 3 and 5).
    auto divergence = getAttenuatedDivergence(gradients, grad_atten);
    imageRgbToFloat(normalizeRGBImage(imageFloatToRgb(divergence))).writeToFile(outDirPath / "6_divergence_G.png");

    // 7. Very simplistic single-scale direct solver of the Poisson equation (Eq. 3).
    time_start = std::chrono::steady_clock::now();
    auto solved_luminance = solvePoisson(divergence);
    time_end = std::chrono::steady_clock::now();
    std::cout << "Poisson Solver | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;

    // Normalize the result to [0,1] range.
    // Note: We are using our code for RGB images here which is inefficient.
    solved_luminance = imageRgbToFloat(normalizeRGBImage(imageFloatToRgb(solved_luminance)));
    solved_luminance.writeToFile(outDirPath / "7_poisson_solution_I.png");

    // 8. Convert back to RGB.
    auto result_rgb = rescaleRgbByLuminance(image, luminance, solved_luminance);
    result_rgb.writeToFile(outDirPath / "8_result_rgb.png");

    // Custom validations
    test();

    std::cout << "All done!" << std::endl;
    return 0;
}
