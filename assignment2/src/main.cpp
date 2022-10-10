#include "your_code_here.h"
#include <glm/gtc/epsilon.hpp>

static const std::filesystem::path dataDirPath { DATA_DIR };
static const std::filesystem::path outDirPath { OUTPUT_DIR };

/// <summary>
/// Enum of the provided scenes.
/// </summary>
enum InputSelection : int {
    Mini = 0,
    Middlebury = 1,
};

/// <summary>
/// Feel free to edit these params or add new.
/// </summary>
static const SceneParams SceneParams_Mini = {
    4, 0, 0, 2,
    2.0f, 1.0f, 4.0f, 2.0f, 6.0f,
    2, 0.02f, 50.0f, 0.95f, 1.0f,
};
static const SceneParams SceneParams_Middlebury = {
    120, 50, 0, 30,
    64.0f, 0.25f, 600.0f, 550.0f, 650.0f,
    9, 0.05f, 1.0f, 30.0f, -1.0f,
};

int test(const InputSelection scene_name)
{

    float EPSILON_error;

    switch (scene_name) {
    
        case InputSelection::Mini:
            EPSILON_error = 1e-2f;
            break;
        case InputSelection::Middlebury:
            EPSILON_error = 1e-1f;
            break;
        default:
            throw std::runtime_error("Invalid scene ID.");

    }

    for (const auto& entry : std::filesystem::directory_iterator(outDirPath)) {

        auto filename = entry.path().stem().string() + ".png";

        ImageRGB expected, results;

        if (scene_name == InputSelection::Mini)
            expected = ImageRGB(dataDirPath / "expected-outputs/mini" / filename);
        else
            expected = ImageRGB(dataDirPath / "expected-outputs/half" / filename);

        results = ImageRGB(outDirPath / filename);

        auto exp_num_pixels = expected.width * expected.height;
        auto res_num_pixels = results.width * results.height;

        if (exp_num_pixels != res_num_pixels) {
            std::cout << "Size mismatch: " + filename << std::endl;
            continue;
        }

        for (int i = 0; i < exp_num_pixels; i++) {

            if (!glm::all(glm::epsilonEqual(expected.data[i], results.data[i], EPSILON_error))) {
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
    // Do not add any noise to the saved images.
    std::srand(unsigned(4733668));
    const float im_write_noise_level = 0.0f;

    std::chrono::steady_clock::time_point time_start, time_end;
    printOpenMPStatus();
    
    
    // 0. Load inputs from files. 
    ImageRGB image;
    ImageFloat src_disparity;
    SceneParams scene_params;
     
    // Change your inputs here!
    const auto input_select = InputSelection::Mini;
    // const auto input_select = InputSelection::Middlebury;

    switch (input_select) {
        case InputSelection::Mini:
            // The 5x4 mini image.
            image = ImageRGB(dataDirPath / "mini/image.png");
            // image = ImageRGB(dataDirPath / "moon_1.jpg");
            src_disparity = loadDisparity(dataDirPath / "mini/disparity.png");
            scene_params = SceneParams_Mini;
            break;
        case InputSelection::Middlebury:
            // The Middleburry ArtScene
            // Find plenty more in: https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/Art-2views.zip
            // Note that we use the half-sized images for faster testing.
            image = ImageRGB(dataDirPath / "half/view1.png");
            src_disparity = loadDisparity(dataDirPath / "half/disp1.png");
            scene_params = SceneParams_Middlebury;
            break;
        default:
            throw std::runtime_error("Invalid scene ID.");
    }

    /*auto tmp = sampleBilinear(image, glm::vec2(0.5f, 0.5f));
    auto middle = (image.data[5] + image.data[6] + image.data[9] + image.data[10]) / 4.0f;

    if (!glm::all(glm::equal(tmp, middle))) {
        std::cout << "Validation Failed: " << std::endl;
    }*/


    // Test save the inputs.
    image.writeToFile(outDirPath / "src_image.png", 1.0f, im_write_noise_level);
    disparityToColor(src_disparity, scene_params.in_disp_min, scene_params.in_disp_max).writeToFile(outDirPath / "src_disparity.png");


    // 1. Filter depth map (guided filter). We do this to remove (most of) the holes.
    time_start = std::chrono::steady_clock::now();
    auto disparity_filtered = jointBilateralFilter(src_disparity, image, scene_params.bilateral_radius, scene_params.bilateral_joint_sigma);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[1] Bilateral filter | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    disparityToColor(disparity_filtered, scene_params.in_disp_min, scene_params.in_disp_max).writeToFile(outDirPath / "disparity_filtered.png", 1.0f, im_write_noise_level);


    // 2. Disparity to depth. We need depth to do Z-testing.
    time_start = std::chrono::steady_clock::now();
    auto linear_depth = disparityToNormalizedDepth(disparity_filtered);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[2] Disparity -> Depth | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    linear_depth.writeToFile(outDirPath / "linear_depth.png", 1.0f, im_write_noise_level);

    // 2.1 Forward warp the image. We use the original (unscaled) disparity to test the warping.
    // This has the benefit that we have a GT for this from the dataset.
    // This test will show us why forward warping is not the best fit.
    time_start = std::chrono::steady_clock::now();
    auto warped_forward = fowardWarpImage(image, linear_depth, disparity_filtered, scene_params.warp_scale);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[2.5] Forward warp the image | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    warped_forward.writeToFile(outDirPath / "warped_forward.png", 1.0f, im_write_noise_level);       

    // 3. Create a grid that covers all pixels.
    time_start = std::chrono::steady_clock::now();
    auto src_grid = createWarpingGrid(image.width, image.height);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[3] Create a grid | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    plotGridMesh(src_grid, { image.width * scene_params.grid_viz_im_scale, image.height * scene_params.grid_viz_im_scale }, scene_params.grid_viz_tri_scale).writeToFile(outDirPath / "src_grid.png", 1.0f, im_write_noise_level);   

    // 4. Warp the grid by moving the vertices according to the disparity.
    time_start = std::chrono::steady_clock::now();
    auto dst_grid = warpGrid(src_grid, disparity_filtered, scene_params.warp_scale);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[4] Warp the grid | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    plotGridMesh(dst_grid, { image.width * scene_params.grid_viz_im_scale, image.height * scene_params.grid_viz_im_scale }, scene_params.grid_viz_tri_scale).writeToFile(outDirPath / "dst_grid.png", 1.0f, im_write_noise_level);   

    // 5. Resample the image using the warped grid (= mesh-based warping).
    // Compare the result to the warped_forward.png. Should have no holes.
    time_start = std::chrono::steady_clock::now();
    auto warped_backward = backwardWarpImage(image, linear_depth, src_grid, dst_grid);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[5] Backward warp the image | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    warped_backward.writeToFile(outDirPath / "warped_backward.png", 1.0f, im_write_noise_level);

    
    // 6. Now we convert the depth back to disparity.
    // We do this to have control over the disparity magnitude and position.
    // This allows us to tune a comfortably viewable stereoscopic effect.
    time_start = std::chrono::steady_clock::now();
    auto target_disparity = normalizedDepthToDisparity(
        linear_depth,
        scene_params.iod_mm,
        scene_params.px_size_mm,
        scene_params.screen_distance_mm,
        scene_params.near_plane_mm,
        scene_params.far_plane_mm);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[6] Depth -> Target disparity | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    disparityToColor(target_disparity, scene_params.out_disp_min, scene_params.out_disp_max).writeToFile(outDirPath / "target_disparity.png", 1.0f, im_write_noise_level);

    // 7. Repeat the mesh-based warping twice to obtain left and right stereo pair.
    // We use the disparity scaling warp_factor to obtain mirrored effect for left and right image.
    time_start = std::chrono::steady_clock::now();
    std::vector<ImageRGB> image_pair;
    for (int i = 0; i < 2; i++) {
        // The total disparity is split in half between both images as each gets shifted in an opposite direction.
        auto warp_factor = i == 0 ? -0.5f : 0.5f;

        // Warp the grid. We can reuse src_grid because it does not change.
        auto dst_grid = warpGrid(src_grid, target_disparity, warp_factor);

        // Resample the image.
        ImageRGB dst_image = backwardWarpImage(image, linear_depth, src_grid, dst_grid);
        dst_image.writeToFile(outDirPath / (i == 0 ? "stereo_left.png" : "stereo_right.png"), 1.0f, im_write_noise_level);
        image_pair.push_back(std::move(dst_image));
    }
    time_end = std::chrono::steady_clock::now();
    std::cout << "[7] Create a stereo pair | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;

    // 8. Combine the left/right image pair into an anaglyph stereoscopic image.
    time_start = std::chrono::steady_clock::now();
    auto anaglyph = createAnaglyph(image_pair[0], image_pair[1], 0.3f);
    time_end = std::chrono::steady_clock::now();
    std::cout << "[8] Create anaglyph | Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1e3f
              << " ms" << std::endl;
    anaglyph.writeToFile(outDirPath / "anaglyph.png", 1.0f, im_write_noise_level);

    
    test(input_select);

    std::cout << "All done!" << std::endl;
    return 0;
}
