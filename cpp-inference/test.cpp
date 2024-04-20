#include <map>
#include <array>
#include <vector>
#include <iostream>
#include <cppflow/cppflow.h>
#include <opencv2/opencv.hpp>

cv::TickMeter timer;
cppflow::model *SegLight;

int height = 240;
int width = 320;

int main() {
    std::map<int, std::array<uint8_t, 3>> COLOR_MAP = {
        {0, {0, 0, 0}},   // Black
        {1, {0, 255, 0}}, // Green
        {2, {255, 255, 255}} // White
    };

    std::vector<cv::Vec3b> color_lookup_bgr;
    for (const auto& [class_idx, color] : COLOR_MAP) {
        color_lookup_bgr.push_back(cv::Vec3b(color[0], color[1], color[2]));
    }

    SegLight = new cppflow::model("../model/zeinali");



    cppflow::tensor input = cppflow::decode_png(cppflow::read_file(std::string("../../dataset/images/9.png")));

    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    input = input / 255.f;



    // average all timer and timer times
    int num_iterations = 100; // Configurable number of iterations

    double total_time_zeinali = 0.0;

    // Just call the model to load it into memory
    cppflow::tensor output = (*SegLight)(input);

    for (int i = 0; i < num_iterations; i++) {
        cv::TickMeter timer;

        timer.start();
        cppflow::tensor output = (*SegLight)(input);
        timer.stop();
        std::cout << "Inference time zeinali, ms: " << timer.getTimeMilli()  << std::endl;
        total_time_zeinali += timer.getTimeMilli();
    }

    double average_time_zeinali = total_time_zeinali / num_iterations;

    std::cout << "Average inference time zeinali, ms: " << average_time_zeinali << std::endl;

    cppflow::tensor pred = cppflow::arg_max(output, 3);

    cv::Mat colorized_output(height, width, CV_8UC3);

    std::vector<int64_t> output_data = pred.get_data<int64_t>();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int class_idx = output_data[(y * width) + x];
            cv::Vec3b bgr_color = color_lookup_bgr[class_idx];
            colorized_output.at<cv::Vec3b>(y, x) = bgr_color;
        }
    }

    cv::imshow("Colorized Output", colorized_output);
    cv::waitKey(0);

    return 0;
}
