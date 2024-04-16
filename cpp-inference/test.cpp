#include <map>
#include <array>
#include <vector>
#include <iostream>
#include <cppflow/cppflow.h>
#include <opencv2/opencv.hpp>

cv::TickMeter timer;

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

    cppflow::model model("/home/mahdi/Desktop/hslSegment/SegLight/test/model");

    cppflow::tensor input = cppflow::decode_png(cppflow::read_file(std::string("/home/mahdi/Desktop/hslSegment/SegLight/test/9.png")));

    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    input = input / 255.f;

    timer.start();
    cppflow::tensor output = model(input);
    timer.stop();
    std::cout << "Inference time, ms: " << timer.getTimeMilli()  << std::endl;

    cppflow::tensor pred = cppflow::arg_max(output, 3);

    int height = 240;
    int width = 320;

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
