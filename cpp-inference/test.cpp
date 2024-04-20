#include <map>
#include <array>
#include <vector>
#include <iostream>
#include <cppflow/cppflow.h>
#include <opencv2/opencv.hpp>

cv::TickMeter timer;
cv::TickMeter timer2;
cppflow::model *SegLight;
cppflow::model *SegLight2;

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

    SegLight = new cppflow::model("../model/khatibi");
    SegLight2 = new cppflow::model("../model/zeinali");


    cv::Mat frame = cv::imread("../../dataset/images/9.png");
    std::vector<unsigned char> img_vec(frame.data, frame.data + width * height * 3);
    cppflow::tensor input(img_vec, {height, width, 3});

    cppflow::tensor input2 = cppflow::decode_png(cppflow::read_file(std::string("../../dataset/images/9.png")));

    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    // input = input / 255.f;

    input2 = cppflow::cast(input2, TF_UINT8, TF_FLOAT);
    input2 = cppflow::expand_dims(input2, 0);
    input2 = input2 / 255.f;



    // average all timer and timer2 times
    int num_iterations = 100; // Configurable number of iterations

    double total_time_khatibi = 0.0;
    double total_time_zeinali = 0.0;

    // Just call the model to load it into memory
    auto output = (*SegLight)({{"input:0", input}}, {"predictor/predictions_argmax:0"});
    cppflow::tensor output2 = (*SegLight2)(input2);

    for (int i = 0; i < num_iterations; i++) {
        cv::TickMeter timer;
        cv::TickMeter timer2;
        timer.start();
        auto output = (*SegLight)({{"input:0", input}}, {"predictor/predictions_argmax:0"});
        timer.stop();
        std::cout << "Inference time khatibi, ms: " << timer.getTimeMilli()  << std::endl;
        total_time_khatibi += timer.getTimeMilli();

        timer2.start();
        cppflow::tensor output2 = (*SegLight2)(input2);
        timer2.stop();
        std::cout << "Inference time zeinali, ms: " << timer2.getTimeMilli()  << std::endl;
        total_time_zeinali += timer2.getTimeMilli();
    }

    double average_time_khatibi = total_time_khatibi / num_iterations;
    double average_time_zeinali = total_time_zeinali / num_iterations;

    std::cout << "Average inference time khatibi, ms: " << average_time_khatibi << std::endl;
    std::cout << "Average inference time zeinali, ms: " << average_time_zeinali << std::endl;

    // timer.start();
    // auto output = (*SegLight)({{"input:0", input}}, {"predictor/predictions_argmax:0"});
    // timer.stop();
    // std::cout << "Inference time khatibi, ms: " << timer.getTimeMilli()  << std::endl;

    // timer2.start();
    // cppflow::tensor output2 = (*SegLight2)(input2);
    // timer2.stop();
    // std::cout << "Inference time zeinali, ms: " << timer2.getTimeMilli()  << std::endl;

    cppflow::tensor pred = cppflow::arg_max(output2, 3);

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
