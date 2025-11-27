#ifndef INFERENCER_HPP
#define INFERENCER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "cppflow/cppflow.h"
#include <opencv2/opencv.hpp>

class SegLightModel {
    public:
        explicit SegLightModel(const std::string& model_path);
        void inferenceOnImage(const std::string& image_path);
        void inferenceOnCamera(int cam_num = 0);

    private:
        cppflow::model model;
        
        static constexpr std::array<std::array<uint8_t, 3>, 3> COLOR_LUT = {{
            {0, 0, 0},
            {0, 255, 0},
            {255, 255, 255}
        }};
        
        static constexpr int INPUT_WIDTH = 320;
        static constexpr int INPUT_HEIGHT = 240;
        
        cppflow::tensor preprocessImage(const cv::Mat& image, bool resize = true) const;
        cv::Mat postprocessOutput(const cppflow::tensor& output, const cv::Size& size) const;
        cv::Mat createVisualization(const cv::Mat& input, const cv::Mat& mask, const std::string& info = "") const;
};

#endif
