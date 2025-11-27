#ifndef INFERENCER_HPP
#define INFERENCER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <array>
#include <Eigen/Dense>
#include "cppflow/cppflow.h"
#include <opencv2/opencv.hpp>

// 1D signal representation extracted from the segmentation
struct SignalRepresentation {
    std::vector<float> field_bottom_norm;  // per column, [0..1] (0=top, 1=bottom)
    std::vector<float> line_y_norm;        // per column, [0..1], 0 if no line
    std::vector<float> line_presence;      // 0 or 1 per column (line exists)
};

class SegLightModel {
public:
    explicit SegLightModel(const std::string& model_path);
    void inferenceOnImage(const std::string& image_path);
    void inferenceOnCamera(int cam_num = 0);

private:
    cppflow::model model;
    
    static constexpr std::array<std::array<uint8_t, 3>, 3> COLOR_LUT = {{
        {0,   0,   0},   // background
        {0, 255,   0},   // field / grass
        {255, 255, 255}  // line
    }};
    
    static constexpr int INPUT_WIDTH  = 320;
    static constexpr int INPUT_HEIGHT = 240;
    
    cppflow::tensor preprocessImage(const cv::Mat& image, bool resize = true) const;
    cv::Mat postprocessOutput(const cppflow::tensor& output, const cv::Size& size) const;
    cv::Mat createVisualization(const cv::Mat& input, const cv::Mat& mask,
                                const std::string& info = "") const;

    SignalRepresentation computeSignal(const cppflow::tensor& output,
                                       const cv::Size& size) const;
    cv::Mat visualizeSignal(const SignalRepresentation& sig,
                            int height = 120) const;
    cv::Mat visualizeSignalFFT(const std::vector<float>& sig,
                               int height = 120) const;
    cv::Mat visualizeImageFFT2D(const cv::Mat& src) const;
};

#endif // INFERENCER_HPP
