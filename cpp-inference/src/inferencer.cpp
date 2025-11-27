#include "inferencer.hpp"
#include <iomanip>
#include <sstream>

SegLightModel::SegLightModel(const std::string& model_path) : model(model_path) {
    std::cout << "Model loaded: " << model_path << std::endl;
}

cppflow::tensor SegLightModel::preprocessImage(const cv::Mat& image, bool resize) const {
    cv::Mat tmp;

    if (resize)
        cv::resize(image, tmp, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    else
        tmp = image;

    cv::Mat rgb_float;
    cv::cvtColor(tmp, rgb_float, cv::COLOR_BGR2RGB);
    rgb_float.convertTo(rgb_float, CV_32F, 1.f/255.f);

    size_t count = rgb_float.rows * rgb_float.cols * 3;
    std::vector<float> data(rgb_float.ptr<float>(),
                            rgb_float.ptr<float>() + count);

    return cppflow::tensor(data, {1, tmp.rows, tmp.cols, 3});
}

cv::Mat SegLightModel::postprocessOutput(const cppflow::tensor& output, const cv::Size& size) const {
    auto pred = cppflow::arg_max(output, 3);
    auto data = pred.get_data<int64_t>();

    const int total = size.width * size.height;

    Eigen::Map<Eigen::Array<int64_t, Eigen::Dynamic, 1>> classes(data.data(), total);
    Eigen::Array<int, Eigen::Dynamic, 1> clamped = classes.max(0).min(2).cast<int>();

    cv::Mat result(size, CV_8UC3);
    uint8_t* out = result.ptr<uint8_t>();

    for (int i = 0; i < total; ++i) {
        const auto& c = COLOR_LUT[clamped(i)];
        out[i*3 + 0] = c[0];
        out[i*3 + 1] = c[1];
        out[i*3 + 2] = c[2];
    }

    return result;
}

cv::Mat SegLightModel::createVisualization(const cv::Mat& input, const cv::Mat& mask, 
                                           const std::string& info) const {
    cv::Mat vis;
    cv::hconcat(input, mask, vis);
    
    if (!info.empty()) {
        int baseline;
        auto textSize = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::rectangle(vis, cv::Point(5, 5), 
                     cv::Point(15 + textSize.width, 15 + textSize.height),
                     cv::Scalar(0, 0, 0), -1);
        cv::putText(vis, info, cv::Point(10, 10 + textSize.height), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
    
    return vis;
}

void SegLightModel::inferenceOnImage(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }

    cppflow::tensor input = cppflow::decode_png(cppflow::read_file(image_path));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    input = input / 255.f;

    constexpr int iterations = 100;
    double total_time = 0.0;

    cppflow::tensor output;
    for (int i = 0; i < iterations; ++i) {
        auto start = cv::getTickCount();
        output = model(input);
        auto end = cv::getTickCount();

        double ms = (end - start) * 1000.0 / cv::getTickFrequency();
        total_time += ms;

        if (i == 0) {
            cv::Mat mask = postprocessOutput(output, image.size());
            cv::Mat vis = createVisualization(image, mask, "Press Q/ESC to exit");

            cv::imshow("SegLight Inference", vis);
            while (true) {
                int key = cv::waitKey(1);
                if (key == 'q' || key == 'Q' || key == 27) break;
            }
            cv::destroyAllWindows();
        }
    }

    std::cout << "Average inference time: " << std::fixed << std::setprecision(2)
              << (total_time / iterations) << " ms" << std::endl;
}

void SegLightModel::inferenceOnCamera(int cam_num) {
    cv::VideoCapture cap(cam_num);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera: " << cam_num << std::endl;
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  INPUT_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT);

    std::cout << "Camera inference started (press Q/ESC to quit)" << std::endl;

    cv::Mat frame, rgb, rgb_float;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        if (frame.cols != INPUT_WIDTH || frame.rows != INPUT_HEIGHT) {
            std::cerr << "Warning: camera did not output 320x240. Got "
                      << frame.cols << "x" << frame.rows << std::endl;
        }

        cppflow::tensor input = preprocessImage(frame, false);

        auto t0 = cv::getTickCount();
        auto output = model(input);
        auto t1 = cv::getTickCount();
        double ms = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        double fps = (ms > 0.0) ? 1000.0 / ms : 0.0;

        cv::Mat mask = postprocessOutput(output, frame.size());

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << fps << " FPS | "
            << std::setprecision(2) << ms;

        cv::Mat vis = createVisualization(frame, mask, oss.str());
        cv::imshow("SegLight Camera", vis);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
}
