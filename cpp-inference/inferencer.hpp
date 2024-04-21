#ifndef INFERENCER_HPP
#define INFERENCER_HPP

class SegLightModel {
public:
    SegLightModel(const std::string& model_path) {
        SegLight = new cppflow::model(model_path);
    }

    ~SegLightModel() {
        delete SegLight;
    }

    void runInference(const std::string& image_path) {

        loadColorMap();

        // Load input image
        input_image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (input_image.empty()) {
            std::cerr << "Failed to read input image." << std::endl;
            return;
        }

        // Preprocess input image
        cppflow::tensor input = cppflow::decode_png(cppflow::read_file(image_path));
        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        input = cppflow::expand_dims(input, 0);
        input = input / 255.f;

        // average all timer and timer times
        int num_iterations = 100; // Configurable number of iterations

        double total_time = 0.0;

        // Just call the model to load it into memory
        result = (*SegLight)(input);

        for (int i = 0; i < num_iterations; i++) {
            cv::TickMeter timer;

            timer.start();
            result = (*SegLight)(input);
            timer.stop();
            std::cout << "Inference time zeinali, ms: " << timer.getTimeMilli()  << std::endl;
            total_time += timer.getTimeMilli();
        }

        double average_time_zeinali = total_time / num_iterations;

        std::cout << "Average inference time zeinali, ms: " << average_time_zeinali << std::endl;

    }

    void DisplayOutput() {

        // Post-process result
        cppflow::tensor pred = cppflow::arg_max(result, 3);
        std::vector<int64_t> output_data = pred.get_data<int64_t>();

        cv::Mat colorized_output(input_image.size(), CV_8UC3);
        for (int y = 0; y < input_image.rows; ++y) {
            for (int x = 0; x < input_image.cols; ++x) {
                int class_idx = output_data[(y * input_image.cols) + x];
                cv::Vec3b bgr_color = color_lookup_bgr[class_idx];
                colorized_output.at<cv::Vec3b>(y, x) = bgr_color;
            }
        }

        while (1) {
            cv::imshow("input", input_image);
            cv::imshow("predicted", colorized_output);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q')
                break;
        }
    }

private:
    cppflow::model *SegLight;
    cppflow::tensor result;
    cv::Mat input_image;
    std::vector<cv::Vec3b> color_lookup_bgr;

    void loadColorMap() {
        color_lookup_bgr = {
            {0, 0, 0},     // Black
            {0, 255, 0},   // Green
            {255, 255, 255} // White
        };
    }
};

#endif // INFERENCER_HPP