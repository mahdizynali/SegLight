#include "inferencer.hpp"
#include <iomanip>
#include <sstream>
#include <algorithm> // std::clamp
#include <cmath>

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
    auto pred = cppflow::arg_max(output, 3);        // [1, H, W]
    auto data = pred.get_data<int64_t>();          // length = H*W

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

/**
 * Compute 1D signals from model output:
 *  - field_bottom_norm[x]: normalized y of lowest FIELD pixel in column x (0=top,1=bottom)
 *  - line_y_norm[x]: normalized mean y of LINE pixels in column x
 *  - line_presence[x]: 1 if line present in that column, else 0
 */
SignalRepresentation SegLightModel::computeSignal(
    const cppflow::tensor& output,
    const cv::Size& size) const
{
    const int width  = size.width;
    const int height = size.height;

    // output: [1, H, W, C], argmax over channels -> [1, H, W]
    auto pred = cppflow::arg_max(output, 3);
    auto data = pred.get_data<int64_t>();   // length = H * W

    // Adjust if your class IDs differ
    constexpr int CLASS_BG    = 0;
    constexpr int CLASS_FIELD = 1;
    constexpr int CLASS_LINE  = 2;

    SignalRepresentation sig;
    sig.field_bottom_norm.resize(width, 0.0f);
    sig.line_y_norm.resize(width, 0.0f);
    sig.line_presence.resize(width, 0.0f);

    // data is row-major: idx = y * width + x
    for (int x = 0; x < width; ++x) {
        int field_bottom_y = -1;

        int line_sum_y = 0;
        int line_count = 0;

        // Scan from bottom to top for this column
        for (int y = height - 1; y >= 0; --y) {
            int idx = y * width + x;
            int cls = static_cast<int>(data[idx]);

            if (field_bottom_y == -1 && cls == CLASS_FIELD) {
                field_bottom_y = y;
            }

            if (cls == CLASS_LINE) {
                line_sum_y += y;
                ++line_count;
            }
        }

        // Normalize field bottom (0=top, 1=bottom)
        if (field_bottom_y >= 0) {
            sig.field_bottom_norm[x] =
                static_cast<float>(field_bottom_y) / static_cast<float>(height - 1);
        } else {
            sig.field_bottom_norm[x] = 0.0f;  // or some sentinel
        }

        // Line presence + normalized mean y
        if (line_count > 0) {
            float mean_y = static_cast<float>(line_sum_y) / static_cast<float>(line_count);
            sig.line_y_norm[x] = mean_y / static_cast<float>(height - 1);
            sig.line_presence[x] = 1.0f;
        } else {
            sig.line_y_norm[x] = 0.0f;
            sig.line_presence[x] = 0.0f;
        }
    }

    return sig;
}

/**
 * Visualize the 1D signals as a simple 2D plot:
 *  - Green dots: field_bottom_norm
 *  - Blue dots: line_y_norm where line_presence > 0
 */
cv::Mat SegLightModel::visualizeSignal(const SignalRepresentation& sig,
                                       int height) const
{
    const int width = static_cast<int>(sig.field_bottom_norm.size());
    if (width == 0) {
        return cv::Mat();
    }

    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int x = 0; x < width; ++x) {
        // y = 0 at top, y = height-1 at bottom
        int y_field = static_cast<int>(sig.field_bottom_norm[x] * (height - 1));
        y_field = std::clamp(y_field, 0, height - 1);

        // field bottom in green
        cv::circle(plot, cv::Point(x, y_field), 1, cv::Scalar(0, 255, 0), -1);

        // line position in blue if present
        if (sig.line_presence[x] > 0.5f) {
            int y_line = static_cast<int>(sig.line_y_norm[x] * (height - 1));
            y_line = std::clamp(y_line, 0, height - 1);
            cv::circle(plot, cv::Point(x, y_line), 1, cv::Scalar(255, 0, 0), -1);
        }
    }

    cv::putText(plot, "Green: field bottom  Blue: line",
                cv::Point(5, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(255, 255, 255), 1);

    return plot;
}

/**
 * Visualize 1D Fourier magnitude of a 1D signal (e.g. field_bottom_norm).
 */
cv::Mat SegLightModel::visualizeSignalFFT(const std::vector<float>& sig,
                                          int height) const
{
    const int N = static_cast<int>(sig.size());
    if (N == 0) {
        return cv::Mat();
    }

    // Create a 1xN float Mat from the signal
    cv::Mat src(1, N, CV_32F);
    for (int i = 0; i < N; ++i) {
        src.at<float>(0, i) = sig[i];
    }

    // Compute DFT: complex output (2 channels: real, imag)
    cv::Mat complexI;
    cv::dft(src, complexI, cv::DFT_COMPLEX_OUTPUT);

    // Split into real and imag parts
    std::vector<cv::Mat> planes(2);
    cv::split(complexI, planes);

    // Magnitude spectrum: 1xN
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);

    // Use only first N/2 (positive frequencies)
    int nFreq = N / 2;
    if (nFreq < 2) nFreq = N;

    // Find max magnitude to normalize
    double minVal, maxVal;
    cv::minMaxLoc(mag, &minVal, &maxVal);
    if (maxVal <= 0.0) maxVal = 1.0;

    // Create plot image: width = nFreq, height = height
    cv::Mat plot(height, nFreq, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int x = 0; x < nFreq; ++x) {
        float v = mag.at<float>(0, x) / static_cast<float>(maxVal); // [0..1]
        v = std::clamp(v, 0.0f, 1.0f);

        // y = height-1 at v=0, y=0 at v=1
        int y = static_cast<int>((1.0f - v) * (height - 1));
        y = std::clamp(y, 0, height - 1);

        cv::line(plot,
                 cv::Point(x, height - 1),
                 cv::Point(x, y),
                 cv::Scalar(0, 255, 255), 1); // yellow-ish bars
    }

    cv::putText(plot, "FFT magnitude (field_bottom)",
                cv::Point(5, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(255, 255, 255), 1);

    return plot;
}

/**
 * 2D FFT magnitude spectrum of an image (dog-style):
 * - Converts to grayscale
 * - Computes 2D DFT
 * - Shifts low frequencies to the center
 * - Uses log magnitude and normalizes to 0..255
 */
cv::Mat SegLightModel::visualizeImageFFT2D(const cv::Mat& src) const
{
    if (src.empty()) return cv::Mat();

    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    gray.convertTo(gray, CV_32F);

    // Make complex image (real=gray, imag=0)
    cv::Mat planes[] = {gray, cv::Mat::zeros(gray.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // 2D DFT
    cv::dft(complexI, complexI);

    // Shift the quadrants of the Fourier image (like np.fft.fftshift)
    int cx = complexI.cols / 2;
    int cy = complexI.rows / 2;

    cv::Mat q0(complexI, cv::Rect(0, 0, cx, cy));   // Top-Left
    cv::Mat q1(complexI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(complexI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(complexI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    // Compute magnitude: sqrt(re^2 + im^2)
    cv::split(complexI, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);

    // Switch to log scale
    mag += 1.0;
    cv::log(mag, mag);

    // Normalize for display
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    mag.convertTo(mag, CV_8U);

    return mag;
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

            SignalRepresentation sig = computeSignal(output, image.size());
            cv::Mat signal_plot = visualizeSignal(sig, 120);
            cv::Mat fft1d_plot = visualizeSignalFFT(sig.field_bottom_norm, 120);
            cv::Mat fft2d_plot = visualizeImageFFT2D(image);

            cv::Mat vis = createVisualization(image, mask, "Press Q/ESC to exit");

            cv::imshow("SegLight Inference", vis);
            cv::imshow("SegLight Signal", signal_plot);
            cv::imshow("SegLight Signal FFT (1D)", fft1d_plot);
            cv::imshow("SegLight Image FFT2D", fft2d_plot);

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

    cv::Mat frame;

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


        SignalRepresentation sig = computeSignal(output, frame.size());
        cv::Mat signal_plot = visualizeSignal(sig, 120);
        cv::Mat fft1d_plot = visualizeSignalFFT(sig.field_bottom_norm, 120);
        cv::Mat fft2d_plot = visualizeImageFFT2D(frame);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << fps << " FPS | "
            << std::setprecision(2) << ms << " ms";

        cv::Mat vis = createVisualization(frame, mask, oss.str());
        cv::imshow("SegLight Camera", vis);
        cv::imshow("SegLight Signal", signal_plot);
        cv::imshow("SegLight Signal FFT (1D)", fft1d_plot);
        cv::imshow("SegLight Image FFT2D", fft2d_plot);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
}
