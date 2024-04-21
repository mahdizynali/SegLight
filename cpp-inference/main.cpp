#include <opencv2/opencv.hpp>
#include <cppflow/cppflow.h>
#include <string>

#include "inferencer.hpp"

int main() {

    std::string Model = "../model/zeinali";
    std::string ImagePath = "../../dataset/images/9.png";

    SegLightModel segLight(Model);

    segLight.runInference(ImagePath);
    segLight.DisplayOutput();

    return 0;
}
