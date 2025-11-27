#include "inferencer.hpp"

int main(int argc, char** argv) {
    const std::string model_path = "../models";
    
    SegLightModel model(model_path);
    
    if (argc > 1) {
        model.inferenceOnImage(argv[1]);
    } else {
        model.inferenceOnCamera(0);
    }
    
    return 0;
}
