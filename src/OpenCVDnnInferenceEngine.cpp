#include <string>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "OpenCVDnnInferenceEngine.hpp"

OpenCVDnnInferenceEngine::OpenCVDnnInferenceEngine(const std::string& model_config_path,
    const std::string& model_weight_path) {
    detector_ = cv::dnn::readNetFromDarknet(model_config_path, model_weight_path);
    detector_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    detector_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    if(detector_.empty())
    {
        std::cout << "Model Load Error" << std::endl;
        exit(-1);
    }
}

void OpenCVDnnInferenceEngine::process(cv::Mat& frame, std::vector<cv::Mat>& res) {
    detector_.setInput(frame);
    detector_.forward(res, detector_.getUnconnectedOutLayersNames());
}

OpenCVDnnInferenceEngine::~OpenCVDnnInferenceEngine() {}