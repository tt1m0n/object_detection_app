#include "OpenVINOFramePostprocessor.hpp"

OpenVINOFramePostprocessor::OpenVINOFramePostprocessor() {}

OpenVINOFramePostprocessor::~OpenVINOFramePostprocessor() {}

void OpenVINOFramePostprocessor::run(cv::Mat& frame, const ov::Tensor& info)
{
    std::cout << "OpenVINOFramePostprocessor::run" << std::endl;
}