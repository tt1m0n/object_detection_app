#include "OpenCVFramePreprocessor.hpp"

OpenCVFramePreprocessor::OpenCVFramePreprocessor(float scale, cv::Size input_size, cv::Scalar mean, bool swap_rb)
    : scale_(scale), input_size_(input_size), mean_(mean), swap_rb_(swap_rb)
{
}

OpenCVFramePreprocessor::~OpenCVFramePreprocessor()
{
}

cv::Mat OpenCVFramePreprocessor::run(const cv::Mat& frame)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, scale_, input_size_, mean_, swap_rb_, false);
    return blob;
}
