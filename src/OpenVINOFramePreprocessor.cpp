#include "OpenVINOFramePreprocessor.hpp"

OpenVINOFramePreprocessor::OpenVINOFramePreprocessor(float scale, cv::Size input_size, cv::Scalar mean, bool swap_rb)
    : scale_(scale), input_size_(input_size), mean_(mean), swap_rb_(swap_rb)
{
}

OpenVINOFramePreprocessor::~OpenVINOFramePreprocessor()
{
}

InferenceEngine::Blob::Ptr OpenVINOFramePreprocessor::run(const cv::Mat& frame)
{
    cv::Mat resized_image;
    cv::resize(frame, resized_image, input_size_);
    
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32F, scale_);
    
    // Convert to BGR if needed (OpenVINO expects BGR by default)
    if (swap_rb_) {
        cv::cvtColor(float_image, float_image, cv::COLOR_RGB2BGR);
    }

    // Subtract mean
    cv::Mat mean_image(input_size_, float_image.type(), mean_);
    cv::subtract(float_image, mean_image, float_image);
    
    // Create OpenVINO Blob
    InferenceEngine::TensorDesc tensor_desc(InferenceEngine::Precision::FP32,
                                            {1, 3, input_size_.height, input_size_.width},
                                            InferenceEngine::Layout::NCHW);

    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<float>(
        tensor_desc, (float*)float_image.data);

    return blob;
}