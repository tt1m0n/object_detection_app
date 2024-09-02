#include "OpenVINOFramePreprocessor.hpp"

OpenVINOFramePreprocessor::OpenVINOFramePreprocessor(float scale, cv::Size input_size, cv::Scalar mean, bool swap_rb)
    : scale_(scale), input_size_(input_size), mean_(mean), swap_rb_(swap_rb)
{
}

OpenVINOFramePreprocessor::~OpenVINOFramePreprocessor()
{
}

ov::Tensor OpenVINOFramePreprocessor::run(const cv::Mat& frame)
{
    std::cout << "GERE";
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
    
    // Create OpenVINO Tensor
    ov::Shape tensor_shape = {1, 3, static_cast<size_t>(input_size_.height), static_cast<size_t>(input_size_.width)};
    ov::element::Type tensor_type = ov::element::f32;
    
    // Create tensor from preprocessed data
    ov::Tensor tensor = ov::Tensor(tensor_type, tensor_shape, float_image.data);

    return tensor;
}