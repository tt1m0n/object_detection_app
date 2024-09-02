#ifndef OPENVINOFAMEPREPROCESSOR_HPP_
#define OPENVINOFAMEPREPROCESSOR_HPP_

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "IFramePreprocessor.hpp"

class OpenVINOFramePreprocessor : public IFramePreprocessor<cv::Mat, ov::Tensor>
{
    public:
        OpenVINOFramePreprocessor(float scale, cv::Size input_size, cv::Scalar mean, bool swap_rb);
        ov::Tensor run(const cv::Mat& frame) override;
        ~OpenVINOFramePreprocessor();
    
    private:
        float scale_;
        cv::Size input_size_;
        cv::Scalar mean_;
        bool swap_rb_;
};

#endif // OPENVINOFAMEPREPROCESSOR_HPP_