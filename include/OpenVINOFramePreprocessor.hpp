#ifndef OPENVINOFAMEPREPROCESSOR_HPP_
#define OPENVINOFAMEPREPROCESSOR_HPP_

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

#include "IFramePreprocessor.hpp"

class OpenVINOFramePreprocessor : public IFramePreprocessor<cv::Mat, InferenceEngine::Blob::Ptr>
{
    public:
        OpenVINOFramePreprocessor(float scale, cv::Size input_size, cv::Scalar mean, bool swap_rb);
        InferenceEngine::Blob::Ptr run(const cv::Mat& frame) override;
        ~OpenVINOFramePreprocessor();
    
    private:
        float scale_;
        cv::Size input_size_;
        cv::Scalar mean_;
        bool swap_rb_;
};

#endif // OPENVINOFAMEPREPROCESSOR_HPP_