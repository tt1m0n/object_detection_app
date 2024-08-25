#ifndef __OPENCVFRAMEPREPROCESSOR_HPP__
#define __OPENCVFRAMEPREPROCESSOR_HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "IFramePreprocessor.hpp"

class OpenCVFramePreprocessor : public IFramePreprocessor<cv::Mat, cv::Mat>
{
    public:
        OpenCVFramePreprocessor(float scale, cv::Size input_size, cv::Scalar mean, bool swap_rb);
        cv::Mat run(const cv::Mat& frame) override;
        ~OpenCVFramePreprocessor();
    
    private:
        float scale_;
        cv::Size input_size_;
        cv::Scalar mean_;
        bool swap_rb_;
};

#endif // __OPENCVFRAMEPREPROCESSOR_HPP__