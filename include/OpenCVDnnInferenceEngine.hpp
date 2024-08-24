#ifndef OPENCV_DNN_INFERENCE_ENGINE_HPP_
#define OPENCV_DNN_INFERENCE_ENGINE_HPP_

#include "IInferenceEngine.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class OpenCVDnnInferenceEngine : public IInferenceEngine<cv::Mat, std::vector<cv::Mat>> {
    public:
        OpenCVDnnInferenceEngine(const std::string& model_config_path, const std::string& model_weight_path);
        void process(cv::Mat& frame, std::vector<cv::Mat>& res) override;
        ~OpenCVDnnInferenceEngine();

    private:
        cv::dnn::Net detector_;
};

#endif // OPENCV_DNN_INFERENCE_ENGINE_HPP_
