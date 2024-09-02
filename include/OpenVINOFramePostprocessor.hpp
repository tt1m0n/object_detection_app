#ifndef OPENVINOFRAMEPOSTPROCESSOR_HPP_
#define OPENVINOFRAMEPOSTPROCESSOR_HPP_

#include "IFramePostprocessor.hpp"

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

class OpenVINOFramePostprocessor : public IFramePostprocessor<cv::Mat, ov::Tensor>
{
    public:
        // OpenCVFramePostprocessor(float conf_threshold, float nms_threshold);
        // OpenCVFramePostprocessor(float conf_threshold,
        //     float nms_threshold,
        //     const std::string& model_classes_path,
        //     const std::string& model_class_colors_path);

        OpenVINOFramePostprocessor();

        /**
         * @brief Runs the post-processing step on the input frame and information.
         * 
         * This function scans through the bounding boxes output and keeps only those with high confidence scores.
         * It applies non-maximum suppression to remove overlapping bounding boxes.
         * Finally, it draws the bounding boxes and adds object class information to the frame.
         * 
         * @param frame The input frame to be processed.
         * @param info The vector of information containing bounding box data.
         */
        void run(cv::Mat& frame, const ov::Tensor& info) override;

        ~OpenVINOFramePostprocessor();

    private:
        // float threshold_;
};

#endif // OPENVINOFRAMEPOSTPROCESSOR_HPP_