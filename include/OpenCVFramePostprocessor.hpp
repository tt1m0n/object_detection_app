#ifndef OPENCVFRAMEPOSTPROCESSOR_HPP_
#define OPENCVFRAMEPOSTPROCESSOR_HPP_

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "IFramePostprocessor.hpp"

class OpenCVFramePostprocessor : public IFramePostprocessor<cv::Mat, std::vector<cv::Mat>>
{
    public:
        OpenCVFramePostprocessor(float conf_threshold, float nms_threshold);
        OpenCVFramePostprocessor(float conf_threshold,
            float nms_threshold,
            const std::string& model_classes_path,
            const std::string& model_class_colors_path);
        ~OpenCVFramePostprocessor();

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
        void run(cv::Mat& frame, const std::vector<cv::Mat>& info) override;
    
    private:
        void setClasses(const std::string& model_classes_path);
        void setCollors(const std::string& model_class_colors_path);
        void drawBoundingBoxToFrame(int class_id, int left, int top, int right, int bottom, cv::Mat& frame) const;
        void addObjectClassToFrame(int class_id, float conf, int left, int top, cv::Mat& frame) const;

    private:
        std::vector<std::string> classes_;
        std::vector<std::vector<uint8_t>> colors_;
        float conf_threshold_;
        float nms_threshold_;
};

#endif // OPENCVFRAMEPOSTPROCESSOR_HPP_