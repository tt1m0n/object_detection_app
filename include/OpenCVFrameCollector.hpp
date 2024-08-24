#ifndef OPECVFRAMECOLLECTOR_HPP_
#define OPECVFRAMECOLLECTOR_HPP_

#include <atomic>
#include <string>
#include <opencv2/opencv.hpp>

#include "IFrameCollector.hpp"

class OpenCVFrameCollector : public IFrameCollector<cv::Mat> 
{
    public: 
        OpenCVFrameCollector(FrameQueue<cv::Mat>& frame_queue, const std::string& video_path);
        void run() override;
        bool is_done() const override;
        void reset_done_flag();
    
    private:
        cv::VideoCapture video_capture_;
        std::atomic<bool> done_;
};

#endif // OPECVFRAMECOLLECTOR_HPP_
