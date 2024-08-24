#include <thread>
#include "OpenCVFrameCollector.hpp"

OpenCVFrameCollector::OpenCVFrameCollector(FrameQueue<cv::Mat>& frame_queue, const std::string& video_path)
    : IFrameCollector<cv::Mat>(frame_queue, video_path), done_(false){
        video_capture_.open(video_path);
}

void OpenCVFrameCollector::run() {
    int count;
    cv::Mat frame;
    
    // read frames from videos
    int counter = 0;
    // to increase the speed of the video processing. Now I am didn't reseach of more sophisticated methods
    // to inference the video frames more efficiently (OpeenVino, CUDA, GPU, etc)
    const int KEveryNthFrame = 3;
    while(video_capture_.read(frame))
    {
        if (counter++ % KEveryNthFrame == 0) {
            frame_queue_.push(frame.clone());
        }
    }

    done_ = true;
}

bool OpenCVFrameCollector::is_done() const {
    return done_.load();
}

void OpenCVFrameCollector::reset_done_flag() {
    done_ = false;
}