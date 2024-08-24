#ifndef IFRAMECOLLECTOR_HPP_
#define IFRAMECOLLECTOR_HPP_

#include <string>

#include "FrameQueue.hpp"

/**
 * @brief Interface for a frame collector.
 * 
 * This interface defines the common methods and properties for a frame collector.
 * A frame collector is responsible for collecting frames from a video source and
 * adding them to a frame queue.
 * 
 * @tparam FrameType The type of frames being collecte (cv::Mat, cv::cuda::GpuMat, etc).
 */
template <typename FrameType>
class IFrameCollector {
public:
    IFrameCollector(FrameQueue<FrameType>& frame_queue, const std::string video_path)
        : frame_queue_(frame_queue), video_path_(video_path) {};
    virtual void run() = 0;
    // will be used to track the end of the collection frames into the frame queue
    virtual bool is_done() const = 0;
    virtual ~IFrameCollector() {};

protected:
    FrameQueue<FrameType>& frame_queue_;
    std::string video_path_;
};

#endif // IFRAMECOLLECTOR_HPP_