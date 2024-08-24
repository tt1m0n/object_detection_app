#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <memory>
#include <atomic>
#include <utility>
#include <vector>

#include "FrameQueue.hpp"
#include "IFramePreprocessor.hpp"
#include "IInferenceEngine.hpp"
#include "IFramePostprocessor.hpp"
#include "IFrameCollector.hpp"


/**
 * @brief Configuration struct for the ObjectDetector class.
 * 
 * This struct holds the configuration parameters for the ObjectDetector class.
 * It specifies the input and output frame queues, tracker status flags, and the frame collector.
 * 
 * @tparam FrameType The type of frames used by the ObjectDetector.
 */
template <typename FrameType>
struct ObjectDetectorConfig {
        // reference to the input frame queue, used also by the frame collector
        FrameQueue<FrameType>& input_frame_queue_;
        // reference to the output frame queue will be used by the GUI
        FrameQueue<std::pair<FrameType, FrameType>>& out_frames_queue_;
        // to track if detector finish add frames to output queue will be used in the Drawer(GUI)
        std::atomic<bool>& tracker_object_detector_done_;
        // to track if GUI closed, so we do not need to continue processing frames
        std::atomic<bool>& tracker_is_gui_closed_;
        // to check if the frame collector is done collecting frames and we can sop waiting for new frames
        IFrameCollector<FrameType>& frame_collector_;

        ObjectDetectorConfig(FrameQueue<FrameType>& in_queue,
                                                 FrameQueue<std::pair<FrameType, FrameType>>& out_queue,
                                                 std::atomic<bool>& tracker_obj_detector_done,
                                                 std::atomic<bool>& tracker_is_gui_closed,
                                                 IFrameCollector<FrameType>& frame_collector)
                : input_frame_queue_(in_queue),
                    out_frames_queue_(out_queue),
                    tracker_object_detector_done_(tracker_obj_detector_done),
                    tracker_is_gui_closed_(tracker_is_gui_closed),
                    frame_collector_(frame_collector) {}
};

template <typename FrameType, typename InferenceInfoType>
class ObjectDetector
{
    public:
        ObjectDetector(std::unique_ptr<IFramePreprocessor<FrameType>> frame_preprocessor,
                       std::unique_ptr<IInferenceEngine<FrameType, InferenceInfoType>> inference_engine,
                       std::unique_ptr<IFramePostprocessor<FrameType, InferenceInfoType>> frame_post_processor,
                       ObjectDetectorConfig<FrameType>& config);

        /**
         * @brief Runs the object detection process.
         * 
         * This function continuously processes frames from the input frame queue until it is empty and the frame collector is done.
         * It performs the following steps for each frame:
         * 1. Preprocesses the frame using the Frame Preprocessor.
         * 2. Runs inference on the preprocessed frame using the Inference Engine.
         * 3. Postprocesses the inference results in the Frame Postprocessor.
         * 4. Pushes the original frame and the processed frame to the output frames queue.
         * 
         * The function also checks if the GUI window is closed and stops adding frames to the input frame queue if it is.
         */
        void run();

    private:
        std::unique_ptr<IFramePreprocessor<FrameType>> frame_preprocessor_;
        std::unique_ptr<IInferenceEngine<FrameType, InferenceInfoType>> inference_engine_;
        std::unique_ptr<IFramePostprocessor<FrameType, InferenceInfoType>> frame_postprocessor_;
        ObjectDetectorConfig<FrameType>& cfg_;
};

template <typename FrameType, typename InferenceInfoType>
ObjectDetector<FrameType, InferenceInfoType>::ObjectDetector(
        std::unique_ptr<IFramePreprocessor<FrameType>> frame_preprocessor,
        std::unique_ptr<IInferenceEngine<FrameType,InferenceInfoType>> inference_engine,
        std::unique_ptr<IFramePostprocessor<FrameType, InferenceInfoType>> frame_post_processor,
        ObjectDetectorConfig<FrameType>& config)
        : frame_preprocessor_(std::move(frame_preprocessor)),
          inference_engine_(std::move(inference_engine)),
          frame_postprocessor_(std::move(frame_post_processor)),
          cfg_(config) {}

template <typename FrameType, typename InferenceInfoType>
void ObjectDetector<FrameType, InferenceInfoType>::run() {
    while (true) {
        if (cfg_.input_frame_queue_.empty() && cfg_.frame_collector_.is_done()) {
            cfg_.tracker_object_detector_done_ = true;
            break;
        }

        // Gui window was closed, we do not need to continue adding frames to the queue
        if (cfg_.tracker_is_gui_closed_) {
            break;
        }

        if (!cfg_.input_frame_queue_.empty()) {
            const FrameType frame_ogininal = cfg_.input_frame_queue_.get();
            FrameType blob = frame_preprocessor_->run(frame_ogininal);

            InferenceInfoType inference_res;
            inference_engine_->process(blob, inference_res);

            FrameType frame_processed = frame_ogininal.clone();
            frame_postprocessor_->run(frame_processed, inference_res);

            cfg_.out_frames_queue_.push(std::make_pair(frame_ogininal, frame_processed));
        }
    }
}

#endif // OBJECT_DETECTOR_HPP