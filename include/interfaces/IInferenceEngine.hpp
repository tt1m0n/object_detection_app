#ifndef IINFERENCEENGINE_HPP_
#define IINFERENCEENGINE_HPP_

/**
 * @brief Interface for an inference engine that performs object detection on frames.
 * 
 * This interface defines the contract for an inference engine that processes frames of type FrameType
 * (cv::Mat, cv::cuda::GpuMat, etc) and produces inference results of type InferenceResult (std::vector<cv::Mat>,
 * std::vector<cv::cuda::GpuMat>, etc).
 */
template <typename FrameType, typename InferenceResult>
class IInferenceEngine
{
    public:
        IInferenceEngine() {};
        virtual void process(FrameType& frame_type, InferenceResult& res) = 0;
        virtual ~IInferenceEngine() {};
};

#endif // IINFERENCEENGINE_HPP_