#ifndef IFAMEPREPROCESSOR_HPP_
#define IFAMEPREPROCESSOR_HPP_

/**
 * @brief Interface for frame preprocessing.
 * 
 * This interface defines the contract for frame preprocessing classes.
 * Frame preprocessing involves applying transformations or operations on a frame before further processing.
 * 
 * @tparam FrameType The type of frame to be preprocessed (cv::Mat, cv::cuda::GpuMat, etc).
 */
template <typename FrameType>
class IFramePreprocessor
{
    public:
        IFramePreprocessor() {};
        virtual FrameType run(const FrameType& frame_type) = 0;
        virtual ~IFramePreprocessor() {};
};

#endif // IFAMEPREPROCESSOR_HPP_