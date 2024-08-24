#ifndef IFAMEPOSTPROCESSOR_HPP_
#define IFAMEPOSTPROCESSOR_HPP_

/**
 * @brief Interface for frame postprocessing.
 * 
 * This interface defines the contract for frame postprocessors. Frame postprocessors are responsible for
 * processing the output of an inference enngine and modifying the frame accordingly.
 * 
 * @tparam FrameType The type of the frame to be processed.
 * @tparam InferenceInfo The type of the inference engine information.
 */
template <typename FrameType, typename InferenceInfo>
class IFramePostprocessor
{
    public:
        IFramePostprocessor() {};
        virtual void run(FrameType& frame, const InferenceInfo& info) = 0;
        virtual ~IFramePostprocessor() {};
};

#endif // IFAMEPOSTPROCESSOR_HPP_