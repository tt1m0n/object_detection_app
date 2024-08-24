#ifndef IDEARIMGBACKEND_HPP_
#define IDEARIMGBACKEND_HPP_

/**
 * @brief Interface for a backend implementation of the DearImg object detection app.
 * 
 * This interface defines the methods that a backend implementation of the DearImg object detection app should provide.
 * The backend is responsible for initializing the app, preparing the window for rendering, preparing the textures for the original and processed frames,
 * and rendering the frames.
 * 
 * @tparam FrameType The type of the frame used in the app (cv::Mat, cv::cuda::GpuMat, etc).
 */
template <typename FrameType>
class IDearImgBackend
{
    public:
        IDearImgBackend() {};
        virtual ~IDearImgBackend() = default;
        virtual bool initialize() = 0;
        virtual bool prepareWindowToRender() = 0;
        virtual void* prepareOriginalFrameTexture(const FrameType& original_frame) = 0;
        virtual void* prepareProcessedFrameTexture(const FrameType& processed_frame) = 0;
        virtual void render() = 0;
};

#endif // IDEARIMGBACKEND_HPP_