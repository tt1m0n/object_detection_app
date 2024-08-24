#ifndef DEARIMGUI_HPP
#define DEARIMGUI_HPP

#include <memory>
#include <utility>
#include <atomic>

#include <imgui.h>

#include "IGui.hpp"
#include "IDearImgBackend.hpp"
#include "FrameQueue.hpp"

/**
 * @brief Dear ImGui class to draw frames using Dear ImGui and backend.
 * 
 * This class is responsible for drawing the frames using Dear ImGui and backend. It uses the frame queue to get the
 * original and processed frames and add them to the window. The class is responsible for preparing the frame for
 * rendering in Dear ImGui and rendering the frame using the backend.
 * 
 * @tparam FrameType - the type of the frame (cv::Mat, cv::cuda::GpuMat)
 */
template <typename FrameType>
class DearImGui : public IGui
{
    public:

        /**
         * @brief Class representing the DearImGui object.
         * 
         * This class is responsible for managing the Dear ImGui library and chosen backend.
         * It provides an interface for rendering GUI elements in imgui and backend part.
         * 
         * @param backend A unique pointer to the backend implementation.
         * @param frame_queue A reference to the frame queue used for consume pairs of frame types.
         * @param is_finish A reference to a boolean flag indicating whether the frame prroducer has finished
         *        adding frames to the queue.
         * @param is_gui_window_closed A reference to a boolean flag indicating whether the GUI window is closed
         *        by the user. We need it stop object detection process. 
         */
        DearImGui(std::unique_ptr<IDearImgBackend<FrameType>> backend,
            FrameQueue<std::pair<FrameType, FrameType>>& frame_queue,
            std::atomic<bool>& is_finish,
            std::atomic<bool>& is_gui_window_closed);
        ~DearImGui() override;

        void initialize() override;

        /**
         * @brief Draws the frames using Dear ImGui and backend.
         * 
         * This function it is the main cycle to draw frames. It continuously prepares and renders the frames using Dear ImGui
         * and backend until the queue collector has no more frames to add or the window is closed by usesr.
         */
        void draw() override;

    private:
        /**
         * @brief Prepares the frame for rendering in Dear ImGui.
         * 
         * This function prepares the frame for rendering in Dear ImGui by setting up the window position and size,
         * creating a new frame, get the original and processed frames from the queue, and add them to the window.
         * 
         * @return True if the frame is prepared successfully, false - queue collector has no more frames to add
         */
        bool prepareFrame();

    private:
        std::unique_ptr<IDearImgBackend<FrameType>> backend_;
        FrameQueue<std::pair<FrameType, FrameType>>& frames_queue_to_draw_;
        std::atomic<bool>& is_queue_collector_finish_;
        std::atomic<bool>& is_gui_window_closed_;
};

template <typename FrameType>
DearImGui<FrameType>::DearImGui(std::unique_ptr<IDearImgBackend<FrameType>> backend,
    FrameQueue<std::pair<FrameType, FrameType>>& frame_queue, std::atomic<bool>& is_finish, std::atomic<bool>& is_gui_closed)
    : backend_(std::move(backend)), frames_queue_to_draw_(frame_queue), 
      is_queue_collector_finish_(is_finish), is_gui_window_closed_(is_gui_closed){};

template <typename FrameType>
DearImGui<FrameType>::~DearImGui()
{
    is_gui_window_closed_ = true;
}

template <typename FrameType>
void DearImGui<FrameType>::initialize()
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    backend_->initialize();
}

template <typename FrameType>
void DearImGui<FrameType>::draw()
{
    while (true) {
        if (backend_->prepareWindowToRender() == false || prepareFrame() == false) {
            is_gui_window_closed_ = true;
            return;
        }

        ImGui::Render();
        backend_->render();
    }
}

template <typename FrameType>
bool DearImGui<FrameType>::prepareFrame()
{
    // all frames are drawn
    if (frames_queue_to_draw_.empty() && is_queue_collector_finish_) {
        return false;
    }

    // wait until new element will be added to queue.
    while (frames_queue_to_draw_.empty()) {
        // check maybe flag is set to true and we do not need to wait anymore
        if (is_queue_collector_finish_) {
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    // full screen sizes
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::NewFrame();
    ImGui::Begin("Object Detection Demo", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_HorizontalScrollbar);

    auto [original_frame, processed_frame] = frames_queue_to_draw_.get();

    ImGui::Text("Original Frame / Processed Frame");
    ImGui::Image(backend_->prepareOriginalFrameTexture(original_frame),
        ImVec2(original_frame.cols, original_frame.rows));
    ImGui::SameLine();
    ImGui::Image(backend_->prepareProcessedFrameTexture(processed_frame),
        ImVec2(processed_frame.cols, processed_frame.rows));
    ImGui::End();

    return true;
}


#endif // DEARIMGUI_HPP