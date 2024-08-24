#ifndef OPENGL_BACKEND_HPP
#define OPENGL_BACKEND_HPP

#include "IDearImgBackend.hpp"

#include <glad/gl.h>
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>

class OpenGLBackend : public IDearImgBackend<cv::Mat>
{
    public:
        OpenGLBackend();
        ~OpenGLBackend() override;
        bool initialize() override;
        void render() override;
        bool prepareWindowToRender() override;
        void* prepareOriginalFrameTexture(const cv::Mat& original_frame) override;
        void* prepareProcessedFrameTexture(const cv::Mat& processed_frame) override;

    private:
        void loadTextureFromMat(const cv::Mat& frame, GLuint* texture_id);
        bool prepareFrame();


    private:
        GLFWwindow* window_ = nullptr;
        GLuint texture_id_original_ = 0;
        GLuint texture_id_processed_ = 0;
};

#endif // OPENGL_BACKEND_HPP