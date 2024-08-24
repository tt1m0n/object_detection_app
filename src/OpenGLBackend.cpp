#include <iostream>

#include "OpenGLBackend.hpp"

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

OpenGLBackend::OpenGLBackend() {}
OpenGLBackend::~OpenGLBackend()
{
    if (window_) {
        glfwDestroyWindow(window_);
    }

    if(texture_id_original_ != 0) {
        glDeleteTextures(1, &texture_id_original_);
    }

    if(texture_id_processed_ != 0) {
        glDeleteTextures(1, &texture_id_processed_);
    }
    
    // destroy the GLFW windows
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    glfwTerminate();
}

bool OpenGLBackend::initialize()
{
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // create a mode window and its OpenGL context
    window_ = glfwCreateWindow(1420, 520, "OpenGL Window", NULL, NULL);
    if (window_ == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        return false;
    }

    glfwMakeContextCurrent(window_);
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    // Enable vsync
    glfwSwapInterval(1);

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init();
}

void OpenGLBackend::render()
{
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    // Swap buffers
    glfwSwapBuffers(window_);
}

bool OpenGLBackend::prepareWindowToRender() {
    if(glfwWindowShouldClose(window_)) {
        return false;
    }

    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);

    return true;
}

void OpenGLBackend::loadTextureFromMat(const cv::Mat& frame, GLuint* texture_id)
{
    glGenTextures(1, texture_id);
    glBindTexture(GL_TEXTURE_2D, *texture_id);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLenum format;
    if (frame.channels() == 1) {
        format = GL_RED;
    } else if (frame.channels() == 3) {
        format = GL_RGB;
    } else if (frame.channels() == 4) {
        format = GL_RGBA;
    } else {
        std::cerr << "Unsupported number of channels" << std::endl;
        return;
    }

    glTexImage2D(GL_TEXTURE_2D, 0, format, frame.cols, frame.rows, 0, format, GL_UNSIGNED_BYTE, frame.data);
    glGenerateMipmap(GL_TEXTURE_2D);
}

void* OpenGLBackend::prepareOriginalFrameTexture(const cv::Mat& original_frame) {
    if (texture_id_original_ != 0) {
        glDeleteTextures(1, &texture_id_original_);
    }

    loadTextureFromMat(original_frame, &texture_id_original_);
    return reinterpret_cast<void*>(texture_id_original_);
}


void* OpenGLBackend::prepareProcessedFrameTexture(const cv::Mat& processed_frame) {
    if (texture_id_processed_ != 0) {
        glDeleteTextures(1, &texture_id_processed_);
    }

    loadTextureFromMat(processed_frame, &texture_id_processed_);
    return reinterpret_cast<void*>(texture_id_processed_);
}