#include <iostream>
#include <string>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "FrameQueue.hpp"

// OpenCV includes
#include "OpenCVFrameCollector.hpp"
#include "OpenCVFramePreprocessor.hpp"
#include "OpenCVDnnInferenceEngine.hpp"
#include "OpenCVFramePostprocessor.hpp"
#include "ObjectDetector.hpp"

// OpenVINO includes
#include "OpenVINOFramePreprocessor.hpp"
#include "OpenVINOInferenceEngine.hpp"
#include "OpenVINOFramePostprocessor.hpp"

#include "Drawer.hpp"
#include "DearImGui.hpp"
#include "OpenGLBackend.hpp"

// hardcoded cons
const std::string kConfigPath = "../config.yaml";
const std::string kExePrefix = "../";
const int kArgsNum = 2;

int main(int argc, char* argv[]) {
    if (argc < kArgsNum) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return 1;
    }

    std::string kVideoPath = argv[1];

    // Load the YAML file
    YAML::Node config = YAML::LoadFile(kConfigPath);

    // Access configuration values
    const std::string kModelConfigPath = kExePrefix + config["modelConfigPath"].as<std::string>();
    const std::string kModelWeightPath = kExePrefix + config["modelWeightPath"].as<std::string>();
    const std::string kOnnxYolo3ModelPath = kExePrefix + config["onnxYolo3ModelPath"].as<std::string>();
    const std::string kModelClassesPath = kExePrefix + config["modelClassesPath"].as<std::string>();
    const std::string kModelClassesColorsPath = kExePrefix+ config["modelClassesColorsPath"].as<std::string>();

    // setup OpenCV frame collector
    FrameQueue<cv::Mat> collector_out_queue;
    auto opencv_collector = std::make_unique<OpenCVFrameCollector>(collector_out_queue, kVideoPath);

    // Start the thread to collect frames
    std::thread collector_thread(&OpenCVFrameCollector::run, opencv_collector.get());

    // createt setup for Object detector

    FrameQueue<std::pair<cv::Mat, cv::Mat>> object_detector_out_queue;
    // tracker to check in GUI if object detector is done processing frames
    std::atomic<bool> tracker_object_detector_done(false);
    // tracker to check in Object detector if GUI is closed by user so we do not need continue processing frames
    std::atomic<bool> tracker_is_gui_closed(false);

    ObjectDetectorConfig<cv::Mat> object_detector_config(collector_out_queue,
        object_detector_out_queue,
        tracker_object_detector_done,
        tracker_is_gui_closed,
        *opencv_collector);

    const cv::Size kFrameSize(416, 416);
    const cv::Scalar kMeanValue(0, 0, 0);
    const float kConfThreshold = 0.5;
    const float kNmsThreshold = 0.4;
    const double kScaleFactor = 1.0 / 255.0;

    // Create object detector
    // ObjectDetector<cv::Mat, cv::Mat, std::vector<cv::Mat>> object_detector(
    //     std::make_unique<OpenCVFramePreprocessor>(kScaleFactor, kFrameSize, kMeanValue, true),
    //     std::make_unique<OpenCVDnnInferenceEngine>(kModelConfigPath, kModelWeightPath),
    //     std::make_unique<OpenCVFramePostprocessor>(kConfThreshold, kNmsThreshold, kModelClassesPath, kModelClassesColorsPath),
    //     object_detector_config);

    ObjectDetector<cv::Mat, ov::Tensor, ov::Tensor> object_detector(
        std::make_unique<OpenVINOFramePreprocessor>(kScaleFactor, kFrameSize, kMeanValue, true),
        std::make_unique<OpenVINOInferenceEngine>(kOnnxYolo3ModelPath),
        std::make_unique<OpenVINOFramePostprocessor>(),
        object_detector_config);

    // Start separate thread to run object detector
    std::thread object_detector_thread([&]() {
        object_detector.run();
    });

    // Draw frames in GUI
    // auto gui = std::make_unique<DearImGui<cv::Mat>>(std::make_unique<OpenGLBackend>(),
    //     object_detector_out_queue,
    //     tracker_object_detector_done, tracker_is_gui_closed);
    // Drawer drawer(std::move(gui));
    // drawer.run();

    collector_thread.join();
    object_detector_thread.join();

    return 0;
}
