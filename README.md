### Project info
C++ small application that performs video processing and object detection using YOLO V3 model, Opencv, OpenGL, glfw3 that is provided for object detection and displaying results in a simple graphical user interface (GUI).

### Build for Ubuntu
mkdir build && cd build
cmake ..
make

### Library depency
To build application if should be insalled:
- OpenCV 4.1
- OpenGL
- glfw3

### Usage
./object_dector <video_path>
example: ./object_detector ../resources/data/test.mp4 

to specify/change pathes to model files (config, weighs, names, colors) use config.yaml.
now it is specified according default file locations.

### Short Project Description

#### Command line input and configuration
- video path get from command line argument according task description
- about model files. To avoid hardcoding paths I use config.yaml file and parse it with yaml-cpp library

#### Video Processing
**Collector**\
*base*: IFrameCollector.hpp -> *derived*: OpenCVFrameCollector.hpp
- done in separate thread: read and put frame into thread-safe queue

#### Inference
**Object detector** - ObjecDetector.hpp\
Done in separate thread. The object detector retrieves frames from a thread-safe queue where the collector produces and adds frames, processed frames and put in another thread-safe queue. The object detector has three main elements/steps:\
**Preprocessor**- base: IFramePreprocessor.hpp -> derived: OpenCVFramePreprocessor.hpp\
    - make preparation for Infererence Engine\
**Inference engine** - base: IInferenceEngine.hpp   -> derived: OpenCVDnnInferenceEngine.hpp\
    - forward frame to model and get result\
**Postprocessor**   - base: IFramePostrocessor.hpp -> derived: OpenCVFramePostprocessor.hpp\
    - make postprocessing of result from Inference Engine and draw bounding boxes and confidence on frame\

#### GUI
**Drawer** - Drawer.hpp\
    - The drawer retrieves frames from a thread-safe queue where the object detector produces and adds frames. The drawer displays the frames with bounding boxes and confidence values in a GUI window.
