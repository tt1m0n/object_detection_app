#include "OpenVINOInferenceEngine.hpp"

OpenVINOInferenceEngine::OpenVINOInferenceEngine(const std::string& model_path)
{
    std::cout << "Start reading model" << std::endl;
    // Load the model from the provided path (assuming .xml and .bin files are present)
    std::shared_ptr<ov::Model> network = core_.read_model(model_path);

    std::cout << "finish reading model" << std::endl;

    // Optionally set batch size (not typically necessary in the new API, but can be done manually if needed)
    network->get_parameters().at(0)->set_partial_shape({1, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()});

    // ERROR: yolo3.onnx can has more htan 1 inout so need o process i.
    // BUT I need understan dwhta is input
    auto input_info = network->input();
    input_blob_name_ = input_info.get_any_name();

    // Set pre-processing: Resize and set layout and precision
    auto preproc = ov::preprocess::PrePostProcessor(network);
    preproc.input().tensor().set_layout("NHWC").set_element_type(ov::element::u8);
    preproc.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    preproc.input().model().set_layout("NCHW");

    // Configure output
    auto output_info = network->output();
    output_blob_name_ = output_info.get_any_name();

    preproc.output().tensor().set_element_type(ov::element::f32);

    // Apply the preprocessing to the model
    network = preproc.build();

    // Load the model onto the device (e.g., CPU)
    ov::CompiledModel compiled_model = core_.compile_model(network, "CPU");

    // Create an inference request
    infer_request_ = compiled_model.create_infer_request();
}

void OpenVINOInferenceEngine::process(ov::Tensor& frame, ov::Tensor& res)
{
    // Set the input tensor
    infer_request_.set_input_tensor(frame);

    // Perform inference
    infer_request_.infer();

    // Get the output tensor
    ov::Tensor output = infer_request_.get_output_tensor();

    std::cout << "preprocessed tensor shape" << std::endl;
}

OpenVINOInferenceEngine::~OpenVINOInferenceEngine() {}