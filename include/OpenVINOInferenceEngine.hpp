#ifndef OPENVINOINFERENCEENGINE_HPP_
#define OPENVINOINFERENCEENGINE_HPP_

#include "IInferenceEngine.hpp"

#include "openvino/openvino.hpp"

using namespace ov::preprocess;

class OpenVINOInferenceEngine : public IInferenceEngine<ov::Tensor, ov::Tensor>
{
    public:
        explicit OpenVINOInferenceEngine(const std::string& model_path);
        void process(ov::Tensor& frame, ov::Tensor& res) override;
        ~OpenVINOInferenceEngine();

    private:
        ov::Core core_;                          // OpenVINO Core object
        std::shared_ptr<ov::Model> network_;     // OpenVINO Model object
        ov::CompiledModel compiled_model_;       // Compiled model object
        ov::InferRequest infer_request_;         // Inference request object
        std::string input_blob_name_;            // Name of the input tensor
        std::string output_blob_name_;           // Name of the output tensor};
};

#endif // OPENVINOINFERENCEENGINE_HPP_