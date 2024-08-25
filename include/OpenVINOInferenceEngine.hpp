#ifndef OPENVINOINFERENCEENGINE_HPP_
#define OPENVINOINFERENCEENGINE_HPP_

#include "IInferenceEngine.hpp"

#include <inference_engine.hpp>
#include "openvino/openvino.hpp"

using namespace ov::preprocess;

class OpenVINOInferenceEngine : public IInferenceEngine<InferenceEngine::Blob::Ptr, std::vector<InferenceEngine::Blob::Ptr>>
{
    public:
        OpenVINOInferenceEngine(const std::string& model_xml_path, const std::string& model_bin_path);
        void process(InferenceEngine::Blob::Ptr& frame, std::vector<InferenceEngine::Blob::Ptr>& res) override;
        ~OpenVINOInferenceEngine();

    private:
        InferenceEngine::Core ie_;
        InferenceEngine::CNNNetwork network_;
        InferenceEngine::ExecutableNetwork executable_network_;
        InferenceEngine::InferRequest infer_request_;
};

#endif // OPENVINOINFERENCEENGINE_HPP_