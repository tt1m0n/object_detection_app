#include "OpenCVFramePostprocessor.hpp"

OpenCVFramePostprocessor::OpenCVFramePostprocessor(float conf_threshold, 
    float nms_threshold,
    const std::string& model_classes_path,
    const std::string& model_class_colors_path = "")
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold)
{
    setClasses(model_classes_path);
    if (!model_class_colors_path.empty()) {
        setCollors(model_class_colors_path);
    }
}

void OpenCVFramePostprocessor::run(cv::Mat& frame, const std::vector<cv::Mat>& info)
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> bound_boxes_positions;

    for(int i = 0; i < info.size(); i++)
    {
        //scan through all the boonding boxes output and keep only those with high confidence scores
        float* data = (float*)info[i].data;
        for (int j = 0; j < info[i].rows; ++j, data += info[i].cols)
        {
            cv::Mat scores = info[i].row(j).colRange(5, info[i].cols);
            cv::Point classIdPoint;
	        double confidence;

            // Get the value and location of the maximum score
        	cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            //check if confidence is greater than the thresh than add the classID, confidence and box to there respective vectors
            if (confidence > conf_threshold_)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                class_ids.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                bound_boxes_positions.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Apply nonmax suppression
    std::vector<int> box_indices;
    cv::dnn::NMSBoxes(bound_boxes_positions, confidences, conf_threshold_, nms_threshold_, box_indices);

    for (size_t i = 0; i < box_indices.size(); ++i)
	{
    	int idx = box_indices[i];
    	cv::Rect box = bound_boxes_positions[idx];
    	drawBoundingBoxToFrame(class_ids[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
        addObjectClassToFrame(class_ids[idx], confidences[idx], box.x, box.y, frame);
    }
}

void OpenCVFramePostprocessor::setClasses(const std::string& model_classes_path)
{
    std::fstream class_file(model_classes_path);
    std::string line;
    while(std::getline(class_file, line))
    {
        classes_.push_back(line);
    }
}

void OpenCVFramePostprocessor::setCollors(const std::string& model_class_colors_path)
{
    std::fstream colors_file(model_class_colors_path);
    std::string line;
    while(std::getline(colors_file, line))
    {
        std::istringstream iss(line);
        std::string value;
        std::vector<uint8_t> temp;

        // split the line by commas
        while (std::getline(iss, value, ',')) {
            temp.push_back(static_cast<uint8_t>(std::stoi(value)));
        }

        // store the vector of rgb values in the result vector
        colors_.push_back(temp);
    }
}

void OpenCVFramePostprocessor::drawBoundingBoxToFrame(int class_id,
    int left, int top, int right, int bottom, cv::Mat& frame) const
{
    // default color
    cv::Scalar color(0, 0, 255);

    // set the color based on the class id if the colors and classes are loaded
    if (!classes_.empty() && !colors_.empty())
    {
        const auto& rgb_color = colors_[class_id];
        color = cv::Scalar(rgb_color[0], rgb_color[1], rgb_color[2]);
    }

    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), color);
}

void OpenCVFramePostprocessor::addObjectClassToFrame(int class_id, float conf, int left, int top, cv::Mat& frame) const
{
    // float to string with 3 decimal points
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << conf;
    std::string label = oss.str();

    if (!classes_.empty()) { 
        label = classes_[class_id] + "<" + std::to_string(class_id) + ">:" + label;
    } else {
        std::cerr << "No class names loaded" << std::endl;
        return;
    }

    // default color
    cv::Scalar color(0, 0, 255);

    // set the color based on the class id if the colors and classes are loaded
    if (!colors_.empty())
    {
        const auto& rgb_color = colors_[class_id];
        color = cv::Scalar(rgb_color[0], rgb_color[1], rgb_color[2]);
    }

    // add the label at the top of the bounding box
    int base_line = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.7, 1, &base_line);
    top = std::max(top, label_size.height);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_DUPLEX, 0.7, color, 1);
}


OpenCVFramePostprocessor::~OpenCVFramePostprocessor() {}