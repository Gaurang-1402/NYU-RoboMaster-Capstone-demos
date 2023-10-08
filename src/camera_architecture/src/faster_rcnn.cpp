#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/opencv.hpp"
#include <torch/script.h>

class RCNN : public rclcpp::Node
{
public:
    RCNN() : Node("image_display")
    {
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_topic", 10,
            std::bind(&RCNN::display, this, std::placeholders::_1));

        // Load the ONNX model
        module_ = torch::jit::load("/home/nyu_robomaster/Desktop/NYU_RoboMaster_Cpp_ws/src/camera_architecture/models/faster_rcnn.onnx");
    }

private:
    void display(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
        
        // Convert the frame to RGB
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Convert the frame to a torch tensor
        auto input_tensor = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3}, torch::kByte);
        input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();

        // Prepare the model input
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Run the model
        auto output = module_.forward(inputs);

        // Assume that the output is a tuple of (boxes, scores, labels)
        auto outputs_tuple = output.toTuple();
        auto boxes = outputs_tuple->elements()[0].toTensor().accessor<float, 2>();
        auto scores = outputs_tuple->elements()[1].toTensor().accessor<float, 1>();
        auto labels = outputs_tuple->elements()[2].toTensor().accessor<int64_t, 1>();

        // Threshold for detection
        const float score_threshold = 0.5;

        for (int i = 0; i < boxes.size(0); ++i) {
            float score = scores[i];
            if (score > score_threshold) {
                int x_min = static_cast<int>(boxes[i][0]);
                int y_min = static_cast<int>(boxes[i][1]);
                int x_max = static_cast<int>(boxes[i][2]);
                int y_max = static_cast<int>(boxes[i][3]);

                // Draw the bounding box
                cv::rectangle(frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 255, 0), 2);
            }
        }
        
        // Optionally, convert frame back to BGR for displaying
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        
        // Display the frame
        cv::imshow("Image Display", frame);
        cv::waitKey(30);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    torch::jit::Module module_;  // TorchScript module for the ONNX model
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RCNN>());
    rclcpp::shutdown();
    return 0;
}
