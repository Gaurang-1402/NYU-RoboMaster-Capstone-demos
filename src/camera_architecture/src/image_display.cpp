#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include <torch/torch.h>
#include <iostream>
#include <sstream>

class ImageDisplay : public rclcpp::Node
{
public:
    ImageDisplay() : Node("image_display")
    {
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/camera/color/image_raw", 10,
            std::bind(&ImageDisplay::display, this, std::placeholders::_1));
    }

private:
    void display(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::imshow("Image Display", frame);
        cv::waitKey(30);
        
        // Create a random tensor
        torch::Tensor rand_tensor = torch::rand({2, 3});
        
        // Convert the tensor to a string so it can be logged
        std::ostringstream oss;
        oss << rand_tensor;
        std::string tensor_str = oss.str();
        
        // Log the random tensor
        RCLCPP_INFO(this->get_logger(), "Random Tensor: \n%s", tensor_str.c_str());
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageDisplay>());
    rclcpp::shutdown();
    return 0;
}
