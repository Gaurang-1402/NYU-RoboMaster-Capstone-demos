#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

class CameraPublisher : public rclcpp::Node
{
public:
    CameraPublisher() : Node("camera_publisher"), cap_(0)
    {
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_topic", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30),
            std::bind(&CameraPublisher::capture_and_publish, this));
    }

private:
    void capture_and_publish()
    {
        cv::Mat frame;
        cap_ >> frame;
        if(!frame.empty())
        {
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            pub_->publish(*msg);
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraPublisher>());
    rclcpp::shutdown();
    return 0;
}
