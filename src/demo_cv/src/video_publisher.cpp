#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <torch/torch.h> 

class VideoPublisher : public rclcpp::Node
{
public:
    VideoPublisher() : Node("video_publisher"), cap_(ament_index_cpp::get_package_share_directory("demo_cv") + "/videos/vid.mp4")
    {
        // Check and log GPU availability
        if (torch::cuda::is_available()) {
            RCLCPP_INFO(this->get_logger(), "GPU is available.");
        } else {
            RCLCPP_INFO(this->get_logger(), "GPU is not available.");
        }

        if (!cap_.isOpened()) {
            throw std::runtime_error("Failed to open video file");
        }

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("video", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&VideoPublisher::on_timer, this));
    }

private:
    void on_timer()
    {
        cv::Mat frame;
        cap_ >> frame;
        

        if (frame.empty()) {
            RCLCPP_INFO(this->get_logger(), "End of video stream");
            rclcpp::shutdown();
            return;
        }

        // Display the frame
        cv::imshow("Video Frame", frame);
        cv::waitKey(1);  // wait for 1 ms for any keyboard event

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        publisher_->publish(*msg);
    }

    cv::VideoCapture cap_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoPublisher>());
    rclcpp::shutdown();
    return 0;
}
