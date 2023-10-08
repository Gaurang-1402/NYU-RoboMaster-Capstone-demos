#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

class ImagePublisher : public rclcpp::Node
{
public:
    ImagePublisher() : Node("image_publisher")
    {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&ImagePublisher::on_timer, this));
    }

private:
    void on_timer()
    {
        std::string package_share_directory = ament_index_cpp::get_package_share_directory("demo_cv");
        std::string image_path = package_share_directory + "/images/image1.jpg";
        auto img = cv::imread(image_path, cv::IMREAD_COLOR);
        
        if (img.empty()) {
            RCLCPP_WARN(this->get_logger(), "%s", image_path.c_str());

            RCLCPP_ERROR(this->get_logger(), "Failed to load image");
            return;
        }

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
        publisher_->publish(*msg);

    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImagePublisher>());
    rclcpp::shutdown();
    return 0;
}
