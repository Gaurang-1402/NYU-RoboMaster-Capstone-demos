#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

class ColorPickerNode : public rclcpp::Node
{
public:
    ColorPickerNode() : Node("color_picker_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "video", 10, std::bind(&ColorPickerNode::on_image, this, std::placeholders::_1));
        lower_red_ = cv::Scalar(0, 0, 0);
        upper_red_ = cv::Scalar(102, 255, 255);
    }

private:
    void on_image(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        cv::Mat hsv_image, mask, result;

        // Convert the frame to HSV
        cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);

        // Threshold the HSV image to get only red colors
        cv::inRange(hsv_image, lower_red_, upper_red_, mask);

        // Bitwise-AND mask and original image
        cv::bitwise_and(frame, frame, result, mask);

        // Edge Detection
        cv::Mat edges;
        cv::Canny(mask, edges, 20, 200, 3, false);

        // Contour Finding
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Filter contours based on area to identify objects
        for (const auto& contour : contours)
        {
            double area = cv::contourArea(contour);
            if (2000 < area && area < 10000)  // Adjust these values based on your requirement
            {
                cv::Rect boundingRect = cv::boundingRect(contour);
                cv::rectangle(result, boundingRect, cv::Scalar(0, 255, 0), 2);
            }
        }

        // Display the result
        cv::imshow("Result", result);
        cv::waitKey(1);

    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    cv::Scalar lower_red_;
    cv::Scalar upper_red_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ColorPickerNode>());
    rclcpp::shutdown();
    return 0;
}
