#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torchvision/vision.h>  // Ensure you have the TorchVision library available

class ImageSubscriber : public rclcpp::Node
{
public:
    ImageSubscriber() : Node("image_subscriber")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image", 10, std::bind(&ImageSubscriber::on_image, this, std::placeholders::_1));
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

        cv::Mat img = cv_ptr->image;

        cv::Mat gray, bfilter, edged, mask, new_image, cropped_image;
        
        // Convert to grayscale
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        
        // Noise reduction
        cv::bilateralFilter(gray, bfilter, 11, 17, 17);
 
        
        // Edge detection
        cv::Canny(bfilter, edged, 30, 200);

        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(edged.clone(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        
        // Sort contours based on area
        std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2){
            return cv::contourArea(c1, false) > cv::contourArea(c2, false);
        });
        
        // Find the contour with 4 corners (assuming it's a rectangle)
        std::vector<cv::Point> location;
        for (const auto& contour : contours) {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, 10, true);
            if (approx.size() == 4) {
                location = approx;
                break;
            }
        }
        
        // Draw the contour on a mask
        mask = cv::Mat::zeros(gray.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> locationVec{location};
        cv::drawContours(mask, locationVec, 0, 255, -1);

        
        // Bitwise the original image and mask
        cv::bitwise_and(img, img, new_image, mask);

        
        // Get bounding box coordinates from the contour
        cv::Rect bounding_box = cv::boundingRect(location);

        // Crop the license plate image
        cropped_image = gray(bounding_box);

        // Display the final cropped image
        cv::imshow("Cropped Image", cropped_image);
        cv::waitKey(1); 
        }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageSubscriber>());
    rclcpp::shutdown();
    return 0;
}
