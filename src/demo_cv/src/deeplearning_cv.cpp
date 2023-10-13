#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>  
#include <ament_index_cpp/get_package_share_directory.hpp>


class DeepLearningCV : public rclcpp::Node
{
public:
    DeepLearningCV() : Node("deep_learning_cv")
    {
        // Check and log GPU availability
        if (torch::cuda::is_available()) {
            RCLCPP_INFO(this->get_logger(), "GPU is available.");
        } else {
            RCLCPP_INFO(this->get_logger(), "GPU is not available.");
        }
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "video", 10, std::bind(&DeepLearningCV::on_image, this, std::placeholders::_1));
        
        std::string package_share_directory = ament_index_cpp::get_package_share_directory("demo_cv");
        std::string model_path = package_share_directory + "/models/best.torchscript";
        
        // Load the YOLOv5 model
        try {
            model = torch::jit::load(model_path);
        }
        catch (const c10::Error& e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to load the model: %s", e.what());
            rclcpp::shutdown();
        }
        // Obtain color_selector from a ROS parameter (assuming parameter name is color_selector)
        this->declare_parameter("color_selector", "red");  // Default to red
        this->get_parameter("color_selector", color_selector);
    }

private:
    torch::jit::script::Module model;  // PyTorch model
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    std::string color_selector;

void draw_license_plate_boxes(cv::Mat frame, at::Tensor results, std::string color_selector) {
    cv::Mat expanded_roi;  // Initialize expanded_roi here
    
    at::Tensor result_tensor = results[0];  // Create a non-temporary tensor object
    auto result_data = result_tensor.accessor<float, 2>();

    for (int i = 0; i < result_data.size(0); ++i) {        
        if (result_data[i][4] == 0) {  // Assuming license plates are class 0
            int x1 = static_cast<int>(result_data[i][0]);
            int y1 = static_cast<int>(result_data[i][1]);
            int x2 = static_cast<int>(result_data[i][2]);
            int y2 = static_cast<int>(result_data[i][3]);

            // Crop the region of interest (ROI) from the original image
            cv::Rect roi(x1, y1, x2-x1, y2-y1);
            cv::Mat cropped = frame(roi);
            int new_width = 2 * (x2 - x1);  // Increase the width
            int new_height = 2 * (y2 - y1);  // Increase the height

            // Resize the ROI to the desired dimensions
            cv::resize(cropped, expanded_roi, cv::Size(new_width, new_height));
            cv::Mat hsv;
            cv::cvtColor(expanded_roi, hsv, cv::COLOR_BGR2HSV);

            // Define the lower and upper HSV ranges
            cv::Scalar lower_range(0, 0, 195);
            cv::Scalar upper_range(31, 118, 255);

            // Create a mask based on the defined range
            cv::Mat mask;
            cv::inRange(hsv, lower_range, upper_range, mask);

            // Calculate the percentage of red pixels in the image
            int total_pixels = hsv.total();
            int red_pixels = cv::countNonZero(mask);
            float red_percentage = (float(red_pixels) / total_pixels) * 100;
            float threshold = 5;  // Adjust as needed
            bool is_red = red_percentage >= threshold;

            if (color_selector == "red" && is_red) {
                std::cout << "The image is predominantly red." << std::endl;
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
            }
            else if (color_selector == "blue" && !is_red) {
                std::cout << "The image is predominantly blue." << std::endl;
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
            }
        }
    }

    cv::imshow("Webcam1", frame);
    if (!expanded_roi.empty()) {
        cv::imshow("Webcam2", expanded_roi);
    }
}
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
        cv::resize(frame, frame, cv::Size(416, 416));

        // Convert frame to tensor
        torch::Tensor img_tensor = torch::from_blob(frame.data, { frame.rows, frame.cols, 3 }, torch::kByte).permute({ 2, 0, 1 });

        // Normalize, add batch dimension and perform a forward pass through the model
        img_tensor = img_tensor.to(torch::kFloat).div(255);  // Normalize to [0, 1]
        img_tensor = img_tensor.unsqueeze(0);  // Add batch dimension

        // Create a string stream
        std::ostringstream oss;
        
        // Get the dimensions
        auto sizes = img_tensor.sizes();
        
        // Append each dimension to the string stream
        for (const auto& size : sizes) {
            oss << size << " ";
        }

        // Log the dimensions
        RCLCPP_INFO(this->get_logger(), "Image tensor size: %s", oss.str().c_str());
    
        // Forward pass
        auto output_tuple = model.forward({ img_tensor });
        
        if (!output_tuple.isTuple()) {
            RCLCPP_ERROR(this->get_logger(), "Model output is not a tuple as expected");
            return;
        }
        auto tuple_elements = output_tuple.toTuple()->elements();
        // Assuming the tensor you are interested in is the first element of the tuple
        auto output = tuple_elements[0].toTensor();


        draw_license_plate_boxes(frame, output, color_selector);
        cv::imshow("Webcam1", frame);

        }


};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DeepLearningCV>());
    rclcpp::shutdown();
    return 0;
}
