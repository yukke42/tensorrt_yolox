// Copyright 2022 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <rclcpp/rclcpp.hpp>
#include <tensorrt_yolox/tensorrt_yolox.hpp>
#include <string>

#if(defined(_MSC_VER) or (defined(__GNUC__) and (7 <= __GNUC_MAJOR__)))
#include <filesystem>
namespace fs = ::std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = ::std::experimental::filesystem;
#endif

#include <memory>
#include <string>

namespace tensorrt_yolox
{
class YoloXSingleImageInferenceNode : public rclcpp::Node
{
public:
  explicit YoloXSingleImageInferenceNode(const rclcpp::NodeOptions & node_options)
  : Node("yolox_single_image_inference", node_options)
  {
    const auto image_path = declare_parameter("image_path", "");
    const auto model_path = declare_parameter("model_path", "");
    const auto precision = declare_parameter("precision", "fp32");
    const auto save_image = declare_parameter("save_image", false);
    auto p = fs::path(image_path);
    const auto ext = p.extension().string();
    p.replace_extension("");
    const auto output_image_path = declare_parameter(
      "output_image_path",
      p.string() + "_detect" + ext
    );

    const std::vector<cv::Scalar> label_colors = {
      cv::Scalar(255, 255, 255),
      cv::Scalar(30, 144, 255),
      cv::Scalar(255, 30, 144),
      cv::Scalar(144, 255, 30),
      cv::Scalar(119, 11, 32),
      cv::Scalar(32, 11, 119),
      cv::Scalar(255, 192, 203),
      cv::Scalar(255, 255, 255)
    };

    auto trt_yolox = std::make_unique<tensorrt_yolox::TrtYoloX>(model_path, precision);
    auto image = cv::imread(image_path);
    tensorrt_yolox::ObjectArrays objects;
    trt_yolox->doInference({image}, objects);
    for (const auto & object : objects[0]) {
      const auto left = object.x_offset;
      const auto top = object.y_offset;
      const auto right = std::clamp(left + object.width, 0, image.cols);
      const auto bottom = std::clamp(top + object.height, 0, image.rows);

      const auto color = label_colors[object.type];
      cv::rectangle(
        image, cv::Point(left, top), cv::Point(right, bottom), color, 2, 8, 0);
      
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3) << object.score;
      std::string score_str = oss.str();
      const int font = cv::FONT_HERSHEY_SIMPLEX;
      constexpr float font_size = 0.7;
      int base_line = 0;
      const auto text_size = cv::getTextSize(score_str, font, font_size, 1, &base_line);
      cv::putText(image, score_str, cv::Point(left, top + text_size.height), font, font_size, color, 2, 8, 0);
    }
    if (!save_image) {
      cv::imshow("inference image", image);
      cv::waitKey(0);
      rclcpp::shutdown();
    }
    cv::imwrite(output_image_path, image);
    rclcpp::shutdown();
  }
};
}  // namespace tensorrt_yolox

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tensorrt_yolox::YoloXSingleImageInferenceNode)
