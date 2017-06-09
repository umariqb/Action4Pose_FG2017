/*
 * common.hpp
 *
 *  Created on: Sep 30, 2013
 *      Author: mdantone
 */

#ifndef COMMON_FEATURES_HPP_
#define COMMON_FEATURES_HPP_

#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"

namespace bodypose {
namespace features {

struct BoxAnnotation {
  BoxAnnotation() : img_url("") {}
  std::string img_url;
  std::vector<cv::Rect> bboxs;
  std::vector<int> labels;
  std::vector<float> confidences;
  std::vector<cv::Point> parts;

  cv::Mat image;
  cv::Mat_<u_char> mask;
  cv::Mat_<float> app_features;
  cv::Mat_<float> spatials_features;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & img_url;
    ar & bboxs;
    ar & labels;
    ar & confidences;
    ar & parts;
  }

};

} /* namespace features */
} /* namespace bodypose */


#endif /* COMMON_FEATURES_HPP_ */
