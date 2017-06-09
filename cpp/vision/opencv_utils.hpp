/*
 * opencvutils.hpp
 *
 *  Created on: Apr 30, 2013
 *      Author: mdantone
 */

#ifndef OPENCVUTILS_HPP_
#define OPENCVUTILS_HPP_

#include <opencv2/opencv.hpp>

namespace vision {
namespace opencv_utils {


  cv::Mat extract_roi(const cv::Mat img, const cv::Rect roi);

  bool check_uniqueness( const std::vector<cv::Rect> rects, const cv::Rect rect );

} /* namespace opencv_utils */
} /* namespace vision */
#endif /* OPENCVUTILS_HPP_ */
