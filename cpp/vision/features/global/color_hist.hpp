/*
 * color_hist.hpp
 *
 *  Created on: Sep 11, 2013
 *      Author: mdantone
 */

#ifndef COLOR_HIST_HPP_
#define COLOR_HIST_HPP_

#include <opencv2/core/core.hpp>

namespace vision {
namespace features {
namespace global {


class ColorHist {
public:
  ColorHist() {};

  void static extract_hist(const cv::Mat_<cv::Vec3b>& img,
      cv::Mat_<int>& hist,
      int dim = 8,
      cv::Mat_<uchar> mask = cv::Mat_<uchar>());

  static int color_to_index(const cv::Vec3b color, int dim = 8 );
  static cv::Vec3b index_to_color(const int index, int dim = 8 );
  static void color_to_hist(const cv::Vec3b color, std::vector<unsigned short>& hist, int dim = 8);

  virtual ~ColorHist(){};

};

} /* namespace global */
} /* namespace features */
} /* namespace vision */

#endif /* COLOR_HIST_HPP_ */
