/*
 * region_growing.hpp
 *
 *  Created on: Sep 17, 2013
 *      Author: mdantone
 */

#ifndef REGION_GROWING_HPP_
#define REGION_GROWING_HPP_
#include <opencv2/opencv.hpp>

namespace vision {
namespace segmentation {

bool inline check(const cv::Mat_<uchar>& img,
                  uchar color,
                  uchar threshold,
                  cv::Point anker,
                  cv::Mat_<uchar>* mask,
                  uchar bg_value = cv::GC_BGD) {
  if(anker.x < 0 || anker.y < 0 || anker.x >= img.cols || anker.y >= img.rows)
    return false;

  if( mask->at<uchar>(anker) == bg_value ){
    return false;
  }

  int diff = static_cast<int>(img.at<uchar>(anker)) - static_cast<int>(color);
  if( abs(diff) > threshold )
    return false;

  return true;

}
void inline region_growing(const cv::Mat_<uchar>& img,
                            uchar color,
                            uchar threshold,
                            cv::Point seed,
                            cv::Mat_<uchar>* mask,
                            uchar bg_value = cv::GC_BGD) {


  std::vector<cv::Point> candidates;
  candidates.push_back(seed);
  while(candidates.size() > 0 ) {

    cv::Point anker = candidates.back();
    candidates.pop_back();

    std::vector<cv::Point> points(4, anker);
    points[0].x -= 1;
    points[1].x += 1;
    points[2].y -= 1;
    points[3].y += 1;

    for(int i=0; i < points.size(); i++) {
      cv::Point p = points[i];
      if( check(img, color, threshold, p, mask, bg_value) ) {
        mask->at<uchar>(p) = bg_value;
        candidates.push_back(p);
      }
    }
  }
}

} /* namespace global */
} /* namespace segmentation */

#endif /* REGION_GROWING_HPP_ */
