/*
 * opencvutils.cpp
 *
 *  Created on: Apr 30, 2013
 *      Author: mdantone
 */

#include "opencv_utils.hpp"
#include "geometry_utils.hpp"
#include <glog/logging.h>

using namespace cv;
namespace vision {
namespace opencv_utils {


void rect_to_string(Rect rect, string s = "" ) {

  LOG(INFO) << s  << rect.x << " " << rect.y << " " << rect.width << " " << rect.height;

}
Mat extract_roi(const Mat img, const Rect roi) {

  Mat roi_img;
  Rect inter = ::vision::geometry_utils::intersect(Rect(0,0, img.cols, img.rows), roi );

  // roi is within the matrix
  if( inter.width == roi.width && inter.height == roi.height) {
    roi_img = img(roi).clone();

  }else{
    roi_img = Mat(roi.height, roi.width, img.type(), Scalar(0) );

    cv::Mat min(1,1,CV_8UC1, cv::Scalar::all(0));
    cv::Mat max(1,1,CV_8UC1, cv::Scalar::all(255));
    cv::theRNG().fill(roi_img, cv::RNG::UNIFORM, min, max);


    Rect inter_roi = Rect(0,0,inter.width,inter.height);
    Rect inter_img = Rect(roi.x,roi.y,inter.width,inter.height);

    if(inter.width != roi.width ) {
      if(roi.x < 0) {
        inter_roi.x = abs(roi.x);
        inter_img.x = 0;
      }else{
        inter_img.x = roi.x;
      }
    }

    if(inter.height != roi.height ) {
      if(roi.y < 0) {
        inter_roi.y = abs(roi.y);
        inter_img.y = 0;

      }else{
        inter_img.y = roi.y;
      }
    }
    roi_img(inter_roi).setTo(Scalar(0));
    add( roi_img(inter_roi), img(inter_img), roi_img(inter_roi) );
  }

  return roi_img;
}

bool check_uniqueness( const std::vector<cv::Rect> rects, const cv::Rect rect ) {
  for(int i=0; i < rects.size(); i++) {
    if( (rect.x == rects[i].x) && (rect.y == rects[i].y) &&
        (rect.width == rects[i].width) && (rect.height == rects[i].height)) {
      return false;
    }
  }
  return true;
}


} /* namespace opencv_utils */
} /* namespace vision */
