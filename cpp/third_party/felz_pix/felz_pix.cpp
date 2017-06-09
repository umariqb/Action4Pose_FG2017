/*
 * felz_pix.cpp
 *
 *  Created on: Aug 7, 2013
 *      Author: lbossard
 */

#include "felz_pix.hpp"

#include <boost/scoped_ptr.hpp>
#include <glog/logging.h>

#include "segment-image.h"

namespace awesomeness {
namespace third_party {
namespace felz_pix {

typedef image<float> FelzImage;
typedef boost::scoped_ptr<FelzImage> FelzImagePtr;

int segment_image(
    const cv::Mat_<cv::Vec3b>& img,
    const float sigma,
    const float k,
    const int min_size,
    cv::Mat_<int32_t>* components_
){

  const int cols = img.cols;
  const int rows = img.rows;

  FelzImagePtr r(new FelzImage(cols, rows));
  FelzImagePtr g(new FelzImage(cols, rows));
  FelzImagePtr b(new FelzImage(cols, rows));

  // split image channels
  {
    typedef cv::Mat_<cv::Vec3b> ImgMat;
    ImgMat::const_iterator it = img.begin();
    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < cols; ++x) {
        const cv::Vec3b& pixel = *it;
        imRef(b, x, y) = pixel[0];
        imRef(g, x, y) = pixel[1];
        imRef(r, x, y) = pixel[2];
        ++it;
      }
    }
    CHECK(it == img.end());
  }

  // smooth each color channel
  FelzImagePtr smooth_r(::smooth(r.get(), sigma));
  FelzImagePtr smooth_g(::smooth(g.get(), sigma));
  FelzImagePtr smooth_b(::smooth(b.get(), sigma));

  // do segmentation
  boost::scoped_ptr<universe> u(
      ::segment_image(smooth_r.get(), smooth_g.get(), smooth_b.get(), sigma, k, min_size));

  // copy output
  {
    typedef std::map<int, int> IdMap;
    IdMap segment_id_map;
    components_->create(img.rows, img.cols);
    typedef cv::Mat_<int32_t> IntMat;
    IntMat::iterator components_it = components_->begin();
    for (int y = 0; y < img.rows; y++) {
      for (int x = 0; x < img.cols; x++) {
        const int segment_id = u->find(y * img.cols + x);

        IdMap::const_iterator component_id_it = segment_id_map.find(segment_id);
        if (component_id_it == segment_id_map.end()){
          int new_component_id = segment_id_map.size();
          segment_id_map[segment_id] = new_component_id;
          *components_it = new_component_id;
        }
        else {
          *components_it = component_id_it->second;
        }
        ++components_it;
      }
    }
    CHECK(components_it == components_->end());
  }

   return u->num_sets();

}

void superpixel_to_mean_rgb(const cv::Mat_<cv::Vec3b>& img,
                            const cv::Mat_<int32_t>& labels,
                            int num_components,
                            cv::Mat_<cv::Vec3b>& dest_img) {


  // compute avg colors and size of regions
  std::vector<int> size(num_components,0);
  std::vector<cv::Vec3f> mean_colors(num_components,cv::Vec3f(0.0,0.0,0.0));

  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      const int segment_id = labels.at<int32_t>(y,x);
      const cv::Vec3b& color = img.at<cv::Vec3b>(y,x);
      size[segment_id] ++;
      mean_colors[segment_id] += color;
    }
  }
  dest_img = img.clone();
  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      const int segment_id = labels.at<int32_t>(y,x);
      if(size[segment_id] > 0 ) {
        u_char r =  mean_colors[segment_id](0) / size[segment_id];
        u_char g =  mean_colors[segment_id](1) / size[segment_id];
        u_char b =  mean_colors[segment_id](2) / size[segment_id];
        dest_img.at<cv::Vec3b>(y,x)(0) = r;
        dest_img.at<cv::Vec3b>(y,x)(1) = g;
        dest_img.at<cv::Vec3b>(y,x)(2) = b;
      }else{
        dest_img.at<cv::Vec3b>(y,x)(0) = 0;
        dest_img.at<cv::Vec3b>(y,x)(1) = 0;
        dest_img.at<cv::Vec3b>(y,x)(2) = 0;
      }

    }
  }
}


double next_halton(int32_t index, int32_t prime_base) {
  double result = 0;
  double f = 1. / prime_base;
  while (index > 0) {
    result += f * (index % prime_base);
    index /= prime_base;
    f /= prime_base;
  }
  return result;
}
void superpixel_to_halton_rgb(const cv::Mat_<int32_t>& labels,
                             cv::Mat_<cv::Vec3b>& img) {
  img = cv::Mat_<cv::Vec3b>(labels.rows, labels.cols);

  for (int r = 0; r < labels.rows; ++r) {
    for (int c = 0; c < labels.cols; ++c) {
      cv::Vec3b& pixel = img(r, c);
      pixel[0] = 127 * next_halton(labels(r, c), 2);
      pixel[1] = 255 * (.5 + next_halton(labels(r, c), 3) * .5);
      pixel[2] = 255 * (.7 + next_halton(labels(r, c), 5) * .3);
    }
  }
  cv::cvtColor(img, img, cv::COLOR_HSV2BGR);

}

} /* namespace felz_pix */
} /* namespace third_party */
} /* namespace awesomeness */
