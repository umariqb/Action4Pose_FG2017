/*
 * color_hist.cpp
 *
 *  Created on: Sep 11, 2013
 *      Author: mdantone
 */

#include "color_hist.hpp"
#include <glog/logging.h>
#include "cpp/third_party/felz_pix/felz_pix.hpp"
#include <opencv2/opencv.hpp>

namespace vision {
namespace features {
namespace global {


cv::Vec3b ColorHist::index_to_color(const int index, int dim ) {
  int s = 256 / dim;
  cv::Vec3b c;
  c(0) = index / (dim*dim);
  c(1) = (index / dim ) % dim;
  c(2) = index % dim;

  c(0) *= s;
  c(1) *= s;
  c(2) *= s;
  return c;
}

int ColorHist::color_to_index(const cv::Vec3b color, int dim ) {
  int s = 256 / dim;
  int index =  dim * dim* (color(0)/s) +
               dim * (color(1)/s) +
               (color(2)/s);
  return index;
}

void ColorHist::color_to_hist(const cv::Vec3b color,
    std::vector<unsigned short>& hist,
    int dim){
  int s = 256 / dim;
  int n_dim = static_cast<int>( pow(double(dim),3.0) );

  cv::Vec3b color_scaled;
  color_scaled(0) = color(0)/s;
  color_scaled(1) = color(1)/s;
  color_scaled(2) = color(2)/s;
  hist.resize(n_dim,0);

  for(int r=-1; r< 2; r++) {
    for(int g=-1; g< 2; g++) {
      for(int b=-1; b< 2; b++) {

        cv::Vec3f c = color_scaled;
        c(0) += r;
        c(1) += g;
        c(2) += b;
        if(c(0) >= 0 && c(0) < dim && c(1) >= 0 && c(1) < dim && c(2) >= 0 && c(2) < dim ) {
          int index =  dim * dim* c(0)+
                       dim * c(1)+
                       c(2);
          hist[index] += 1;
        }
      }
    }
  }

  int index =  dim * dim* color_scaled(0)+
               dim * color_scaled(1)+
               color_scaled(2);
  hist[index] += 5;

}



void ColorHist::extract_hist(const cv::Mat_<cv::Vec3b>& colors_big_image,
    cv::Mat_<int>& hist,
    int dim,
    cv::Mat_<uchar> mask) {
  float sigma = .1;
  int c = 20;
  int min_size = 10;
  cv::Mat_<int32_t> labels;

  // get segmentation
  int num_components = awesomeness::third_party::felz_pix::segment_image(colors_big_image, sigma, c, min_size, &labels);

  // compute avg colors and size of regions
  std::vector<int> size(num_components,0);
  std::vector<cv::Vec3f> mean_colors(num_components,cv::Vec3f(0.0,0.0,0.0));


  if(mask.data) {
    for (int y = 0; y < colors_big_image.rows; y++) {
      for (int x = 0; x < colors_big_image.cols; x++) {
        if(mask.at<uchar>(y,x) > 0) {
          const int segment_id = labels.at<int32_t>(y,x);
          const cv::Vec3b& color = colors_big_image.at<cv::Vec3b>(y,x);
          size[segment_id] ++;
          mean_colors[segment_id] += color;
        }
      }
    }
  }else{
    for (int y = 0; y < colors_big_image.rows; y++) {
      for (int x = 0; x < colors_big_image.cols; x++) {

        const int segment_id = labels.at<int32_t>(y,x);
        const cv::Vec3b& color = colors_big_image.at<cv::Vec3b>(y,x);
        size[segment_id] ++;
        mean_colors[segment_id] += color;

      }
    }
  }

//  std::vector<int> background_component_id;
//  background_component_id.push_back( labels.at<int32_t>(0,0) );
//  //background_component_id.push_back( labels.at<int32_t>(0,colors_big_image.cols-1) );


//  cv::Mat_<uchar> mask(colors_big_image.rows, colors_big_image.cols ); // segmentation result (4 possible values)
//  mask.setTo(cv::Scalar(cv::GC_PR_FGD));
//  cv::Mat_<cv::Vec3b> mean_color = colors_big_image.clone();
//  for (int y = 0; y < colors_big_image.rows; y++) {
//    for (int x = 0; x < colors_big_image.cols; x++) {
//      const int segment_id = labels.at<int32_t>(y,x);
//
//      u_char r =  mean_colors[segment_id](0) / size[segment_id];
//      u_char g =  mean_colors[segment_id](1) / size[segment_id];
//      u_char b =  mean_colors[segment_id](2) / size[segment_id];
//      if(segment_id == background_component_id[0]) {
//        r = 255;
//        g = 0;
//        b = 255;
//        mask.at<uchar>(y,x) = cv::GC_BGD;
//      }
//      mean_color.at<cv::Vec3b>(y,x)(0) = r;
//      mean_color.at<cv::Vec3b>(y,x)(1) = g;
//      mean_color.at<cv::Vec3b>(y,x)(2) = b;
//    }
//  }
//
//  cv::Rect fg_mask(colors_big_image.cols/4,
//      colors_big_image.rows/4,
//      colors_big_image.cols/2,
//      colors_big_image.rows/2);
//
//
//
//
//
//  // Get the pixels marked as likely foreground
//  cv::compare(mask,cv::GC_PR_FGD,mask,cv::CMP_EQ);
//  // Generate output image
//  cv::Mat foreground(colors_big_image.size(),CV_8UC3,cv::Scalar(255,0,255));
//  colors_big_image.copyTo(foreground,mask); // bg pixels not copied
//
//  mean_color(fg_mask).setTo( cv::Vec3b(0,255,0) );
//  cv::imshow("X", mean_color);
//  cv::imshow("org", colors_big_image);
//  cv::imshow("foreground", foreground);
//
//  cv::imshow("mask", mask);
//  cv::waitKey(0);
//



  // calculate hist
  int n_dim = static_cast<int>( pow(double(dim),3.0) );
  hist = cv::Mat_<int>(1,n_dim, 0.0 );

  for(int i=0; i < num_components; i++) {

    if(size[i] > 0 ) {
      cv::Vec3b c;
      c(0) = mean_colors[i](0) / size[i];
      c(1) = mean_colors[i](1) / size[i];
      c(2) = mean_colors[i](2) / size[i];

      int index = color_to_index( c, dim );


      CHECK_LT(index, n_dim);
      hist(index) += size[i];
    }
  }
}

} /* namespace global */
} /* namespace features */
} /* namespace vision */
