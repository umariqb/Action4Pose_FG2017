/*
 * test_felz_pix.cpp
 *
 *  Created on: Aug 7, 2013
 *      Author: lbossard
 */


#include "cpp/third_party/gtest/gtest.h"

#include "felz_pix.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cpp/vision/image_utils.hpp"

const cv::Mat colors_big_image = cv::imread("src/cpp/vision/features/test/colors_big.bmp");



TEST(FelzPixTest, SimpleTest){
  float sigma = .1;
  int c = 20;
  int min_size = 10;
  cv::Mat_<int32_t> labels;
  int num_components = awesomeness::third_party::felz_pix::segment_image(colors_big_image, sigma, c, min_size, &labels);
  ASSERT_EQ(num_components,9);
  ASSERT_EQ(cv::countNonZero(labels > 8), 0);
//  cv::Mat_<uchar> showlabels = labels * (int)(255./num_components);
//  cv::imshow("rgb", colors_big_image);
//  cv::imshow("abgr", showlabels);
//  cv::waitKey();
}

