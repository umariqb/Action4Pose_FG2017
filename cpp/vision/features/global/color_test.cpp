/*
 * bow_histogram_test.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: lbossard
 */




#include "cpp/third_party/gtest/gtest.h"

#include "cpp/vision/features/global/color_hist.hpp"


TEST(ColorTest, IndexMapping)
{
  std::vector<int> index;
  index.push_back(0);
  index.push_back(511);
  index.push_back(215);
  index.push_back(124);

  vision::features::global::ColorHist ch;

  for (unsigned int i = 0; i < index.size(); ++i) {
    cv::Vec3b color = ch.index_to_color( index[i]);
    int index_test = ch.color_to_index( color);
    ASSERT_TRUE(index_test == index[i]);
  }


  ASSERT_TRUE(ch.color_to_index( cv::Vec3b(0,0,0)) ==  0);
  ASSERT_TRUE(ch.color_to_index( cv::Vec3b(255,255,255)) == 511);
//
//  cv::Vec3b black = ch.index_to_rgb( 0);
//  ASSERT_TRUE(black(0) ==  0);
//  ASSERT_TRUE(black(1) ==  0);
//  ASSERT_TRUE(black(2) ==  0);
//
//  cv::Vec3b w = ch.index_to_rgb(511);
//  ASSERT_TRUE(w(0) ==  255);
//  ASSERT_TRUE(w(1) ==  255);
//  ASSERT_TRUE(w(2) ==  255);


}
