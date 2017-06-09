/*
 * demo.cpp
 *
 *  Created on: Jan 21, 2013
 *      Author: mdantone
 */


#include "DistanceTransform.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {


  Mat_<float> score_in, score_dt;
  Mat img = imread("/home/mdantone/git-repos/awesomeness/src/cpp/third_party/PartBasedDetector/data/map.jpg", -1);


  img.convertTo(score_in, CV_32F,1/255.0);
  Mat_<int> Ix_dt, Iy_dt;

  Quadratic fx(-0.01, -0.01);
  Quadratic fy(-0.01, -0.01);

  Point anchor(0,0);
  DistanceTransform<float> dt_;

  dt_.compute(score_in, fx, fy, anchor, score_dt, Ix_dt, Iy_dt);


  double min_v,max_v;
  cv::Point max_loc, max_before_dt;

  minMaxLoc(score_in, 0, 0, 0, &max_before_dt);
  minMaxLoc(score_dt, &min_v, &max_v, 0, &max_loc);
  cout << "min: " << min_v << ", max: " << max_v << endl;


  cout << max_before_dt.x  << " " << max_before_dt.y << endl;
  cout << max_loc.x  << " " << max_loc.y << endl;



  cv::normalize(score_dt, score_dt, 0, 1, CV_MINMAX);

  minMaxLoc(score_dt, &min_v, &max_v, 0, 0);
  cout << "min: " << min_v << ", max: " << max_v << endl;

  cv::imshow("score_in", score_in);
  cv::imshow("score_dt", score_dt);

  cv::waitKey(0);

  return 0;
}
