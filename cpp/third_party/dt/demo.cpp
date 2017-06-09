/*
 * demo.cpp
 *
 *  Created on: Nov 4, 2012
 *      Author: mdantone
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "pnmfile.h"
#include "imconv.h"
#include "dt.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

// creates only header
template <class T>
void image_to_mat(const image<T>& img, cv::Mat_<T>& mat ){
  mat = cv::Mat_<T>(img.height(), img.width(), img.data);
}


// makes a hard copy
template <class T>
image<T>* mat_to_image(const cv::Mat_<T>& mat_){
  cv::Mat mat = mat_;
  if (!mat_.isContinuous()) {
      mat = mat_.clone();
  }
  int width = mat.cols;
  int height = mat.rows;
  image<T>* img = new image<T>(width,height,false);
  memcpy(img->data,mat.data, width*height*sizeof(float));
  return img;
}

int main(int argc, char **argv) {

  string input_name = "/home/mdantone/git-repos/awesomeness/src/cpp/third_party/dt/input.pbm";

//  input_name = "/home/mdantone/Desktop/221237_465204466857559_360867203_o.jpg";


  // load input
  image<uchar>* input = loadPBM(input_name.c_str());
//  cv::Mat_<uchar> m = cv::imread(input_name,0);
//  m = 255-m;
//  m /= 255;
//  input = mat_to_image(m);


  // compute dt
  image<float> *out = dt(input);


  image<float> *input2 = new image<float>(input->width(), input->height(), false);
//  for (int y = 0; y < input->height(); y++) {
//    for (int x = 0; x < input->width(); x++) {
//      cout << int(imRef(input, x, y)) <<" ";
//      if (imRef(input, x, y) == 1) {
//        imRef(input2, x, y) = 1;
//      }else {
//        imRef(input2, x, y) = 0;
//      }
//    }
//    cout << endl;
//  }
  dt(input2);

  for (int y = 0; y < input->height(); y++) {
    for (int x = 0; x < input->width(); x++) {
      cout << int(imRef(input2, x, y)) <<" ";
    }
    cout << endl;
  }

  Mat_<uchar> m3;
  image<uchar> *u = imageFLOATtoUCHAR(input2);
  image_to_mat(*u, m3);
  imshow("output",m3);
  waitKey(0);

  return 0;
  // take square roots
//  for (int y = 0; y < out->height(); y++) {
//    for (int x = 0; x < out->width(); x++) {
//      imRef(out, x, y) = sqrt(imRef(out, x, y));
//    }
//  }





  // convert to grayscale
  image<uchar> *gray = imageFLOATtoUCHAR(out);
  cv::Mat_<float> m2;
  image_to_mat(*out, m2);

  cv::normalize(m2, m2, 0, 1, CV_MINMAX);

//  m2 *= 255;
  imshow("output",m2);
  waitKey(0);


  // save output
  savePGM(gray, "/home/mdantone/Desktop/test.png");

  delete input;
  delete out;
  delete gray;
  return 0;
}

