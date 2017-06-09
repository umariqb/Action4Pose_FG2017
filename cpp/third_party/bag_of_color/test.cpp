#include <iostream>

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

const float Xn = 0.950456f;
const float Zn = 1.08875f;
const float threshold = 0.008856f;

#define f(t) (((t)>threshold)?pow(t, 1.0f/3.0f):7.787f*t+16.0f/116.0f)

int main(int argc, char** argv) {
  cv::Mat BGR_image;
  BGR_image = cv::imread(argv[1], 1);

  //Convert to float and scale
  cv::Mat BGR_imagef;
  cv::Vec3f BGR_pixel;
  BGR_image.convertTo(BGR_imagef, CV_32F);
  BGR_imagef *= 1./255.0;
  cout << "BGR values as float:" << endl;
  for (int i = 0; i < BGR_image.rows; i++) {
    for (int j = 0; j < BGR_image.cols; j++) {
      BGR_pixel = BGR_imagef.at<cv::Vec3f>(i, j);
      cout << "(" << i << "," << j << "): [" << BGR_pixel[0] << "," << BGR_pixel[1] << "," << BGR_pixel[2] << "]" << endl;
    }
  }

  // XYZ
  cv::Mat XYZ_image(BGR_imagef.size(), CV_32FC3);
  cv::Vec3f XYZ_pixel;
  cv::cvtColor(BGR_imagef, XYZ_image, CV_BGR2XYZ);
  cout << "XYZ values:" << endl;
  for (int i = 0; i < BGR_image.rows; i++) {
    for (int j = 0; j < BGR_image.cols; j++) {
      XYZ_pixel = XYZ_image.at<cv::Vec3f>(i, j);
      cout << "(" << i << "," << j << "): [" << XYZ_pixel[0] << "," << XYZ_pixel[1] << "," << XYZ_pixel[2] << "]" << endl;
    }
  }
  cout << "Lab values (manual):" << endl;
  double X, Y, Z, L, a, b;
  for (int i = 0; i < BGR_image.rows; i++) {
    for (int j = 0; j < BGR_image.cols; j++) {
      XYZ_pixel = XYZ_image.at<cv::Vec3f>(i, j);
      X = XYZ_pixel[0]/Xn;
      Y = XYZ_pixel[1];
      Z = XYZ_pixel[2]/Zn;
      L = (Y>threshold)?116.0f*pow(Y, 1.0f/3.0f)-16.0:903.3f*Y;
      a = 500.0f*(f(X) - f(Y));
      b = 200.0f*(f(Y) - f(Z));
      cout << "(" << i << "," << j << "): [" << L << "," << a << "," << b << "]" << endl;
    }
  }

  // Lab
  cv::Mat Lab_image(BGR_imagef.size(), CV_32FC3);
  cv::Vec3f Lab_pixel;
  cv::cvtColor(BGR_imagef, Lab_image, CV_BGR2Lab);
  cout << "Lab values (OpenCV):" << endl;
  for (int i = 0; i < BGR_image.rows; i++) {
    for (int j = 0; j < BGR_image.cols; j++) {
      Lab_pixel = Lab_image.at<cv::Vec3f>(i, j);
      cout << "(" << i << "," << j << "): [" << Lab_pixel[0] << "," << Lab_pixel[1] << "," << Lab_pixel[2] << "]" << endl;
    }
  }
}

