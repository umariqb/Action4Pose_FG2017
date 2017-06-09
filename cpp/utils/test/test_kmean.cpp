/*
 * test_armadillo.cpp
 *
 *  Created on: Sep 6, 2013
 *      Author: lbossard
 */



#include "cpp/third_party/gtest/gtest.h"
#include "cpp/utils/vl_kmean.hpp"
#include <opencv2/opencv.hpp>


TEST(VL_KMEAN, vocabulary_assignment) {

  int descriptor_dimension = 10;
  int num_sampels = 10;
  int voc_size= 2;
  cv::Mat_<float> features = cv::Mat_<int>::zeros(num_sampels, descriptor_dimension);

  for(int i=0; i < num_sampels; i++) {
    features.row(i).setTo(i);
  }

  cv::Mat_<float> vocabulary;
  std::vector<uint32_t> labels;
  std::vector<float> distances;

  utils::kmean::vl_kmean(features, voc_size, vocabulary, labels, distances);



  cv::Mat dists;
  cv::Mat_<int> indices;
  cv::flann::SearchParams p;
  p.setAlgorithm(cvflann::FLANN_DIST_EUCLIDEAN);
  cv::flann::Index index(vocabulary, p);
  index.knnSearch(features, indices, dists, 1, p);

  ASSERT_EQ(num_sampels, labels.size());
  for (int d = 0 ; d < labels.size() ; ++d) {
    std::cout << d << ": " << indices(d,0) << " " << labels[d] << std::endl;
    std::cout << d << ": " << dists.at<float>(d,0) << " " << distances[d] << std::endl;
    ASSERT_EQ(indices(d,0), labels[d]);
    ASSERT_EQ(dists.at<float>(d,0), distances[d]);

  }
}

TEST(VL_KEMANS, quantisation){

  int num_samples = 20;
  int num_dims = 3;
  cv::Mat_<float> points(num_samples, num_dims);
  points = 0;

  // values of 1st half of data set is set to 10
  //change the values of 2nd half of the data set; i.e. set it to 20
  for (int r = 0; r < points.rows/2; r++) {
    points(r, 0) = 0;
    points(r,1) = r / (points.rows/2);
  }
  for (int r = points.rows/2; r < points.rows; r++) {
    points(r, 0) = 3;
    points(r,1) =  3 + r / (points.rows/2);
  }

  cv::Mat_<float> voc;
  std::vector<float> distances;
  std::vector<unsigned int> labels;
  utils::kmean::vl_kmean(points, /*num_clusters*/ 3, voc, labels, distances, /*num_iterations*/ 5);
  ASSERT_EQ(labels.size(), num_samples);
  ASSERT_EQ(distances.size(), num_samples);

  int first_label = labels[0];
  int last_label = labels[labels.size() - 1];
  ASSERT_NE(first_label, last_label);

  //
  for (int i = 0; i < num_samples/2; ++i){
    ASSERT_EQ(labels[i], first_label);
  }
  for (int i =  num_samples/2; i < num_samples; ++i){
    ASSERT_EQ(labels[i], last_label);
  }



}
