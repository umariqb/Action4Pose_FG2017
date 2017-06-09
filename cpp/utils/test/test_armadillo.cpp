/*
 * test_armadillo.cpp
 *
 *  Created on: Sep 6, 2013
 *      Author: lbossard
 */



#include "cpp/third_party/gtest/gtest.h"


#include "cpp/utils/armadillo.hpp"


TEST(ArmadilloTest, opencv_to_arma_same_type) {
  arma::fmat arma_mat = arma::randu<arma::fmat>(5, 6);

  cv::Mat_<float> cv_mat;
  utils::armadillo::to_opencv(arma_mat, & cv_mat);

  ASSERT_EQ(arma_mat.n_rows, cv_mat.rows);
  ASSERT_EQ(arma_mat.n_cols, cv_mat.cols);

  for (int r = 0; r < arma_mat.n_rows; ++r){
    for (int c = 0; c < arma_mat.n_cols; ++c){
     ASSERT_EQ(arma_mat(r,c), cv_mat(r,c));
    }
  }
}

TEST(ArmadilloTest, opencv_to_arma_type_change) {
  arma::fmat arma_mat = arma::randu<arma::fmat>(5, 6);

  cv::Mat_<double> cv_mat;
  utils::armadillo::to_opencv(arma_mat, & cv_mat);

  ASSERT_EQ(arma_mat.n_rows, cv_mat.rows);
  ASSERT_EQ(arma_mat.n_cols, cv_mat.cols);

  for (int r = 0; r < arma_mat.n_rows; ++r){
    for (int c = 0; c < arma_mat.n_cols; ++c){
     ASSERT_NEAR(arma_mat(r,c), cv_mat(r,c), .0001);
    }
  }
}

TEST(ArmadilloTest, arma_to_opencv_same_type) {
  cv::Mat_<float> cv_mat(5,6);
   cv::randu(cv_mat, -10, 10);
   std::cout << cv_mat;
   arma::fmat arma_mat;
   utils::armadillo::from_opencv(cv_mat, &arma_mat);

   ASSERT_EQ(arma_mat.n_rows, cv_mat.rows);
   ASSERT_EQ(arma_mat.n_cols, cv_mat.cols);

   for (int r = 0; r < arma_mat.n_rows; ++r){
     for (int c = 0; c < arma_mat.n_cols; ++c){
      ASSERT_NEAR(arma_mat(r,c), cv_mat(r,c), .0001);
     }
   }
}

TEST(ArmadilloTest,arma_to_opencv_type_change) {
  cv::Mat_<double> cv_mat(5,6);
  cv::randu(cv_mat, -10, 10);

  arma::fmat arma_mat;
  utils::armadillo::from_opencv(cv_mat, &arma_mat);

  ASSERT_EQ(arma_mat.n_rows, cv_mat.rows);
  ASSERT_EQ(arma_mat.n_cols, cv_mat.cols);

  for (int r = 0; r < arma_mat.n_rows; ++r){
    for (int c = 0; c < arma_mat.n_cols; ++c){
     ASSERT_NEAR(arma_mat(r,c), cv_mat(r,c), .0001);
    }
  }
}

