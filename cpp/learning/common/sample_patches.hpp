/*
 * sample_patches.hpp
 *
 *  Created on: Aug 1, 2013
 *      Author: mdantone
 */

#ifndef SAMPLE_PATCHES_HPP_
#define SAMPLE_PATCHES_HPP_

#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace learning {
namespace common {

void sample_hard_negatives(std::vector<cv::Mat_<float> >& scores,
                           int num_samples,
                           std::vector<cv::Rect> mask,
                           cv::Rect rect_size,
                           std::vector<std::pair<double, std::pair<int,cv::Rect> > >& hard_negatives);

void sample_hard_negatives(std::vector<cv::Mat_<float> >& scores,
                           int num_samples,
                           cv::Rect rect_size,
                           std::vector<std::pair<double, std::pair<int,cv::Rect> > >& hard_negatives,
                           int save = 0);


void sample_hard_negatives_mt(std::vector<cv::Mat_<float> >& scores,
                           int num_samples,
                           cv::Rect rect_size,
                           std::vector<std::pair<double, std::pair<int,cv::Rect> > >& hard_negatives,
                           int save = 0);

void sample_rectangles_outside_roi(const cv::Mat image,
                      const cv::Rect roi,
                      int num_samples ,
                      boost::mt19937* rng,
                      std::vector<cv::Rect>& rectangles,
                      int max_iterations = 50000);

void sample_rectangles_around_roi(const cv::Mat images,
                      const cv::Rect roi,
                      int num_samples,
                      boost::mt19937* rng,
                      std::vector<cv::Rect>& rectangles,
                      const float sigma = 1.0,
                      int max_iterations = 50000);

void inline asd_test() {};


} /* namespace common */
} /* namespace learning */
#endif /* SAMPLE_PATCHES_HPP_ */
