/*
 * sample_patches.cpp
 *
 *  Created on: Aug 1, 2013
 *      Author: mdantone
 */

#include "sample_patches.hpp"

#include <iostream>
#include <glog/logging.h>
#include <boost/heap/priority_queue.hpp>
#include <boost/random.hpp>

#include "cpp/vision/geometry_utils.hpp"
#include "cpp/vision/opencv_utils.hpp"

#include "cpp/utils/system_utils.hpp"
#include <boost/thread/thread.hpp>
#include "cpp/utils/thread_pool.hpp"
#include <boost/bind.hpp>


using namespace std;
using namespace cv;

typedef pair<int,Rect> HardNeg;
typedef pair<double, HardNeg> HardNegElementQueue;

namespace learning {
namespace common {


template< typename FirstType, typename SecondType >
struct PairComparator {
  bool operator()( const pair<FirstType, SecondType>& p1, const pair<FirstType, SecondType>& p2 ) const {
    return( p1.first < p2.first );
  }
};

void sample_hard_negatives(vector<Mat_<float> >& scores,
                           int num_samples,
                           vector<Rect> mask,
                           cv::Rect rect_size,
                           vector<HardNegElementQueue>& hard_negatives) {
  if(scores.size() == 0) {
    return;
  }

  CHECK_EQ(scores.size(), mask.size());
  for(int i=0; i < scores.size(); i++) {
    Rect rect_scores = Rect(0,0, scores[i].cols, scores[i].rows);
    Rect inter = vision::geometry_utils::intersect(rect_scores, mask[i]);
    scores[i](inter).setTo(Scalar(-(rect_size.width*rect_size.height)));
  }
  sample_hard_negatives(scores, num_samples, rect_size, hard_negatives, 0);
}




void non_max_supression( Mat& score,
    int n_patches_per_image,
    Rect rect_size,
    vector< pair<float, Rect> >& peaks ) {

  Rect rect_scores = Rect(0,0, score.cols, score.rows);
  for(int j=0; j < n_patches_per_image; j++) {
    // get max
    Point max; double max_v;
    minMaxLoc(score, 0, &max_v, 0, &max);

    // no maximus available anymore
    if(max_v < 0)
      break;

    // non max suppression
    Rect box(max.x - rect_size.width/2,
             max.y - rect_size.height/2,
             rect_size.width, rect_size.height);

    Rect inter = vision::geometry_utils::intersect(box, rect_scores);
    if(inter.height == rect_size.height && inter.width == rect_size.width) {
      float avg_value =  sum(score(box))[0] / (rect_size.width*rect_size.height);
      int s_width = rect_size.width *.5 ;
      int s_height = rect_size.height *.5 ;

      Rect small_bbox = Rect(max.x - s_width/2, max.y - s_height/2, s_width, s_height );
      score(small_bbox).setTo(Scalar(0));

      if( avg_value > 0 ) {
        peaks.push_back( make_pair(max_v, box));
      }
    }else{
      // TODO
      score(inter).setTo(Scalar(-1));
    }
    score.at<float>(max) = -1;
  }

}

void sample_hard_negatives_mt(vector<Mat_<float> >& scores,
                           int num_samples,
                           cv::Rect rect_size,
                           vector<HardNegElementQueue>& hard_negatives,
                           int save) {
  if(scores.size() == 0) {
    return;
  }
  LOG(INFO) << "sampling hard round: " << save;
  int n_patches_per_image = (num_samples) / static_cast<float>(scores.size()) * 10;

  std::vector<HardNegElementQueue> queue;
  vector<vector< pair<float, Rect> > > peaks(scores.size());


  int num_threads = utils::system::get_available_logical_cpus();
  if(num_threads > 1){
    boost::thread_pool::executor e(num_threads);
    for(unsigned int i=0; i < scores.size(); i++) {
      e.submit(boost::bind(&non_max_supression, boost::ref(scores[i]), n_patches_per_image,
          rect_size, boost::ref(peaks[i])));
    }
    e.join_all();
  }else{
    for(int i=0; i < scores.size(); i++) {
      non_max_supression(scores[i], n_patches_per_image, rect_size, peaks[i]);
    }
  }

  // combining
  for(int i=0; i < peaks.size(); i++) {
    const vector< pair<float, Rect> >& peaks_per_image = peaks[i];
    for(int j=0; j < peaks_per_image.size(); j++) {
      float max_v = peaks_per_image[j].first;
      Rect bbox = peaks_per_image[j].second;
      HardNeg n = make_pair(i, bbox);
      HardNegElementQueue ne = make_pair(max_v, n);
      hard_negatives.push_back(ne);
    }
  }

  std::sort(hard_negatives.begin(), hard_negatives.end(), PairComparator<double,HardNeg>() );
  if(hard_negatives.size() < num_samples && save < 2 ) {
    LOG(INFO) << "hard neg founds: " << hard_negatives.size()  << ", nedded: " <<  num_samples;
    sample_hard_negatives(scores,num_samples, rect_size, hard_negatives, save +1);
  }
}

void sample_hard_negatives(vector<Mat_<float> >& scores,
                           int num_samples,
                           cv::Rect rect_size,
                           vector<HardNegElementQueue>& hard_negatives,
                           int save) {

  if(scores.size() == 0) {
    return;
  }

  LOG(INFO) << "sampling hard round: " << save;

  double sigma = 5;
  cv::Mat kernelX = cv::getGaussianKernel(rect_size.width, sigma, CV_32FC1);
  cv::Mat kernelY = cv::getGaussianKernel(rect_size.height, sigma, CV_32FC1);
  cv::Mat kernel = kernelY * kernelX.t();
  cv::normalize(kernel, kernel, 0, 0.25, CV_MINMAX);


  kernel *= -1;
  int n_patches_per_image = (num_samples) / static_cast<float>(scores.size()) * 10;

  std::vector<HardNegElementQueue> queue;


  for(int i=0; i < scores.size(); i++) {
    Mat& score = scores[i];

    Rect rect_scores = Rect(0,0, score.cols, score.rows);
    for(int j=0; j < n_patches_per_image; j++) {

      Mat plot = score.clone();

      // get max
      Point max;
      double max_v;
      minMaxLoc(score, 0, &max_v, 0, &max);

      // no maximus available anymore
      if(max_v < 0)
        break;



      // non max suppression
      Rect box(max.x - rect_size.width/2,
               max.y - rect_size.height/2,
               rect_size.width, rect_size.height);


//      cv::circle(plot, max, 3, cv::Scalar(255, 255, 255, 0), -1);
//      cv::rectangle(plot, box, cv::Scalar(255, 255, 255, 0));
//      cv::imshow("score", plot);
//      cv::imshow("kernel", kernel);
//      cv::waitKey(0);


      Rect inter = vision::geometry_utils::intersect(box, rect_scores);
      if(inter.height == rect_size.height && inter.width == rect_size.width) {

        float avg_value =  sum(score(box))[0] / (rect_size.width*rect_size.height);

//        add(score(box), kernel, score(box));
        int s_width = rect_size.width *.5 ; //std::min(6, rect_size.width);
        int s_height = rect_size.height *.5 ; //std::min(6, rect_size.height);

        Rect small_bbox = Rect(max.x - s_width/2, max.y - s_height/2, s_width, s_height );
        score(small_bbox).setTo(Scalar(0));

        if( avg_value > 0 ) {
          HardNeg n = make_pair(i, box);
          HardNegElementQueue ne = make_pair(max_v, n);
          hard_negatives.push_back(ne);
        }
      }else{
        // TODO
        score(inter).setTo(Scalar(-1));
      }
      score.at<float>(max) = -1;

    }
  }

  std::sort(hard_negatives.begin(), hard_negatives.end(), PairComparator<double,HardNeg>() );

  if(hard_negatives.size() < num_samples && save < 2 ) {
    LOG(INFO) << "hard neg founds: " << hard_negatives.size()  << ", nedded: " <<  num_samples;
    sample_hard_negatives(scores,num_samples, rect_size, hard_negatives, save +1);
  }

}

void sample_rectangles_outside_roi(const Mat image,
                     const Rect roi,
                     int num_samples ,
                     boost::mt19937* rng,
                     vector<Rect>& rectangles,
                     int max_iterations) {

  int width = image.cols;
  int height = image.rows;
  if(( width - roi.width - 1) <= 0)
    return;
  if(( height - roi.height - 1) <= 0)
    return;

  boost::uniform_int<> dist_x(0, width - roi.width - 1);
  boost::uniform_int<> dist_y(0, height - roi.height - 1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(*rng, dist_x);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(*rng, dist_y);

  int count = 0;
  while(count < max_iterations && rectangles.size() < num_samples) {
    count ++;
    Rect rect = Rect(rand_x(),rand_y(), roi.width, roi.height);

    Rect inter = vision::geometry_utils::intersect(rect, roi);
    if(inter.width != 0 || inter.height != 0) {
      continue;
    }

    if(vision::opencv_utils::check_uniqueness(rectangles, rect) ) {
      rectangles.push_back(rect);
    }
  }
}

void sample_rectangles_around_roi(const Mat image,
                     const Rect roi,
                     int num_samples,
                     boost::mt19937* rng,
                     vector<Rect>& rectangles,
                     const float sigma,
                     int max_iterations){
  boost::normal_distribution<> nd(0.0, sigma);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > rand_gauss(*rng, nd);

  const Point anker_point(roi.x, roi.y);
  int count = 0;
  Rect img_rect(0,0, image.cols, image.rows);

  while(count < max_iterations && rectangles.size() < num_samples) {
    count ++;

    Rect rect = roi;
    if(rectangles.size() != 0 && count != 0) {
      rect.x += rand_gauss();
      rect.y += rand_gauss();
    }

    Rect inter = vision::geometry_utils::intersect(rect, img_rect);
    if(inter.width != roi.width || inter.height != roi.height) {
      continue;
    }


    if(vision::opencv_utils::check_uniqueness(rectangles, rect) ) {
      rectangles.push_back(rect);
    }
  }
}




} /* namespace common */
} /* namespace learning */
