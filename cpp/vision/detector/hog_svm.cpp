/*
 * hog_svm.cpp
 *
 *  Created on: Aug 21, 2013
 *      Author: mdantone
 */

#include "hog_svm.hpp"
#include "boost/random.hpp"


#include "cpp/utils/string_utils.hpp"
#include "cpp/utils/thread_pool.hpp"


#include "cpp/vision/features/hog_visualizer.hpp"
#include "cpp/vision/features/low_level/low_level_feature_extractor.hpp"

#include "cpp/utils/system_utils.hpp"
#include "cpp/vision/geometry_utils.hpp"

#include "cpp/learning/common/sample_patches.hpp"


using namespace cv;
using namespace std;
using namespace awesomeness::learning::svm;
using namespace learning::common;

namespace awesomeness {
namespace vision {
namespace detector {



HoGSVMTrainer::HoGSVMTrainer( const cv::Size& win_size,
                LibLinearSvmParameters param,
                int num_samples) :
                    hog_blockSize_(Size(16,16)),
                    hog_blockStride_(Size(8,8)),
                    hog_cellSize_(Size(8,8)),
                    hog_nbins_(9),
                    n_neg_sampels(0),
                    n_pos_sampels(0){

  // win-size needs to be a multiple of the blocksize
  int width = round(win_size.width / hog_blockSize_.width) * hog_blockSize_.width;
  int height = round(win_size.height / hog_blockSize_.height) * hog_blockSize_.height;
  hog_win_size_ = Size(width, height);

  CHECK_EQ((hog_win_size_.width - hog_blockSize_.width) % hog_blockStride_.width , 0);
  CHECK_EQ((hog_win_size_.height - hog_blockSize_.height) % hog_blockStride_.height, 0);
  hog_extractor_ = HOGDescriptor(hog_win_size_, hog_blockSize_, hog_blockStride_, hog_cellSize_, hog_nbins_);


  svm_problem_.reset(
      (LibLinearSvmProblem*)SvmProblem::create(
          param, 2, num_samples, hog_extractor_.getDescriptorSize()*num_samples));
}


bool HoGSVMTrainer::clean_vector(const cv::Mat& image, vector<cv::Point>& locations ) {


  for (vector<cv::Point>::iterator it=locations.begin();
                                   it!=locations.end(); ) {
    if(it->x < 0 || it->y < 0 ||
       it->x + get_win_width() > image.cols ||
       it->y + get_win_height() > image.rows )
    {
      it = locations.erase(it);
    }else{
      ++it;
    }
  }
  return locations.size() > 0;
}

void HoGSVMTrainer::push( const cv::Mat& image, vector<cv::Point>& locations, int label){

  if( clean_vector(image, locations) ) {
    vector<float> descriptors;
    hog_extractor_.compute(image, descriptors, cv::Size(), cv::Size(), locations);
    cv::Mat_<float> descriptor_mat(descriptors, false);

    descriptor_mat = descriptor_mat.reshape(0, locations.size());
    CHECK_EQ(descriptor_mat.cols, hog_extractor_.getDescriptorSize());
    CHECK_EQ(descriptor_mat.rows, locations.size());

    for(int i=0 ; i < locations.size(); i++) {
      svm_problem_->push_sample(descriptor_mat.row(i) , label);
    }
    if(label) {
      n_pos_sampels += locations.size();
    }else{
      n_neg_sampels += locations.size();
    }
  }
}

void HoGSVMTrainer::push( const cv::Mat& image, cv::Point location, int label){

  std::vector<cv::Point> locations;
  locations.push_back(location);

  push(image, locations, label);
}

HoGSVM HoGSVMTrainer::train( bool use_weights) {
  LOG(INFO) << "start training,  pos: " << n_pos_sampels << ", neg: " << n_neg_sampels;
  if(n_pos_sampels ==  0 || n_neg_sampels == 0) {

    HoGSVM hog_svm(hog_win_size_, hog_blockSize_, hog_blockStride_, hog_cellSize_, hog_nbins_ );
    std::vector<double> svm_weights(hog_extractor_.getDescriptorSize(), 0);
    hog_svm.set_weights( svm_weights );

    return hog_svm;
  }


  if(use_weights) {
    std::vector<int> weight_label(2);
    weight_label[0] = 0;
    weight_label[1] = 1;
    std::vector<double> weights(2);
    weights[0] = static_cast<double>(n_neg_sampels) / n_pos_sampels;
    weights[1] = 1.0;
    LOG(INFO) << "weights: 0:" << weights[0] << ", 1: " << weights[1];
    svm_.reset( (LibLinearSvm*)svm_problem_->train(weight_label, weights));
  }else{
    svm_.reset( (LibLinearSvm*)svm_problem_->train());
  }

  std::vector<double> svm_weights;
  svm_->get_weights(svm_weights);

  HoGSVM hog_svm(hog_win_size_, hog_blockSize_, hog_blockStride_, hog_cellSize_, hog_nbins_ );
  hog_svm.set_weights( svm_weights );
  return hog_svm;
}





void HoGSVM::display_weights() {
  Mat plot =  Mat(hog_win_size_, CV_8U, Scalar(0));

  std::vector<float> weights_f;
  for(int i=0; i < weights_.size(); i++) {
    weights_f.push_back( (weights_[i]+1) / 5.5  );
  }
  ::vision::features::visualize_hog_features(plot, weights_f);
}

double HoGSVM::predict( const cv::Mat& image, cv::Point location) {
  vector<float> descriptors;
  std::vector<cv::Point> locations;
  locations.push_back(location);
  hog_extractor_.compute(image, descriptors, cv::Size(), cv::Size(), locations);

  cv::Mat_<float> descriptor_mat(descriptors, false);

//  Mat p = image(Rect(location.x, location.y, get_win_width(), get_win_height())).clone();
//  ::vision::features::visualize_hog_features(p, descriptors);
//  display_weights();

  CHECK_EQ(descriptors.size(), weights_.size() );
  double result =  0;
  for(int i=0; i < descriptors.size(); i++) {
    result += (descriptors[i]*weights_[i]);
  }
  //svm_->predict(descriptor_mat, &values);
  return result;
}



HoGSVM::HoGSVM( const cv::Size& win_size ,
                const cv::Size& hog_blockSize,
                const cv::Size hog_blockStride,
                const cv::Size hog_cellSize,
                const int hog_nbins) :
                    hog_win_size_(win_size),
                    hog_blockSize_(hog_blockSize),
                    hog_blockStride_(hog_blockStride),
                    hog_cellSize_(hog_cellSize),
                    hog_nbins_(hog_nbins){

  CHECK_EQ((hog_win_size_.width - hog_blockSize_.width) % hog_blockStride_.width , 0);
  CHECK_EQ((hog_win_size_.height - hog_blockSize_.height) % hog_blockStride_.height, 0);
  hog_extractor_ = HOGDescriptor(hog_win_size_, hog_blockSize_, hog_blockStride_, hog_cellSize_, hog_nbins_);


}

void generateGridLocations(const cv::Size& image,
    const cv::Size& window_size,
    const cv::Size& grid_spacing,
    std::vector<cv::Point>& keypoints) {

  const int max_row = image.height - window_size.height;
  const int max_col = image.width - window_size.width;
  int points_w = std::ceil((static_cast<float>(image.width - window_size.width) ) / grid_spacing.width);
  int points_h = std::ceil((static_cast<float>(image.height - window_size.height) ) / grid_spacing.height);
  keypoints.clear();
  if (points_w < 1 || points_h < 1) {
      return;
  }
  keypoints.reserve(points_w * points_h);

  for (int row = 0 ; row < max_row ; row += grid_spacing.height)
  {
      for (int col = 0; col < max_col; col += grid_spacing.width)
      {
          keypoints.push_back(cv::Point(col, row));
      }
  }
}

void HoGSVM::set_weights(const std::vector<double>& weights) {
  weights_ = weights;
  set_svm_detector(weights);
}

void HoGSVM::get_weights(std::vector<float>& weights) {
  for(int i=0; i < weights_.size(); i++) {
    weights.push_back( static_cast<float>(weights_[i]) );
  }
}


void HoGSVM::detect_ptr( const cv::Mat* image,
                     cv::Mat_<float>* score ) const {
  detect(*image, *score);
}
void HoGSVM::detect( const cv::Mat& image,
                     cv::Mat_<float>& score ) const {

  std::vector<cv::Point> locations;

  // compute dense grid points
  locations.clear();
  generateGridLocations(
          image.size(),
          hog_win_size_,
          cv::Size(1,1),
          locations);
  if (locations.size() == 0) {
      return;
  }

  vector<float> descriptors;
  hog_extractor_.compute(image, descriptors, cv::Size(), cv::Size(), locations);
  cv::Mat_<float> descriptor_mat(descriptors, false);
  descriptor_mat = descriptor_mat.reshape(0, locations.size());
  CHECK_EQ(descriptor_mat.cols, hog_extractor_.getDescriptorSize());
  CHECK_EQ(descriptor_mat.rows, locations.size());


//  for(int i=0;i < locations.size(); i++) {
//    Mat plot = image.clone();
//    cv::rectangle(plot, Rect(locations[i].x, locations[i].y, get_win_width(), get_win_height()),
//            cv::Scalar(255, 0, 255, 0));
//    imshow("X", plot);
//    waitKey(0);
//  }

//  std::vector<float> weights_f;
//  for(int i=0; i < weights_.size(); i++) {
//    weights_f.push_back(weights_[i]);
//  }
//  cv::Mat_<float> weight_mat(weights_f, false);
//  weight_mat = weight_mat.reshape(0, 1);
//  CHECK_EQ(descriptor_mat.cols, weight_mat.cols);


  score = cv::Mat::zeros(image.rows, image.cols, cv::DataType<float>::type);
  for(int i=0; i < descriptor_mat.rows; i++) {

    double score_value = 0;
    for(int j=0; j < weights_.size(); j++) {
      score_value += ( descriptor_mat.at<float>(i,j) * weights_[j]);
    }
    Point center(locations[i].x + get_win_width()/2,
                 locations[i].y + get_win_height()/2);

    score_value = 1.0/(1.0+exp(-score_value));
    score(center) = static_cast<float>(score_value);

  }
//  imshow("X", score);
//  imshow("img", image);
//
//  waitKey(0);
}


HoGSVM HoGSVMBootstrap::bootstrap(const vector<Mat>& images,
    vector<Rect>& rois, int n_rounds ) {
  boost::mt19937 rng;



  // add the positive sampels
  for(int i=0; i < images.size(); i++) {
    push(images[i], Point(rois[i].x, rois[i].y), 1);
  }
  LOG(INFO) << "added positive samples";

  // adding random negatives
//  Rect roi(0,0, get_win_width(), get_win_height());
  for(int i=0; i < images.size(); i++) {
    int width = images[i].cols;
    int height = images[i].rows;
    if(( width - rois[i].width - 1) <= 0 || ( height - rois[i].height - 1) <= 0)
      continue;

    boost::uniform_int<> dist_x(0, width - rois[i].width - 1);
    boost::uniform_int<> dist_y(0, height - rois[i].height - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);

    for(int j=0; j < 100; j++) {
      Rect rect = Rect(rand_x(),rand_y(), rois[i].width, rois[i].height);
      Rect inter = ::vision::geometry_utils::intersect(rect, rois[i]);
      if(inter.width != 0 || inter.height != 0) {
        continue;
      }
      push(images[i], Point(rect.x, rect.y), 0);
      break;
    }
  }

  LOG(INFO) << "added negative samples";

  typedef pair<double, pair<int,Rect> > HardNeg;

  bool use_weights = true;

  LOG(INFO) << "start training";
  HoGSVM final_svm = train(use_weights);
  LOG(INFO) << "trained";

  for(int i_round=0; i_round < n_rounds; ++i_round) {

    // eval classifier
    LOG(INFO) << "start classification.";
    vector<Mat_<float> > scores(images.size());
    int num_threads = ::utils::system::get_available_logical_cpus();
    boost::thread_pool::executor e(num_threads);
    for(int i=0; i < images.size(); i++){
      e.submit(boost::bind( &HoGSVM::detect_ptr, &final_svm,  &images[i], &scores[i]));
    }
    e.join_all();

    LOG(INFO) << " sampling hard negatives .";
    vector<HardNeg> hard_negatives;
    ::learning::common::sample_hard_negatives(scores,
        images.size(),
        rois,
        rois[0],
        hard_negatives);

    for(int i=0; i < min(hard_negatives.size(), images.size()) ; i++) {
      int i_img = hard_negatives.back().second.first;
      Rect bbox = hard_negatives.back().second.second;
      push(images[i_img], Point(bbox.x, bbox.y), 0);
      hard_negatives.pop_back();
    }

    // retrain svm
    LOG(INFO) << "retrain after round " << i_round;

    final_svm = train(use_weights);

  }

  return final_svm;

}


} /* namespace detector */
} /* namespace vision */
} /* namespace awesomeness */
