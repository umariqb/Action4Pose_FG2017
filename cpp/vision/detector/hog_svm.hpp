/*
 * hog_svm.hpp
 *
 *  Created on: Aug 21, 2013
 *      Author: mdantone
 */

#ifndef HOG_SVM_HPP_
#define HOG_SVM_HPP_

#include <opencv2/opencv.hpp>

#include "cpp/learning/svm/liblinear_svm.hpp"
#include "cpp/learning/svm/liblinear_svm_problem.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include <boost/serialization/split_member.hpp>

namespace awesomeness {
namespace vision {
namespace detector {



class HoGSVM {
public:
  HoGSVM( const cv::Size& win_size = cv::Size(32,32),
          const cv::Size& hog_blockSize = cv::Size(16,16),
          const cv::Size hog_blockStride = cv::Size(8,8),
          const cv::Size hog_cellSize = cv::Size(8,8),
          const int hog_nbins = 9
          );



  cv::Size get_win_size() const { return hog_win_size_; }
  int get_win_width() const { return hog_win_size_.width; }
  int get_win_height() const { return hog_win_size_.height; }


  double predict( const cv::Mat& image, cv::Point location);

  void detect( const cv::Mat& image, cv::Mat_<float>& score ) const;
  void detect_ptr( const cv::Mat* image, cv::Mat_<float>* score ) const;

  void set_weights(const std::vector<double>& weights);
  void get_weights(std::vector<float>& weights);

  void display_weights();

  ~HoGSVM() {} ;

private:

  void set_svm_detector(const std::vector<double> & weights) {
    std::vector<float> weights_f;
    for(int i=0; i < weights.size(); i++) {
      weights_f.push_back( (weights[i]) );
    }
    hog_extractor_.setSVMDetector(weights_f);
  }

  cv::Size hog_win_size_;
  cv::Size hog_blockSize_;
  cv::Size hog_blockStride_;
  cv::Size hog_cellSize_;
  int hog_nbins_;

  std::vector<double> weights_;

  cv::HOGDescriptor hog_extractor_;


  friend class boost::serialization::access;
  template <class Archive>
  void load(Archive& archive, const unsigned int version);

  template <class Archive>
  void save(Archive& archive, const unsigned int version) const;

  BOOST_SERIALIZATION_SPLIT_MEMBER();

};



class HoGSVMTrainer {
public:
  HoGSVMTrainer( const cv::Size& win_size,
      awesomeness::learning::svm::LibLinearSvmParameters param = awesomeness::learning::svm::LibLinearSvmParameters(),
      int num_samples = 100);

  void push( const cv::Mat& image, cv::Point location, int label);
  void push( const cv::Mat& image, std::vector<cv::Point>& locations, int label);

  HoGSVM train(bool use_weights = false);

  cv::Size get_win_size() const { return hog_win_size_; }
  int get_win_width() const { return hog_win_size_.width; }
  int get_win_height() const { return hog_win_size_.height; }

  ~HoGSVMTrainer(){};

protected:

  cv::Size hog_win_size_;
  cv::Size hog_blockSize_;
  cv::Size hog_blockStride_;
  cv::Size hog_cellSize_;
  int hog_nbins_;

  cv::HOGDescriptor hog_extractor_;

  int n_neg_sampels;
  int n_pos_sampels;

  bool clean_vector(const cv::Mat& image, std::vector<cv::Point>& locations );

  boost::scoped_ptr<awesomeness::learning::svm::LibLinearSvmProblem> svm_problem_;
  boost::scoped_ptr<awesomeness::learning::svm::LibLinearSvm> svm_;



};


class HoGSVMBootstrap : public HoGSVMTrainer {
public:
  HoGSVMBootstrap( const cv::Size& win_size,
      awesomeness::learning::svm::LibLinearSvmParameters param = awesomeness::learning::svm::LibLinearSvmParameters(),
      int num_samples = 100) : HoGSVMTrainer(win_size, param, num_samples) {

    LOG(INFO) << "param.solver_type: " << param.solver_type;
  };

  HoGSVM bootstrap(const std::vector<cv::Mat>& images,
      std::vector<cv::Rect>& rois, int n_rounds = 5 );


};




//------------------------------------------------------------------------------
template <class Archive>
void HoGSVM::load(Archive& ar, const unsigned int version)
{
  ar & hog_blockSize_;
  ar & hog_blockStride_;
  ar & hog_cellSize_;
  ar & hog_nbins_;
  ar & hog_win_size_;
  ar & weights_;
  hog_extractor_ = cv::HOGDescriptor(hog_win_size_, hog_blockSize_,
                                     hog_blockStride_, hog_cellSize_,
                                     hog_nbins_);

  if(weights_.size() > 0) {
    set_svm_detector(weights_);
  }
}


template <class Archive>
void HoGSVM::save(Archive& ar, const unsigned int version) const
{
  ar & hog_blockSize_;
  ar & hog_blockStride_;
  ar & hog_cellSize_;
  ar & hog_nbins_;
  ar & hog_win_size_;
  ar & weights_;
}


} /* namespace detector */
} /* namespace vision */
} /* namespace awesomeness */
#endif /* HOG_SVM_HPP_ */
