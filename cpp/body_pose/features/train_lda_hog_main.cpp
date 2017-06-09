/*
 * train_lda_trick_hog_main.cpp
 *
 *  Created on: Sep 4, 2013
 *      Author: lbossard
 */

#include <google/gflags.h>
#include <glog/logging.h>
#include <armadillo>

#include <opencv2/core/core.hpp>
#include <boost/random.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
namespace fs = boost::filesystem;

#include "cpp/utils/system_utils.hpp"
#include "cpp/utils/file_utils.hpp"
#include "cpp/utils/image_file_utils.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/utils/thread_pool.hpp"
#include "cpp/learning/lda_trick/online_covariance.hpp"
#include "cpp/learning/lda_trick/negative_model.hpp"

#include "cpp/utils/armadillo.hpp"
#include "cpp/vision/features/low_level_features.hpp"

#include "cpp/body_pose/common.hpp"

namespace vf = vision::features;
using namespace std;
using namespace cv;

#ifdef ARMA_USE_BLAS
extern "C" void openblas_set_num_threads(int num_threads);
#endif

HOGDescriptor get_hog_descriptor(int width, int height) {
  int checked_width = (width/16) * 16;
  int checked_height = (height/16) * 16;
  CHECK_GT(checked_width, 0);
  CHECK_GT(height, 0);
  const cv::Size hog_win_size = cv::Size( checked_width, checked_height);
  const cv::Size hog_blockSize = cv::Size(16,16);
  const cv::Size hog_blockStride = cv::Size(8,8);
  const cv::Size hog_cellSize = cv::Size(8,8);
  const int hog_nbins = 9;
  HOGDescriptor hog_extractor(hog_win_size, hog_blockSize, hog_blockStride, hog_cellSize, hog_nbins);
  return hog_extractor;
}

boost::mutex _mutex;
void push_hog_sample(
    const fs::path img_path,
    const int num_patches,
    const int patch_width,
    const int patch_height,
    awesomeness::learning::lda_trick::OnlineCovariance* online_covar){

  cv::Mat image = cv::imread(img_path.string());
  CHECK(image.data) << "could not read image " << img_path;

  // generate num_patches random rectangles
  boost::mt19937 rng;
  boost::uniform_int<> dist_x(0, image.cols - patch_width - 1);
  boost::uniform_int<> dist_y(0, image.rows - patch_height - 1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
  std::vector<Point> locations;
  for(int i=0; i < num_patches; i++) {
    locations.push_back( cv::Point(rand_x(),rand_y()) );
  }

  // extract descriptor
  HOGDescriptor hog_extractor = get_hog_descriptor(patch_width, patch_height);
  vector<float> descriptors;
  hog_extractor.compute(image, descriptors, cv::Size(), cv::Size(), locations);
  cv::Mat_<float> descriptor_mat(descriptors, false);
  descriptor_mat = descriptor_mat.reshape(0, locations.size());
  {
    for(int i=0; i < descriptor_mat.rows; i++) {
      boost::unique_lock<boost::mutex> lock(_mutex);
      online_covar->push_sample(descriptor_mat.row(i));
    }
  }
}

void save_to_file(const cv::Mat_<double>& mean, const cv::Mat_<double>& covar_inverted, const std::string& model_output){
  LOG(INFO) << "save to file " << model_output << " ...";
  awesomeness::learning::lda_trick::NegativeModel neg_model;
  mean.convertTo(neg_model.mutable_mu_neg(), neg_model.mutable_mu_neg().type());
  covar_inverted.convertTo(neg_model.mutable_sigma_inv(), neg_model.mutable_sigma_inv().type());

  utils::serialization::write_binary_archive(model_output, neg_model);
  LOG(INFO) << "save to file... done";
}



DEFINE_string(model_output, "/srv/glusterfs/mdantone/data/lookbook/negative_model/", "outputfile");
DEFINE_int32(num_patches_per_image, 1000, "maximum number of blocks to load");
DEFINE_int32(num_image, 1000000, "maximum number of blocks to load");
DEFINE_int32(num_threads, 0, "max number of threads. if set to 0, will be determined automatically");
DEFINE_double(covar_regualizer, 0, "regularizer added to the covariance matrix bevore inverting");


int main(int argc, char** argv){
  google::InstallFailureSignalHandler();
  google::LogToStderr();
  google::InitGoogleLogging(argv[0]);
  // get command line args
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (false){
    google::ShowUsageWithFlags(argv[0]);
    return -1;
  }

  const fs::path model_output = FLAGS_model_output;
  const double covar_regularizer = FLAGS_covar_regualizer;

  const uint32_t num_image = FLAGS_num_image;
  const uint32_t num_patches_per_image = FLAGS_num_patches_per_image;
  const uint32_t patch_width = 32;
  const uint32_t patch_height = 32;


  // set threads
  //TODO: this probably wont work
  int num_threads = FLAGS_num_threads;
  if (num_threads < 1){
    num_threads = utils::system::get_available_logical_cpus();
  }
  cv::setNumThreads(num_threads);
  LOG(INFO) << "using " << num_threads << " threads";
  LOG(INFO) << "num_images: " << num_image;
  LOG(INFO) << "num_patches_per_image: " << num_patches_per_image;



  //----------------------------------------------------------------------------
  // collect files
  // TODO
  std::vector<fs::path> file_names;
  {
    vector<Annotation> annotations;
    string index_file  = "/srv/glusterfs/mdantone/data/lookbook/index_rescaled_train_clean.txt";
    load_annotations(annotations, index_file);
    for(int i=0; i < annotations.size(); i++) {
      file_names.push_back(annotations[i].url);
    }

    if(file_names.size() > num_image) {
      std::random_shuffle(file_names.begin(), file_names.end());
      file_names.resize(num_image);
    }
  }




  //----------------------------------------------------------------------------
  // extract hog features
  LOG(INFO) << "extracting hog features from " << file_names.size() << " images";
  cv::Mat_<double> covar;
  cv::Mat_<double> mean;
  {
    awesomeness::learning::lda_trick::OnlineCovariance online_covar;

    boost::progress_display progress(file_names.size());

    LOG(INFO) << "threads: " << num_threads;
    utils::BlockingThreadPool thread_pool(num_threads, 100);
#ifdef ARMA_USE_BLAS
    openblas_set_num_threads(1);
#endif
    for (unsigned int img_id = 0; img_id < file_names.size(); ++img_id, ++progress){
      // load image
      const fs::path img_path = file_names[img_id];

      if(num_threads == 1) {
        push_hog_sample(img_path, num_patches_per_image, patch_width, patch_height, &online_covar);
      }else{
        thread_pool.submit(boost::bind(push_hog_sample, img_path, num_patches_per_image, patch_width, patch_height, &online_covar));
      }
    }
    thread_pool.join_all();

    online_covar.get_covariance(&covar);
    online_covar.get_mean(&mean);
  }
  LOG(INFO) << "done";

  //----------------------------------------------------------------------------
  // invert cov matrix
  LOG(INFO) << "regularizing with \\lambda=" << covar_regularizer << " ...";
  CHECK_EQ(covar.rows, covar.cols);
  for (int i = 0; i < covar.rows; ++i){
    covar(i, i) += covar_regularizer;
  }
  LOG(INFO) << "regularizing...done";

  cv::Mat_<double> covar_inverted;
  LOG(INFO) << "inverting regularized...";
  const double result = cv::invert(covar, covar_inverted, cv::DECOMP_LU);

  if (result == 0){
    LOG(WARNING) << "inverting regularized...FAILED";
  }
  else {
    LOG(INFO) << "inverting regularized...DONE";
    save_to_file(mean, covar_inverted, model_output.string());
  }



  //----------------------------------------------------------------------------
  // save


}

