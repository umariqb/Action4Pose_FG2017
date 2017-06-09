/*
 * cnn_features.hpp
 *
 *  Created on:  September 18, 2015
 *      Author:  Umar Iqbal
 */


#ifndef CNN_FEATURES_HPP
#define CNN_FEATURES_HPP

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "cpp/vision/features/cnn/caffe_utils.hpp"
#include "caffe/util/db.hpp"
#include "cpp/utils/system_utils.hpp"

using namespace caffe;
namespace db = caffe::db;
using boost::shared_ptr;

namespace vision
{
namespace features
{

class CNNFeatures
{
  public:
    CNNFeatures();
    CNNFeatures(const std::string pretrained_net_proto,
                const std::string feature_extraction_proto,
                const std::string mean_file = "",
                bool use_gpu = true, int device_id = 0);

    bool extract(const cv::Mat, const std::vector<string> feat_names, std::vector<cv::Mat>& features, bool resize_to_img_size = false, int num_threads = -1);

    bool extract_and_save(const cv::Mat image, const std::vector<std::string> feat_names, std::vector<cv::Mat>& features, bool resize_to_img_size, std::string path);

    bool load_features(std::vector<cv::Mat>& features, const std::string path);

    bool get_caffe_blobs(const cv::Mat image,
                       const std::vector<string> feat_names,
                       std::vector<boost::shared_ptr<Blob<float> > >& feature_blobs);

    cv::Mat preprocess(cv::Mat img,  bool use_mean_pixel = false);


    bool extract_all_and_save(const std::vector<cv::Mat> images,
                              const std::vector<std::string> img_names,
                              const std::vector<std::string> feat_names,
                              std::string dir_path,
                              std::string database_type = "lmdb");



    bool set_mean_pixel(cv::Scalar mc);

    cv::Size get_input_geometry();

    virtual ~CNNFeatures();
  protected:


  private:

    bool WrapInputLayer(std::vector<cv::Mat>& input_channels);
    boost::shared_ptr<Net<float> > net;
    cv::Size input_geometry;
    int num_channels;
    cv::Mat mean;
    cv::Scalar channel_mean;
};

} /* namespace features */
} /* namespace vision */

#endif // CNN_FEATURES_HPP

