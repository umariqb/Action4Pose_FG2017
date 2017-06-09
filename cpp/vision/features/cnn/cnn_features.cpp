/*
 * cnn_features.cpp
 *
 *  Created on:  September 18, 2015
 *      Author:  Umar Iqbal
 */

#include "cpp/vision/features/cnn/cnn_features.hpp"
#include "cpp/vision/features/cnn/caffe_utils.hpp"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/utils/thread_pool.hpp"
#include "cpp/utils/timing.hpp"

using namespace cv;
using namespace std;
using boost::shared_ptr;

namespace vision
{
namespace features
{


CNNFeatures::CNNFeatures()
{
  //ctor
}

CNNFeatures::CNNFeatures(const std::string pretrained_net_proto,
                         const std::string feature_extraction_proto,
                         const std::string mean_file,
                         bool use_gpu, int device_id):
                         net(new caffe::Net<float>(feature_extraction_proto, caffe::TEST))
{
  if (use_gpu) {
    LOG(INFO)<< "Using GPU";
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  net->CopyTrainedLayersFrom(pretrained_net_proto);

  Blob<float>* input_layer = net->input_blobs()[0];
  num_channels = input_layer->channels();
  CHECK(num_channels == 3 || num_channels == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  if(mean_file != ""){
    mean = caffe::utils::GetMean(mean_file);
    mean.convertTo(mean, CV_32F);
    channel_mean = cv::mean(mean);
  }
  else{
    mean = cv::Mat::zeros(input_geometry, CV_32FC3);
    channel_mean = cv::Scalar(0,0,0);
  }

}

cv::Mat CNNFeatures::preprocess(cv::Mat img,  bool use_mean_pixel)
{
  cv::Mat sample;
  if (img.channels() == 3 && num_channels == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  // do mean normalization
  cv::Mat sample_normalized;
  cv::Mat sample_resized, sample_float;

  if(use_mean_pixel){  // if a single mean pixel has to be used
    if (sample.size() != input_geometry)
      cv::resize(sample, sample_resized, input_geometry);
    else
      sample_resized = sample;

    sample_resized.convertTo(sample_float, CV_32F);
    cv::Mat mean_mat = cv::Mat(input_geometry, mean.type(), channel_mean);
    cv::subtract(sample_float, mean_mat, sample_normalized);
  }
  else{ // if complete mean image has to be used
    if (sample.size() != mean.size())
      cv::resize(sample, sample_resized, mean.size());
    else
      sample_resized = sample;

    sample_resized.convertTo(sample_float, CV_32F);
    cv::subtract(sample_float, mean, sample_normalized);

    cv::resize(sample_normalized, sample_normalized, input_geometry);
  }
  return sample_normalized;
}

bool CNNFeatures::get_caffe_blobs(Mat image, vector<string> feat_names,  std::vector<boost::shared_ptr<Blob<float> > >& feature_blobs)
{
    CHECK(feat_names.size());

    cv::Mat norm_image;
    norm_image = preprocess(image, true);

    //wrap input layer with image
    vector<cv::Mat> input_channels;
    WrapInputLayer(input_channels);
    split(norm_image,input_channels);

    ///checks weather wraping is correct
    CHECK(reinterpret_cast<float*>(input_channels.at(0).data) ==
          net->input_blobs()[0]->cpu_data() )
          <<"Input channels are not wrapping the input layer";

    //forward pass
    net->ForwardPrefilled();

    int num_feat = feat_names.size();

    feature_blobs.resize(num_feat);
    for(int i=0; i<num_feat; i++){
      feature_blobs[i] = net->blob_by_name(feat_names[i]);
    }
    return true;
}

bool extract_and_resize_mt(cv::Size img_sz, boost::shared_ptr<Blob<float> > feature_blob,
          vector<Mat>& features, bool resize_to_img_size, int start_idx)
{
  int height = feature_blob->height();
  int width = feature_blob->width();
  int channels = feature_blob->channels();
  int dim = width * height;

  for (int c=0; c<channels; c++){
    cv::Mat feat_mat(cv::Mat(height, width, CV_32FC1, feature_blob->mutable_cpu_data()+dim*c));
    if(resize_to_img_size){
      resize(feat_mat, feat_mat, img_sz, CV_INTER_LINEAR);
    }
    features[start_idx+c] = feat_mat;
  }
}

bool CNNFeatures::extract(Mat image, vector<string> feat_names, vector<Mat>& features, bool resize_to_img_size, int num_threads)
{

    std::vector<boost::shared_ptr<Blob<float> > > feature_blobs;
    get_caffe_blobs(image, feat_names, feature_blobs);
    CHECK_EQ(feat_names.size(), feature_blobs.size());

    int total_channels = 0;
    vector<int> cum_chn_count(feature_blobs.size(), 0);
    for(int i=0; i<feature_blobs.size(); i++){
      total_channels +=  feature_blobs[i]->channels();
    }

    if(num_threads < 1){
        num_threads = ::utils::system::get_available_logical_cpus();
    }

    if(0){ //FIXME
      features.resize(total_channels);
      boost::thread_pool::executor e(num_threads);
      int start_idx = 0;
      for(int i=0; i<feature_blobs.size(); i++){

        if(i > 0){
          start_idx += feature_blobs[i-1]->channels();
        }
        e.submit(boost::bind(&extract_and_resize_mt, image.size(), feature_blobs[i], features, resize_to_img_size, start_idx));
      }
      e.join_all();
    }
    else{
      features.reserve(total_channels);
      for(int i=0; i<feature_blobs.size(); i++){
        int height = feature_blobs[i]->height();
        int width = feature_blobs[i]->width();
        int channels = feature_blobs[i]->channels();
        int dim = width * height;

        for (int c=0; c<channels; c++){
          cv::Mat feat_mat(cv::Mat(height, width, CV_32FC1, feature_blobs[i]->mutable_cpu_data()+dim*c));
          if(resize_to_img_size){
            resize(feat_mat, feat_mat, cv::Size(image.cols, image.rows), CV_INTER_LINEAR);
          }
          features.push_back(feat_mat);
          if(0){
            imshow("image", image);
            normalize(feat_mat, feat_mat, 1, 0, CV_MINMAX);
            imshow("feat_mat", feat_mat);
            waitKey(0);
          }
        }
      }
    }
    return true;
}

bool CNNFeatures::extract_and_save(Mat image, vector<string> feat_names, vector<cv::Mat>& features, bool resize_to_img_size, std::string path){

  extract(image, feat_names, features, resize_to_img_size);

    try{
      std::ofstream ofs(path.c_str());
      if(ofs==0){
      LOG(INFO)<<"Error: Cannot open the given path to save features.";
      return false;
      }
      boost::archive::text_oarchive oa(ofs);
      oa<<features;
      ofs.flush();
      ofs.close();
      LOG(INFO)<<"Features saved at :"<<path;
      return true;
    }
    catch(boost::archive::archive_exception& ex){
      LOG(INFO)<<"Archive exception during deserialization:" <<std::endl;
      LOG(INFO)<< ex.what() << std::endl;
      LOG(INFO)<< "it was file: "<<path;
    }
    return true;
}

bool CNNFeatures::load_features(vector<Mat>& features, string path)
{
    std::ifstream ifs(path.c_str());
    if(!ifs){
      LOG(INFO)<<"file not found.";
    }
    else{
      try{
        boost::archive::text_iarchive ia(ifs);
        ia>>features;
        LOG(INFO)<<"Features loaded";
        return true;
      }
      catch(boost::archive::archive_exception& ex){
        LOG(INFO)<<"Reload Tree: Archive exception during deserializiation: "
              <<ex.what();
          LOG(INFO)<<"not able to load features from: "<<path;
      }
    }
    return false;
}

//TODO: Incorporate batch processing!!!!
bool CNNFeatures::extract_all_and_save(const std::vector<cv::Mat> images,
                                        const std::vector<std::string> img_names,
                                        const vector<string> feat_names,
                                        std::string dir_path,
                                        std::string database_type)

{
    CHECK(images.size());

    int num_feat = feat_names.size();

    std::vector<boost::shared_ptr<caffe::db::DB> > feature_dbs;
    std::vector<boost::shared_ptr<caffe::db::Transaction> > txns;
    for (size_t i = 0; i < num_feat; ++i) {
      boost::filesystem::path dataset_name(dir_path+"/"+feat_names[i]);

      caffe::db::Mode db_mode;
      if(boost::filesystem::exists(dataset_name)){
        LOG(INFO)<<"Dataset already exists. Opening existing database in WRITE mode: "<< dataset_name;
        db_mode = caffe::db::WRITE;
      }
      else{
        db_mode = caffe::db::NEW;
        LOG(INFO)<< "Opening dataset " << dataset_name;
      }

      boost::shared_ptr<caffe::db::DB> db(caffe::db::GetDB(database_type.c_str()));

      db->Open(dataset_name.string(), db_mode);
      feature_dbs.push_back(db);
      boost::shared_ptr<caffe::db::Transaction> txn(db->NewTransaction());
      txns.push_back(txn);
    }

    const int kMaxKeyStrLength = 100;
    char key_str[kMaxKeyStrLength];

    Datum datum;
    for(int m=0; m < images.size(); m++){
      cv::Mat image = images[m];

      std::vector<boost::shared_ptr<Blob<float> > > feature_blobs;
      get_caffe_blobs(image, feat_names, feature_blobs);
      CHECK_EQ(feat_names.size(), feature_blobs.size());

      for(int i=0; i<feature_blobs.size(); i++){
        datum.set_height(feature_blobs[i]->height());
        datum.set_width(feature_blobs[i]->width());
        datum.set_channels(feature_blobs[i]->channels());
        datum.clear_data();
        datum.clear_float_data();

        const float* feat_blob_data = feature_blobs[i]->cpu_data();

        int dim_features = feature_blobs[i]->count();
        for (int d = 0; d < dim_features; ++d) {
          datum.add_float_data(feat_blob_data[d]);
        }

//        int length = snprintf(key_str, kMaxKeyStrLength, img_names[m].c_str());
        int length = snprintf(key_str, kMaxKeyStrLength, "%010d", m);
        string out;
        CHECK(datum.SerializeToString(&out));
        txns.at(i)->Put(std::string(key_str, length), out);

        if (m % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
          LOG(ERROR)<< "Extracted features of " << m <<
              " query images for feature blob " << feat_names[i];
        }
      }
    }
}

bool CNNFeatures::WrapInputLayer(vector<cv::Mat>& input_channels){
    Blob<float>* input_layer = net->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for(int i=0; i<input_layer->channels(); i++){
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels.push_back(channel);
      input_data += width * height;
    }
    return true;
}

bool CNNFeatures::set_mean_pixel(cv::Scalar mc){
  channel_mean =  mc;
  return true;
}

cv::Size CNNFeatures::get_input_geometry()
{
  return input_geometry;
}

CNNFeatures::~CNNFeatures()
{
  //dtor
}


} /* namespace features */
} /* namespace vision */
