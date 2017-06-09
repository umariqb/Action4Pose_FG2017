/*
 * feature_clustering.cpp
 *
 *  Created on: Sep 30, 2013
 *      Author: mdantone
 */


#include <google/gflags.h>
#include <glog/logging.h>
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
#include "cpp/vision/features/low_level_features.hpp"
#include "cpp/vision/geometry_utils.hpp"
#include "cpp/learning/forest/utils/eval_utils.hpp"

#include "cpp/body_pose/common.hpp"
#include "common.hpp"
#include "cpp/body_pose/utils.hpp"
#include "cpp/learning/forest/forest.hpp"
#include "cpp/body_pose/features/feature_sample.hpp"

namespace vf = vision::features;
namespace fs = boost::filesystem;

using namespace learning::forest;
using namespace learning::common;
using namespace bodypose::features;

using namespace std;
using namespace cv;


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

bool is_unique(vector<Point> locations, Point p, float threshold = 10  ) {
  for(int i=0; i < locations.size(); i++) {
    if( vision::geometry_utils::dist(locations[i], p )  < threshold) {
      return false;
    }
  }
  return true;
}

void sample_patches(
    vector<bodypose::features::BoxAnnotation>& annotations,
    const int num_patches,
    const int patch_width,
    const int patch_height,
    bool debug = false) {
  boost::mt19937 rng;
  for(int i_img=0; i_img < annotations.size(); i_img++) {
    cv::Mat& image = annotations[i_img].image;
    cv::Mat_<uchar>& mask = annotations[i_img].mask;

    // generate num_patches random rectangles
    boost::uniform_int<> dist_x(0, image.cols - patch_width - 1);
    boost::uniform_int<> dist_y(0, image.rows - patch_height - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
    std::vector<Point> locations;
    if(mask.data) {
      int save = 0;
      while(locations.size() < num_patches && save < num_patches*150 ) {
        save++;
        Point p = cv::Point(rand_x(),rand_y());
       CHECK_LT( p.x + patch_width, image.cols );
       CHECK_LT( p.y + patch_height, image.rows );

       Rect bbox(p.x, p.y, patch_width, patch_height);
       float ratio_foreground = cv::countNonZero(mask(bbox))/ static_cast<float>(patch_width*patch_height);
       if(ratio_foreground >= 0.5 && is_unique(locations, p, patch_width*0.25 ) ) {
         locations.push_back( p );
         annotations[i_img].bboxs.push_back( bbox );
       }
      }
    }else{
      for(int i=0; i < num_patches; i++) {
        Point p = cv::Point(rand_x(),rand_y());
        if(is_unique(locations, p, patch_width*0.25 )) {
          locations.push_back( p );
          CHECK_LT( p.x + patch_width, image.cols );
          CHECK_LT( p.y + patch_height, image.rows );
          annotations[i_img].bboxs.push_back( cv::Rect(p.x, p.y, patch_width, patch_height) );
        }
      }
    }

    if(debug) {
      Mat img = imread(annotations[i_img].img_url);
      for(int i=0; i < annotations[i_img].bboxs.size(); i++) {
        cv::rectangle(img, annotations[i_img].bboxs[i], cv::Scalar(255, 255, 255, 0));
      }
      imshow("img", img);
      cv::waitKey(0);
    }
  }
}

void extract_features_hog( bodypose::features::BoxAnnotation* ann  ) {
  std::vector<Point> locations;
  for(int i=0;i < ann->bboxs.size(); i++) {
    locations.push_back( Point( ann->bboxs[i].x, ann->bboxs[i].y ));
  }
  if(locations.size() == 0)
    return;

  HOGDescriptor hog_extractor = get_hog_descriptor(ann->bboxs[0].width, ann->bboxs[0].height);
  vector<float> descriptors;
  hog_extractor.compute(ann->image, descriptors, cv::Size(), cv::Size(), locations);
  cv::Mat_<float> descriptor_mat(descriptors, false);
  descriptor_mat = descriptor_mat.reshape(0, locations.size());
  ann->app_features = descriptor_mat.clone();

//  LOG(INFO) << ann->app_features.row(0);
//  LOG(INFO) << cv::sum(ann->app_features)[0];
}

void extract_features_rf_mt(Forest<FeatureSample>* forest,
                            bodypose::features::BoxAnnotation* ann ) {
  if(ann->bboxs.size() ==0)
    return;

  learning::common::Image img;
  vision::features::feature_channels::FeatureChannelFactory fcf;
  img.init(ann->image, forest->getParam().features, &fcf, false, 0);

  // eval image
  cv::vector<cv::Mat> voting_maps;
  cv::Mat foreground_map;
  learning::forest::utils::eval_mc_forest( *forest, img,
     50, 2, voting_maps, foreground_map, false);


  // allocate features
  int patch_width = ann->bboxs[0].width;
  int patch_height = ann->bboxs[0].height;
  int num_features = ann->bboxs.size();
  int size = 16;// std::min(patch_width, patch_height);//
  int feature_dim = voting_maps.size() * (patch_width/size) * (patch_height/size);
  ann->app_features = Mat_<float>(num_features, feature_dim, 0.0);
  for(int i_patch =0; i_patch < num_features; i_patch++) {
    int i =0;
    int x = ann->bboxs[i_patch].x;
    while( x <=  (ann->bboxs[i_patch].x + patch_width-size) ) {
      int y = ann->bboxs[i_patch].y;
      while( y <=  (ann->bboxs[i_patch].y + patch_height-size) ) {

        Rect roi(x,y,size,size);
        for(int j=0; j < voting_maps.size(); j++) {
          ann->app_features.at<float>(i_patch, i) = sum(voting_maps[j](roi))[0];
          i++;
        }
        y += size;
      }
      x += size;
    }
  }
}

void extract_features_rf(vector<bodypose::features::BoxAnnotation>& annotations) {
  string base = "/home/mdantone/scratch/grid/features/test_50_16/";
  fs::path tree_save_dir = base+"forest_0";
  fs::path forest_config_file = base+"config.txt";
  LOG(INFO) << base;
  ForestParam param;
  CHECK(learning::forest::loadConfigFile(forest_config_file.string(), param));

  Forest<FeatureSample> forest;
  CHECK(forest.load(tree_save_dir.string()+"/tree_", param) );
  LOG(INFO) << "done loading, start extracting augmenting features";

  int num_threads = ::utils::system::get_available_logical_cpus();
  if( num_threads <= 1) {
    for(int i=0; i < annotations.size(); i++) {
      extract_features_rf_mt(&forest, &annotations[i]);
    }
  }else{
    boost::thread_pool::executor e(num_threads);
    for(int i=0; i < annotations.size(); i++) {
      e.submit(boost::bind( &extract_features_rf_mt, &forest, &annotations[i]) );
    }
    e.join_all();
  }
}

void extract_features_hog(vector<bodypose::features::BoxAnnotation>& annotations) {
  for(int i_img=0; i_img < annotations.size(); i_img++) {
    extract_features_hog( &annotations[i_img]);
  }
}

bool load_images_and_mask(string index_file, vector<bodypose::features::BoxAnnotation>& annotations,  int num_image ) {

  vector<Annotation> body_annotations;
  load_annotations(body_annotations, index_file);
  if(body_annotations.size() > num_image) {
    std::random_shuffle(body_annotations.begin(), body_annotations.end());
    body_annotations.resize(num_image);
  }

  annotations.resize(body_annotations.size() );
  boost::progress_display progress(body_annotations.size());
  for(int i=0; i < body_annotations.size(); i++, ++progress) {
    annotations[i].img_url = body_annotations[i].url;
    annotations[i].parts = body_annotations[i].parts;
    annotations[i].image = cv::imread(annotations[i].img_url);
    CHECK(annotations[i].image.data) << "could not read image " << annotations[i].img_url;
    get_mask( annotations[i].image, annotations[i].parts, annotations[i].mask, -1);
  }
  return annotations.size() == body_annotations.size();
}

void add_spatial_information(vector<bodypose::features::BoxAnnotation>& annotations, float ration = 1.0) {

  for(int img_id=0; img_id < annotations.size(); img_id++) {

    int n_bboxes = annotations[img_id].bboxs.size();

    // calculate offset to head
    int n_parts = annotations[img_id].parts.size();
    annotations[img_id].spatials_features = Mat_<float>(n_bboxes, n_parts*2, 0.0);

    for(int i_box=0; i_box < n_bboxes; i_box++) {

      Mat_<float> row = annotations[img_id].spatials_features.row(i_box);

      Point corner_bbox(annotations[img_id].bboxs[i_box].x,
                        annotations[img_id].bboxs[i_box].y);

      for(int i_part = 0; i_part < n_parts; i_part++) {
        Point part = annotations[img_id].parts[i_part];
        Point offset = part - corner_bbox;

        row(i_part*2 ) = offset.x;
        row(i_part*2 +1 ) = offset.y;
      }
    }
    CHECK_EQ(annotations[img_id].app_features.rows,
             annotations[img_id].spatials_features.rows);
  }
}

DEFINE_int32(num_patches_per_image, 50, "maximum number of blocks to load");
DEFINE_int32(num_image, 1000, "maximum number of blocks to load");
DEFINE_int32(num_threads, 0, "max number of threads. if set to 0, will be determined automatically");
DEFINE_int32(num_clusters, 50, "num_clusters ");
DEFINE_int32(patch_size, 32, "patch_size ");

DEFINE_string(model_output, "/srv/glusterfs/mdantone/data/lookbook/patch_mining/", "outputfile");


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

  const uint32_t num_image = FLAGS_num_image;
  const uint32_t num_patches_per_image = FLAGS_num_patches_per_image;
  const uint32_t patch_width = FLAGS_patch_size;
  const uint32_t patch_height = FLAGS_patch_size;
  const uint32_t num_clusters = FLAGS_num_clusters;
  const fs::path model_output = FLAGS_model_output;


  // set threads
  //TODO: this probably wont work
  int num_threads = FLAGS_num_threads;
  if (num_threads < 1){
    num_threads = ::utils::system::get_available_logical_cpus();
  }
  cv::setNumThreads(num_threads);
  LOG(INFO) << "using " << num_threads << " threads";
  LOG(INFO) << "num_images: " << num_image;
  LOG(INFO) << "num_patches_per_image: " << num_patches_per_image;
  LOG(INFO) << "num_clusters: " << num_clusters;



  //----------------------------------------------------------------------------
  // load images and mask
  LOG(INFO) << "start loading features loaded.";
  vector<bodypose::features::BoxAnnotation> annotations;
  string index_file  = "/srv/glusterfs/mdantone/data/lookbook/index_rescaled_train_clean_part1.txt";
  CHECK(load_images_and_mask(index_file, annotations, num_image));
  LOG(INFO) <<annotations.size() << " images loaded.";


  //----------------------------------------------------------------------------
  // generate random boxes
  sample_patches(annotations, num_patches_per_image, patch_width, patch_height);
  LOG(INFO) << "boxes extracted.";


  //----------------------------------------------------------------------------
  // extract app_features
  if(true) {
    extract_features_hog(annotations);
  }else{
    extract_features_rf(annotations);
  }
  LOG(INFO) << "features extracted.";


  //----------------------------------------------------------------------------
  // add spatial
  add_spatial_information(annotations);

  //----------------------------------------------------------------------------
  // allocate app_features
  int num_features = 0;
  int feature_size_app = 0;
  int feature_size_spatial = 0;

  for(int i_img=0; i_img < annotations.size(); i_img++) {
    int i_features = annotations[i_img].app_features.rows;
    if(i_features) {
      num_features += i_features;
      feature_size_app = annotations[i_img].app_features.cols;
      feature_size_spatial = annotations[i_img].spatials_features.cols;
    }
  }
  int feature_size = feature_size_app+feature_size_spatial;
  cv::Mat_<float> features(num_features, feature_size, 0.0);
  LOG(INFO) << "features allocated, size (" << features.rows << " x " << features.cols << " )";

  //----------------------------------------------------------------------------
  // copy features into one Mat app_features
  int i_features = 0;
  if(feature_size_spatial == 0) {
    for(int i_img=0; i_img < annotations.size(); i_img++) {
      cv::Rect roi(0, i_features, feature_size, annotations[i_img].app_features.rows);
          annotations[i_img].app_features.copyTo( features(roi) );
      i_features += annotations[i_img].app_features.rows;
    }
  }else{
    for(int i_img=0; i_img < annotations.size(); i_img++) {
      if(annotations[i_img].app_features.rows == 0) {
        continue;
      }
      CHECK_EQ(annotations[i_img].app_features.rows, annotations[i_img].spatials_features.rows );
      cv::Rect roi_app(0, i_features, feature_size_app, annotations[i_img].app_features.rows);
      cv::Rect roi_spa(feature_size_app, i_features, feature_size_spatial, annotations[i_img].spatials_features.rows);
      annotations[i_img].app_features.copyTo( features(roi_app) );
      annotations[i_img].spatials_features.copyTo( features(roi_spa) );

      i_features += annotations[i_img].app_features.rows;
    }
    LOG(INFO) << "copied app_features";

    // normalize vectors
    cv::Rect roi_app(0,0,feature_size_app, i_features);
    cv::Rect roi_spat(feature_size_app, 0, feature_size_spatial, i_features );

    float sum_app = cv::sum(cv::abs(features(roi_app)))[0];
    float sum_spat = cv::sum(cv::abs(features(roi_spat)))[0];
    if(sum_app != 0)
      features(roi_app) /= sum_app;

    if(sum_spat != 0)
      features(roi_spat) /= sum_spat;
    LOG(INFO) << "normalized features";

  }
  CHECK_EQ(i_features, num_features );


  //----------------------------------------------------------------------------
  // clustering
  LOG(INFO) << "start clustering";

  cv::TermCriteria term_criteria;
  term_criteria.epsilon = 1;
  term_criteria.maxCount = 5;
  term_criteria.type = cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS;
  cv::Mat labels;
  cv::Mat centers;
  srand(1);
  cv::kmeans(features, num_clusters, labels, term_criteria, 10, cv::KMEANS_PP_CENTERS, centers);

  LOG(INFO) << "done clustering";

  vector<int> hist(num_clusters, 0);
  {
    Mat dists;
    Mat_<int> indices;
    LOG(INFO) << "calculate distances";
    flann::SearchParams p;
    p.setAlgorithm(cvflann::FLANN_DIST_EUCLIDEAN);
    flann::Index index(centers, p);
    index.knnSearch(features, indices, dists, 1, p);

    int i_feature = 0;
    for (unsigned int img_id = 0; img_id <annotations.size(); ++img_id){
      for (unsigned int i_patch = 0; i_patch < annotations[img_id].bboxs.size(); ++i_patch){
//        LOG(INFO) << dists.row(i_feature);
//        LOG(INFO) << labels.at<int>(0,i_feature) << " -> " << indices.row(i_feature);
        int label = indices(i_feature,0);
        float conf = dists.at<float>(i_feature,0);
        annotations[img_id].confidences.push_back( -conf );
        annotations[img_id].labels.push_back( label );
        hist[label]++;
        i_feature++;
      }
      CHECK_EQ(annotations[img_id].confidences.size(), annotations[img_id].labels.size() );
      CHECK_EQ(annotations[img_id].confidences.size(), annotations[img_id].bboxs.size() );

    }
    LOG(INFO) << "distances calculated";
  }
  for (unsigned int i = 0; i <hist.size(); ++i){
    LOG(INFO) << i << ": " << hist[i];
  }


  //----------------------------------------------------------------------------
  // saving annotations

  LOG(INFO) << "save to file " << model_output << " ...";
  CHECK(::utils::serialization::write_binary_archive(model_output.string(), annotations));
  LOG(INFO) << "save to file... done";


  //----------------------------------------------------------------------------
  // saving clusters
  fs::path output_cluster =  model_output;
  output_cluster.replace_extension("centers");
  LOG(INFO) << "save to file " << output_cluster.string() << " ...";
  CHECK(::utils::serialization::write_binary_archive(output_cluster.string(), annotations));
  LOG(INFO) << "save to file... done";
}

