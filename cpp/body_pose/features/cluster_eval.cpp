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


DEFINE_int32(num_patches_per_image, 50, "maximum number of blocks to load");
DEFINE_int32(num_image, 1000, "maximum number of blocks to load");
DEFINE_int32(num_threads, 0, "max number of threads. if set to 0, will be determined automatically");
DEFINE_int32(num_clusters, 50, "num_clusters ");
DEFINE_int32(patch_size, 32, "patch_size ");
DEFINE_string(forest_path, "/srv/glusterfs/mdantone/data/lookbook/patch_mining/", "outputfile");

DEFINE_string(index_file, "/srv/glusterfs/mdantone/data/lookbook/index_rescaled_train_clean_part0.txt", "index_file ");

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
  const fs::path forest_path = FLAGS_forest_path;
  const fs::path index_file  = FLAGS_index_file;

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
  CHECK(load_images_and_mask(index_file.string(), annotations, num_image));
  LOG(INFO) <<annotations.size() << " images loaded.";


  //----------------------------------------------------------------------------
  // generate random boxes
  sample_patches(annotations, num_patches_per_image, patch_width, patch_height);
  LOG(INFO) << "boxes extracted.";


  //----------------------------------------------------------------------------
  // load forest
  ForestParam forest_param;
  Forest<FeatureSample> forest;
  {
    fs::path tree_save_dir = forest_path.string()+"/forest_0";
    fs::path forest_config_file = forest_path.string()+"/config.txt";

    LOG(INFO) << forest_path;
    CHECK(learning::forest::loadConfigFile(forest_config_file.string(), forest_param));


    CHECK(forest.load(tree_save_dir.string()+"/tree_", forest_param) );
    LOG(INFO) << "done loading, start extracting augmenting features";

  }

  //----------------------------------------------------------------------------
  //
  for(int i_img=0; i_img < annotations.size(); i_img++) {

  }
}

