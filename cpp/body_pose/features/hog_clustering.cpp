/*
 * hog_clustering.cpp
 *
 *  Created on: Sep 30, 2013
 *      Author: mdantone
 */

/*
 * train_lda_trick_hog_main.cpp
 *
 *  Created on: Sep 4, 2013
 *      Author: lbossard
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

#include "cpp/body_pose/common.hpp"
#include "common.hpp"
#include "cpp/body_pose/utils.hpp"
namespace vf = vision::features;
using namespace std;
using namespace cv;
using namespace boost;

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
    const Mat& image,
    const int num_patches,
    const int patch_width,
    const int patch_height,
    std::vector<cv::Rect>&  sample_rectangles,
    cv::Mat_<uchar>& mask) {

  // generate num_patches random rectangles
  boost::mt19937 rng;
  boost::uniform_int<> dist_x(0, image.cols - patch_width - 1);
  boost::uniform_int<> dist_y(0, image.rows - patch_height - 1);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
  std::vector<Point> locations;

  if(mask.data) {
    int save = 0;
    while(locations.size() < num_patches && save < num_patches*250 ) {
      save++;
      Point p = cv::Point(rand_x(),rand_y());
     CHECK_LT( p.x + patch_width, image.cols );
     CHECK_LT( p.y + patch_height, image.rows );
     if(mask.at<uchar>(p) > 0 && is_unique(locations, p, patch_width*0.25 ) ) {
       locations.push_back( p );
       sample_rectangles.push_back( cv::Rect(p.x, p.y, patch_width, patch_height) );
     }
    }
  }else{
    for(int i=0; i < num_patches; i++) {
      Point p = cv::Point(rand_x(),rand_y());
      if(is_unique(locations, p, patch_width*0.25 )) {
        locations.push_back( p );
        sample_rectangles.push_back( cv::Rect(p.x, p.y, patch_width, patch_height) );
        CHECK_LT( p.x + patch_width, image.cols );
        CHECK_LT( p.y + patch_height, image.rows );
      }
    }
  }
}

void push_hog_sample(
    const Mat& image,
    const int patch_width,
    const int patch_height,
    int featue_id_low,
    int num_features,
    const std::vector<cv::Rect>&  rectangles,
    cv::Mat_<float>* features
    ){

  std::vector<Point> locations;
  for(int i=0;i < rectangles.size(); i++) {
    locations.push_back( Point( rectangles[i].x, rectangles[i].y ));
  }

  // extract descriptor
  HOGDescriptor hog_extractor = get_hog_descriptor(patch_width, patch_height);
  vector<float> descriptors;
  hog_extractor.compute(image, descriptors, cv::Size(), cv::Size(), locations);
  cv::Mat_<float> descriptor_mat(descriptors, false);
  descriptor_mat = descriptor_mat.reshape(0, locations.size());
  {
    cv::Rect roi(0, featue_id_low, hog_extractor.getDescriptorSize(), num_features);
    cv::Mat_<float> feat = (*features)(roi);
    descriptor_mat.copyTo(feat);
    // copy data
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
    num_threads = utils::system::get_available_logical_cpus();
  }
  cv::setNumThreads(num_threads);
  LOG(INFO) << "using " << num_threads << " threads";
  LOG(INFO) << "num_images: " << num_image;
  LOG(INFO) << "num_patches_per_image: " << num_patches_per_image;



  //----------------------------------------------------------------------------
  // collect files
  // TODO
  std::vector<cv::Mat> images;
  std::vector<cv::Mat_<uchar> > masks;
  std::vector<fs::path> file_names;
  vector<Annotation> annotations;

  {
    string index_file  = "/srv/glusterfs/mdantone/data/lookbook/index_rescaled_train_clean_part1.txt";
    load_annotations(annotations, index_file);
    if(annotations.size() > num_image) {
      std::random_shuffle(annotations.begin(), annotations.end());
      annotations.resize(num_image);
    }

    for(int i=0; i < annotations.size(); i++) {

      cv::Mat image = cv::imread(annotations[i].url);
      CHECK(image.data) << "could not read image " << annotations[i].url;
      Mat_<uchar> mask;
      get_mask( image, annotations[i], mask, -1);

      file_names.push_back(annotations[i].url);
      masks.push_back(mask);
      images.push_back(image);
    }


  }


  HOGDescriptor hog_desc =  get_hog_descriptor(patch_width, patch_height);


  //----------------------------------------------------------------------------
  // caluclating boxes
  std::vector< std::vector<cv::Rect> > sample_rectangles(images.size() );

  for (unsigned int img_id = 0; img_id < images.size(); ++img_id){
    sample_patches( images[img_id], num_patches_per_image, patch_width, patch_height,sample_rectangles[img_id], masks[img_id]);
  }

  int num_features = 0;
  for (unsigned int img_id = 0; img_id < images.size(); ++img_id){
    num_features += sample_rectangles[img_id].size();
  }


  //----------------------------------------------------------------------------
  // extract hog features
  LOG(INFO) << "extracting hog features from " << images.size() << " images";

  cv::Mat_<float> features(images.size()*num_patches_per_image,
                           hog_desc.getDescriptorSize(),
                           0.0);


  LOG(INFO) << "features allocated, size (" << features.rows << " x " << features.cols << " )";


//  void push_hog_sample(
//      const Mat& image,
//      const int patch_width,
//      const int patch_height,
//      int featue_id_low,
//      int num_features,
//      cv::Mat_<float>* features,
//      std::vector<cv::Rect>*  sample_rectangles){


  {
    boost::progress_display progress(images.size());

    LOG(INFO) << "threads: " << num_threads;
    utils::BlockingThreadPool thread_pool(num_threads, 100);

    int i_features = 0;
    for (unsigned int img_id = 0; img_id < images.size(); ++img_id, ++progress){
      // load image
      const cv::Mat& image = images[img_id];
      if(num_threads == 1) {
        push_hog_sample(image,  patch_width, patch_height, i_features, sample_rectangles[img_id].size(), sample_rectangles[img_id], &features);
      }else{
        thread_pool.submit(boost::bind(push_hog_sample,  boost::ref(image), patch_width, patch_height,
            i_features, sample_rectangles[img_id].size(), boost::ref(sample_rectangles[img_id]), &features));
      }

      i_features += sample_rectangles[img_id].size();
    }
    CHECK_EQ(i_features, num_features );
    thread_pool.join_all();

  }
  LOG(INFO) << "done feature extraction";

  //----------------------------------------------------------------------------
  // clustering
  LOG(INFO) << "start clustering";

  cv::TermCriteria term_criteria;
  term_criteria.epsilon = 1;
  term_criteria.maxCount = 5;
  term_criteria.type = cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS;
  cv::Mat labels;
  cv::Mat centers;
  cv::kmeans(features, num_clusters, labels, term_criteria, 10, cv::KMEANS_PP_CENTERS, centers);

  LOG(INFO) << "done clustering";
  Mat dists;
  Mat_<int> indices;

  {
    LOG(INFO) << "calculate distances";

    flann::SearchParams p;
    p.setAlgorithm(cvflann::FLANN_DIST_EUCLIDEAN);
    flann::Index index(centers, p);
    index.knnSearch(features, indices, dists, 1, p);
//
//    int i_feature = 0;
//    for (unsigned int img_id = 0; img_id <images.size(); ++img_id){
//      for (unsigned int i_patch = 0; i_patch < sample_rectangles[img_id].size(); ++i_patch){
//        LOG(INFO) << dists.row(i_feature);
//        LOG(INFO) << labels.at<int>(0,i_feature) << " -> " << indices.row(i_feature);
//
////        CHECK_EQ(indices(i_feature,0), labels.at<int>(0,i_feature)) << i_feature;
//
//        i_feature++;
//
//        if ( i_feature > 50)
//          return 0;
//      }
//    }

    LOG(INFO) << "distances calculated";
  }





  if(false) {

    vector< vector< pair<float, int> > > clusters(num_clusters);
    // collect all patches
    int i_feature = 0;
    for (unsigned int img_id = 0; img_id <images.size(); ++img_id){
      for (unsigned int i_patch = 0; i_patch < sample_rectangles[img_id].size(); ++i_patch){

        int label = indices(i_feature,0);
        float conf = dists.at<float>(i_feature,0);
        clusters[label].push_back( make_pair(conf, i_feature) );
        i_feature++;
      }
    }
    CHECK_EQ(i_feature, num_features );

    for(int i_cluster =0; i_cluster < clusters.size(); i_cluster++) {
      sort(clusters[i_cluster].begin(), clusters[i_cluster].end());

      for(int count =0; count < 50; count ++) {

        i_feature = clusters[i_cluster][count].second;
        int img_id_selected =0;
        int patch_id_selected =0;
        int i = 0;

        bool found =false;
        for (unsigned int img_id = 0; img_id <images.size(); ++img_id){
          for (unsigned int i_patch = 0; i_patch < sample_rectangles[img_id].size(); ++i_patch){
            if (i_feature == i) {
              img_id_selected = img_id;
              patch_id_selected = i_patch;
              found = true;
              break;
            }
            i++;
          }
          if(found) break;
        }
        cout << clusters[i_cluster][count].first << " " << img_id_selected << ", ";
        const fs::path img_path = file_names[img_id_selected];
        const cv::Mat image = cv::imread(img_path.string());

        //string save_path( boost::str(boost::format("/home/mdantone/public_html/share/features/clustering_kmean/%1%_%2%.jpg") % i_cluster % (count) ));
        //cv::imwrite(save_path, image(sample_rectangles[img_id_selected][patch_id_selected]) );
      }
    }
    cout << " " << endl;
    LOG(INFO) << "done extracting";
  }


  // saving
  vector<bodypose::features::BoxAnnotation> sampels;
  int i_feature = 0;
  for (unsigned int img_id = 0; img_id <file_names.size(); ++img_id){

    const fs::path img_path = file_names[img_id];
    bodypose::features::BoxAnnotation s;
    s.img_url = img_path.string();
    s.parts = annotations[img_id].parts;
    for (unsigned int i_patch = 0; i_patch < sample_rectangles[img_id].size(); ++i_patch){

      int label = indices(i_feature,0);
      float conf = dists.at<float>(i_feature,0);
      s.bboxs.push_back( sample_rectangles[img_id][i_patch] );
      s.labels.push_back(label );
      s.confidences.push_back( -conf );
      i_feature++;
    }
    sampels.push_back(s);
  }
  CHECK_EQ(i_feature, num_features );


  LOG(INFO) << "save to file " << model_output << " ...";
  CHECK_EQ(sampels.size(), file_names.size());
  utils::serialization::write_binary_archive(model_output.string(), sampels);
  LOG(INFO) << "save to file... done";

}

