/*
 * patch_mining.cpp
 *
 *  Created on: Sep 30, 2013
 *      Author: mdantone
 */

#include <glog/logging.h>
#include <google/gflags.h>

#include "common.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/learning/common/image_sample.hpp"
#include "feature_sample.hpp"
#include "cpp/body_pose/utils.hpp"
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include "cpp/body_pose/common.hpp"
#include "cpp/vision/geometry_utils.hpp"
#include "cpp/utils/system_utils.hpp"
#include "cpp/body_pose/utils.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/assign.hpp>

using namespace bodypose::features;
namespace fs = boost::filesystem;

using namespace cv;
using namespace std;
using namespace learning::common;
using namespace learning::forest;


struct more_confident{
  inline bool operator() (const FeatureSample& struct1, const FeatureSample& struct2) {
    return (struct1.get_confidents() > struct2.get_confidents());
  }
};

void create_neg_trainings_samples(
    vector<Image>& image_samples,
    const vector<cv::Mat_<uchar> > masks,
    vector<FeatureSample>& training_sampels,
    int n_tree){
  int n_pos_samples = training_sampels.size();

  int save = 0;
  boost::mt19937 rng;
  rng.seed( n_tree +1);

  int patch_width = training_sampels[0].roi.width;
  int patch_height = training_sampels[0].roi.height;
  while( training_sampels.size() < n_pos_samples*2 && save < n_pos_samples*100) {
    save++;
    boost::uniform_int<> dist_img(0, image_samples.size() - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_img(rng, dist_img);
    int img_id =rand_img();


    boost::uniform_int<> dist_x(0, image_samples[img_id].width() - patch_width - 1);
    boost::uniform_int<> dist_y(0, image_samples[img_id].height() - patch_height - 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);

    vector<cv::Rect> bboxes;
    for(int i=0; i < 10; i++) {
      cv::Rect bbox(rand_x(), rand_y(), patch_width, patch_height);

      float ratio_foreground = cv::countNonZero(masks[img_id](bbox))/
                               static_cast<float>(patch_width*patch_height);
      if( ratio_foreground < 0.15 ) {
        FeatureSample s( &image_samples[img_id], bbox, -1, 0 );
        training_sampels.push_back(s);
        bboxes.push_back(bbox);
      }
    }

    if( false ){
      Mat img = image_samples[img_id].get_feature_channel(0).clone();
      for(int i=0; i < bboxes.size(); i++) {
       cv::rectangle(img, bboxes[i], cv::Scalar(255, 255, 255, 0));
      }
      imshow("img", img);
      cv::waitKey(0);
    }
  }
}

int reassinge_class_labels(vector<FeatureSample>& training_sampels) {

  vector<int> class_hist;
  vector<int> new_class_hist;
  vector<int> new_labels;

  int lable_count = 0;
  for(int i=0; i < training_sampels.size(); i++) {
    int l = training_sampels[i].get_label();
    if(l >= class_hist.size()) {
      class_hist.resize(l+1, 0);
      new_labels.resize(l+1, 0);
    }

    class_hist[l] += 1;
    if(class_hist[l] == 1) {
      new_labels[l] = lable_count;
      lable_count++;
      new_class_hist.push_back(0);
    }

    CHECK_LT(l, new_labels.size() );
    CHECK_LT(l, class_hist.size() );

    new_class_hist[ new_labels[l] ] += 1;
    training_sampels[i].set_label(new_labels[l]);
  }

  LOG(INFO) << "reduced labes from " << class_hist.size() << " too " << lable_count;
  return lable_count;
}

void create_training_sampels(const std::vector<BoxAnnotation>& annotations,
                             vector<Image>& image_samples,
                             const vector<cv::Mat_<uchar> > masks,
                             vector<FeatureSample>& training_sampels,
                             int min_samples_per_class = 50,
                             int max_samples_per_class = 5000) {


  vector< vector<FeatureSample > > training_sampels_per_class;
  for(int i=0; i < annotations.size(); i++) {
    for(int j=0; j < annotations[i].bboxs.size(); j++) {
      Rect bbox = annotations[i].bboxs[j];
      int label = annotations[i].labels[j];
      if( label >= training_sampels_per_class.size()) {
        training_sampels_per_class.resize(label+1);
      }
      CHECK_EQ(masks[i].cols, image_samples[i].width() );
      CHECK_EQ(masks[i].rows, image_samples[i].height() );

      // center
      float ratio_foreground = cv::countNonZero(masks[i](bbox))/
          static_cast<float>(bbox.width*bbox.height);
      if(ratio_foreground > 0.5) {
        FeatureSample s( &image_samples[i], bbox, label, annotations[i].confidences[j]  );
        training_sampels_per_class[label].push_back( s);
      }
    }
  }

  LOG(INFO) << "pos samples created, now ordering";
  LOG(INFO) << "min_samples_per_class: " << min_samples_per_class;

  // sort based on confidence
  for(int i_class=0; i_class < training_sampels_per_class.size(); i_class++) {
    std::cout << i_class <<":" <<training_sampels_per_class[i_class].size() <<", ";
    if(training_sampels_per_class[i_class].size() < min_samples_per_class) {
      continue;
    }
    std::sort(training_sampels_per_class[i_class].begin(),
              training_sampels_per_class[i_class].end(),
              more_confident());

    // resize
    if(training_sampels_per_class[i_class].size() > max_samples_per_class) {
      training_sampels_per_class[i_class].resize(max_samples_per_class);
    }

//    LOG(INFO) << i_class << " added " << training_sampels_per_class[i_class].size()<< " samples";
    for(int i_sample=0; i_sample < training_sampels_per_class[i_class].size(); i_sample++) {
      training_sampels.push_back(training_sampels_per_class[i_class][i_sample] );
    }
  }
  std::cout << std::endl;

}

void eval_single_image_fast(Forest<FeatureSample>* forest,
    Image* image, cv::Mat_<uchar>* mask_ptr,
    BoxAnnotation* annotation,
    int num_classes = 100,
    float min_score_value = -0.0) {

  BoxAnnotation& ann = *annotation;
  const cv::Mat_<uchar>& mask = *mask_ptr;

  // eval forest
  int step_size = 3;
  cv::Rect roi(0,0, image->width(), image->height());
  ForestParam param = forest->getParam();
  cv::Rect sliding_window = cv::Rect(0,0, param.patch_width, param.patch_height);
  for (int x = roi.x; x < roi.x + roi.width - sliding_window.width; x += step_size) {
    for (int y = roi.y; y < roi.y + roi.height - sliding_window.height; y += step_size) {

      cv::Point center(x+sliding_window.width/2, y+sliding_window.height/2);
      if(mask(center) == 0 ) {
        continue;
      }
      sliding_window.x = x;
      sliding_window.y = y;
      FeatureSample s(image, sliding_window);
      std::vector<FeatureSample::Leaf*> leafs;
      forest->evaluate(&s, leafs);

      std::vector<float> class_hist(num_classes, 0);
      for(int i=0; i < leafs.size();i++) {
        const FeatureSample::Leaf* l = leafs[i];
        for(int j=0; j < l->class_hist.size(); j++){
          class_hist[j] += (l->class_hist[j] / static_cast<float>(l->num_samples) );
        }
      }

      for(int i_lable =0; i_lable < num_classes; i_lable++) {
        float p = class_hist[i_lable] / (leafs.size());
        if(p > min_score_value) {
          ann.bboxs.push_back(sliding_window);
          ann.labels.push_back(i_lable);
          ann.confidences.push_back(p);
        }
      }
    }
  }
}


void eval_single_image(Forest<FeatureSample>* forest,
    Image* image, cv::Mat_<uchar>* mask,
    BoxAnnotation* annotation,
    int num_classes = 100,
    float min_score_value = -0.0) {

  // eval image
  cv::vector<cv::Mat> voting_maps;
  cv::Mat foreground_map;
  learning::forest::utils::eval_mc_forest( *forest, *image,
      num_classes, 2, voting_maps, foreground_map, false);

  ForestParam param = forest->getParam();
  Rect rect_size(0,0,param.patch_width, param.patch_height );
  Rect img_size(0,0,image->width(), image->height());
  // multiply with mask
  if(mask->data) {
    Mat tmp;
    //convert mask to float, first we have to normalize it between 0 and 1
    normalize(*mask, tmp, 0, 1, CV_MINMAX);
    tmp.convertTo(tmp, voting_maps[0].type() );
    for(int i=0; i < voting_maps.size(); i++) {
      multiply(tmp, voting_maps[i], voting_maps[i]);
    }
  }

  BoxAnnotation& ann = *annotation;
  // maxima supression
  for(int i_label=0; i_label < voting_maps.size(); i_label++) {
    for(int i=0; i < 30; i++) {
      Point max;
      double max_v;
      minMaxLoc(voting_maps[i_label], 0, &max_v, 0, &max);
      if(max_v < min_score_value)
        break;

      // non max suppression
      Rect box(max.x - rect_size.width/2,
               max.y - rect_size.height/2,
               rect_size.width, rect_size.height);

      Rect inter = vision::geometry_utils::intersect(box,img_size);
      if(inter.height == rect_size.height && inter.width == rect_size.width) {
        ann.bboxs.push_back(box);
        ann.labels.push_back(i_label);
        ann.confidences.push_back(max_v);
      }
      Rect small_inter = inter;
      small_inter.width /= 2;
      small_inter.height /= 2;
      small_inter.x += small_inter.width/2;
      small_inter.y += small_inter.height/2;
      voting_maps[i_label](small_inter).setTo(0.0);
    }
  }
}

void eval_forest_and_create_boxes(fs::path tree_save_dir,
                                  fs::path forest_config_file,
                                  vector<Image> image_samples,
                                  vector<cv::Mat_<uchar> > masks,
                                  std::vector<BoxAnnotation>& annotations,
                                  float min_score_value = 0 ) {

  // loading old forest
  ForestParam param;
  CHECK(learning::forest::loadConfigFile(forest_config_file.string(), param));

  Forest<FeatureSample> forest;
  CHECK(forest.load(tree_save_dir.string()+"/tree_", param) );
  LOG(INFO) << "done loading ";


  // get num classes
  int num_classes =0;
  std::vector<std::vector<const FeatureSample::Leaf*> > leafs;
  forest.get_all_leafs(leafs);
  for(int i=0; i < leafs.size(); i++) {
    for(int j=0; j < leafs[i].size(); j++) {
      num_classes = std::max( static_cast<int>(leafs[i][j]->class_hist.size()),
          num_classes );
    }
  }
  LOG(INFO) << "number of classes: " << num_classes;

  int num_threads = ::utils::system::get_available_logical_cpus();
  if( num_threads <= 1) {
    for(int img_id =0; img_id < annotations.size(); ++img_id) {
      eval_single_image_fast(&forest, &image_samples[img_id], &masks[img_id],
                        &annotations[img_id], num_classes, min_score_value );
    }
  }else{
    boost::thread_pool::executor e(num_threads);
    for(int img_id =0; img_id < annotations.size(); ++img_id) {
      e.submit(boost::bind( &eval_single_image_fast, &forest, &image_samples[img_id],
                            &masks[img_id], &annotations[img_id], num_classes, min_score_value));
    }
    e.join_all();
  }
}

void train_forest( vector<FeatureSample>& training_sampels,
                  fs::path forest_config_file,
                  int i_tree,
                  fs::path tree_save_dir) {

  ForestParam param;
  CHECK(learning::forest::loadConfigFile(forest_config_file.string(), param));
  LOG(INFO) << "config file parsed. " << forest_config_file;
  param.patch_height = training_sampels[0].get_roi().height;
  param.patch_width = training_sampels[0].get_roi().width;

  LOG(INFO) << "param.patch " << param.patch_width << ", " << param.patch_height;


  string tree_save_path( boost::str(boost::format("%s/tree_%03d.txt") % tree_save_dir.string() % i_tree ));
  LOG(INFO) << tree_save_path;

  // create pointers
  vector<FeatureSample*> samples_ptr;
  for(int i=0; i < training_sampels.size(); i++) {
    samples_ptr.push_back(&training_sampels[i]);

  }
  random_shuffle( samples_ptr.begin(), samples_ptr.end() );


  Timing jobTimer;
  jobTimer.start();

  boost::mt19937 rng;
  rng.seed(i_tree +1);

  Tree<FeatureSample> tree(samples_ptr, param, &rng, tree_save_path, jobTimer);
}

DEFINE_int32(i_round, 0, "round ");
DEFINE_int32(i_tree, 1, "tree ");
DEFINE_int32(num_img, 1000000, "num_img ");
DEFINE_int32(max_samples_per_class, 5000, "max_sample_per_class ");
DEFINE_int32(min_samples_per_class, 100, "min_sample_per_class ");
DEFINE_double(min_sample_probability, 0.0, "min_sample_probability ");


DEFINE_string(img_file_name,      "/srv/glusterfs/mdantone/data/lookbook/patch_mining/48_48_1000patches_1000img_k50_mask_spatial.centers", "input");
DEFINE_string(forest_config_file, "/home/mdantone/scratch/grid/features/test/config.txt", "outputfile");
DEFINE_string(tree_save_dir,      "/home/mdantone/scratch/grid/features/test/forest_0", "outputfile");
DEFINE_string(tree_save_dir_last_round,      "", "outputfile");
DEFINE_string(print_path, "", "print_path");

int main(int argc, char** argv){
  google::InstallFailureSignalHandler();
  google::LogToStderr();
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  const uint32_t i_round = FLAGS_i_round;
  const uint32_t i_tree = FLAGS_i_tree -1;
  const uint32_t num_img = FLAGS_num_img;

  const uint32_t max_samples_per_class = FLAGS_max_samples_per_class;
  const uint32_t min_samples_per_class = FLAGS_min_samples_per_class;


  const fs::path img_file_name = FLAGS_img_file_name;
  const fs::path forest_config_file = FLAGS_forest_config_file;
  const fs::path tree_save_dir = FLAGS_tree_save_dir;
  const fs::path tree_save_dir_last_round = FLAGS_tree_save_dir_last_round;
  fs::path print_path = FLAGS_print_path;

  const float min_sample_probability = FLAGS_min_sample_probability;

  vector<Image> image_samples;
  vector<FeatureSample> training_sampels;

  LOG(INFO) << "i_round: " << i_round;
  LOG(INFO) << "i_tree: " << i_tree;
  LOG(INFO) << "img_file_name: " << img_file_name;
  LOG(INFO) << "forest_config_file: " << forest_config_file;
  LOG(INFO) << "tree_save_dir: " << tree_save_dir;

  ForestParam param;
  CHECK(learning::forest::loadConfigFile(forest_config_file.string(), param));
  LOG(INFO) << "config file parsed. " << forest_config_file;

  std::vector<BoxAnnotation> annotations;

  if(i_round == 0) {
    LOG(INFO) << "reading from file " << img_file_name;
    CHECK(::utils::serialization::read_binary_archive(img_file_name.string(), annotations) );
  }else{
    vector<Annotation> body_annotations;
    load_annotations(body_annotations, img_file_name.string());
    for(int i=0; i < body_annotations.size(); i++) {
      BoxAnnotation ann;
      ann.img_url = body_annotations[i].url;
      ann.parts = body_annotations[i].parts;
      annotations.push_back(ann);
    }
  }
  if( annotations.size() > num_img)
    annotations.resize(num_img);

  //load all images
  LOG(INFO) << "loading images " << annotations.size();

  vector<cv::Mat> images(annotations.size());
  for(int i=0; i < annotations.size(); i++) {
    images[i] = imread( annotations[i].img_url );
    CHECK(images[i].data);
  }

  // creating Image_samples
  create_image_sample_mt(images, param.features, image_samples );
  LOG(INFO) << image_samples.size() << " imagesamples created. ";
  CHECK_EQ(image_samples.size(), annotations.size());

  // createing mask
  vector<cv::Mat_<uchar> > masks(annotations.size());
  for(int i=0; i < annotations.size(); i++) {
    get_mask( images[i], annotations[i].parts, masks[i], -1);
  }



  // eval old forest and create new trainingsset
  if(i_round > 0) {
    eval_forest_and_create_boxes(tree_save_dir_last_round,
                                 forest_config_file,
                                 image_samples,
                                 masks,
                                 annotations,
                                 min_sample_probability);
  }

  LOG(INFO) << "sample positive images";
  create_training_sampels(annotations,
      image_samples,
      masks,
      training_sampels,
      min_samples_per_class,
      max_samples_per_class);


  int num_class = reassinge_class_labels(training_sampels);

  LOG(INFO) << "sample negative images";
  create_neg_trainings_samples(image_samples,
      masks,
      training_sampels,
      i_tree);

  int max_img = std::min(25, std::max(1, 1000 / num_class  ) );
  if(print_path != "" && i_tree == 0) {
    int label = -1;
    int count = 0;
    for(int i=0; i < training_sampels.size(); i++) {
      int l = training_sampels[i].get_label();
      if( l > label) {
        count = 0;
        label = l;
      }else{
        count++;
      }
      if(count < max_img) {
        string save_path( boost::str(boost::format("%1%/%2%_%3%.jpg") % print_path.string() % label % count ));
        Mat img = training_sampels[i].get_feature_channel(0);
        cv::imwrite(save_path, img(training_sampels[i].get_roi()) );
      }
    }
//    return 0;
  }

  // train next round
  train_forest(training_sampels,
              forest_config_file,
              i_tree,
              tree_save_dir);


}
