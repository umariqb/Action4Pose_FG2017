/*
 * body_sample.hpp
 *
 *  Created on: Mar 7, 2013
 *      Author: mdantone
 */

#include "cpp/body_pose/body_sample.hpp"

#include "cpp/utils/string_utils.hpp"

using namespace learning::forest;

using vision::features::PixelComparisonFeature;
using vision::features::SURFFeature;

struct IsPosSampleFunction {
  inline int operator()(const Sample* s) const {
    if( s->is_pos()){
      return 1;
    }else{
      return 0;
    }
  }
};

void BodySample::generate_split(const std::vector<BodySample*>& data,
    int patch_width, int patch_height,
    int min_child_size, int split_mode, int depth,
    const std::vector<float> class_weights,
    int split_id, ThresholdSplit<PixelComparisonFeature>* split) {

  boost::mt19937 rng(abs(split_id + 1) * (depth+1)* data.size());

  // generate the split
  int num_feat_channels = data[0]->get_images().size();
  int num_thresholds = 50;
  int margin = 0;

  split->depth = depth;
  split->margin = margin;
  split->patch_width = patch_width;
  split->patch_height = patch_height;
  split->num_thresholds = num_thresholds;

  int w = data[0]->get_width();
  int h = data[0]->get_height();
  split->feature.generate(w, h, &rng, num_feat_channels);

  // train the
  split->train(data,split_mode, depth, class_weights, min_child_size, &rng);

}


void BodySample::generate_split(const std::vector<BodySample*>& data,
    int patch_width, int patch_height,
    int min_child_size, int split_mode, int depth,
    const std::vector<float> class_weights,
    int split_id,  SVNSplit<SURFFeature>* split) {

  int seed = abs(split_id + 1) * abs(depth+1)* data.size();
  boost::mt19937 rng(seed);
  // generate the split
  int num_feat_channels = data[0]->get_images().size();

  int w = data[0]->get_width();
  int h = data[0]->get_height();

  SURFFeature f;
  f.generate(w, h, &rng, num_feat_channels);
  split->set_feature(f);

  ::utils::liblinear::solver_type::T solver = ::utils::liblinear::solver_type::L2R_L2LOSS_SVC;

  split->train(data, IsPosSampleFunction(), &rng, solver);
  split->evaluate(data, class_weights, split_mode, depth);

}

int BodySample::get_width() {
  return roi.width;
}
int BodySample::get_height() {
  return roi.height;

}

void BodySample::extract(const SURFFeature* feature,
    std::vector<float>& desc)  const {
  feature->extract( get_feature_channels(), get_roi(), desc);
}

void BodySample::create_leaf(Leaf& leaf,
    const std::vector<BodySample*>& set,
    const std::vector<float>& class_weights,
    int leaf_id) {

  bool print = (leaf.num_samples == 0);
  leaf.num_samples = set.size();


  cv::Point_<int> mean;
  double ssd;
  int pos = PartSample::sum_of_square_diff(set, &mean, &ssd);
  leaf.offset = mean;
  leaf.variance = static_cast<float>(ssd);

  float weight = 1.0;
  if ( class_weights.size() > 0 )
    weight = class_weights[0];

  double num_pos = static_cast<double>(pos);
  double num_neg = static_cast<double>(set.size() - pos);
  leaf.forground = num_pos / (num_neg*weight + num_pos );


  if(print) {
    LOG(INFO) << "num_pos : " << pos <<  ", ratio : " << leaf.forground << ", samples: " << leaf.num_samples;
    if(leaf.class_hist.size() > 1 ){
      LOG(INFO) << "class hist: " << ::utils::VectorToString(leaf.class_hist);
    }
  }

  CHECK(leaf.num_samples > 0 );
}

void  BodySample::show() {
  cv::imshow("sample", get_feature_channel(0)(roi));
  cv::Mat img = get_feature_channel(0).clone();

  int x = roi.x + roi.width/2 + offset.x;
  int y = roi.y + roi.height/2  + offset.y;
  cv::circle(img, cv::Point_<int>(x, y), 3, cv::Scalar(255, 255, 255, 0), -1);

  cv::rectangle(img, roi, cv::Scalar(255, 255, 255, 0));
  cv::imshow("image", img);
  cv::waitKey(0);
}
