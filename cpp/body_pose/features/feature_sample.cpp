/*
 * feature_sample.cpp
 *
 *  Created on: Sep 30, 2013
 *      Author: mdantone
 */

#include "feature_sample.hpp"
using namespace std;
using namespace cv;

namespace bodypose {
namespace features {

void FeatureSample::generateSplit(const std::vector<FeatureSample*>& data,
      int patch_width,
      int patch_height,
      int min_child_size,
      int split_mode,
      int depth,
      const std::vector<float> class_weights,
      int split_id,
      Split* split) {

  boost::mt19937 rng(abs(split_id + 1) * (depth+1)* data.size());
  // generate the split
  int num_feat_channels = data[0]->get_images().size();
//
  int num_thresholds = 50;
  int margin = 0;
  split->initialize(&rng, patch_width, patch_height,
      num_feat_channels, num_thresholds,
      margin, depth);

  int sub_sampling = 50000;
  split->train(data,split_mode, depth, class_weights, min_child_size, &rng, sub_sampling);
  split->used_subsampling = true;
}


int create_class_hist(const std::vector<FeatureSample*>& set,
                      vector<int>& class_hist ) {
  int n_pos = 0;
  for (unsigned int i = 0; i < set.size(); ++i) {
    int l = set[i]->get_label();
    if( l >= 0){
      if(l >= class_hist.size()) {
        class_hist.resize(l+1,0);
      }
      class_hist[l]++;
      n_pos++;
    }
  }
  return n_pos;
}
void FeatureSample::create_leaf(learning::forest::ClassificationLeaf& leaf,
                                const std::vector<FeatureSample*>& set,
                                const std::vector<float>& class_weights,
                                int leaf_id) {

  leaf.num_samples = set.size();
  int n_pos_samples = create_class_hist(set, leaf.class_hist);
  leaf.forground = static_cast<float>(n_pos_samples) / set.size();
  LOG(INFO) << "forground: " << leaf.forground << "; " << ::utils::VectorToString(leaf.class_hist);
}



double FeatureSample::evalSplit(const std::vector<FeatureSample*>& setA,
    const std::vector<FeatureSample*>& setB,
    const std::vector<float>& class_weights,
    int split_mode, int depth) {

  if( setA.size() == 0 or setB.size() == 0) {
    return boost::numeric::bounds<double>::lowest();
  }

  //
  int num_pos_samples = 0;
  vector<int> class_hist;
  num_pos_samples += create_class_hist(setA, class_hist);
  num_pos_samples += create_class_hist(setB, class_hist);

  int num_classes = 0;
  for(int i=0; i < class_hist.size(); i++) {
    if(class_hist[i] > 0 ) {
      num_classes++;
    }
  }

  float alpha = static_cast<float>(num_pos_samples) / (setA.size() + setB.size() );
  double var_a = entropy(setA, alpha, num_classes);
  double var_b = entropy(setB, alpha, num_classes);
  int size_a = setA.size();
  int size_b = setB.size();
  return (var_a * size_a + var_b * size_b) / static_cast<double>(size_b + size_a);
}

double FeatureSample::entropy(const std::vector<FeatureSample*>& set,
    float alpha, int num_classes) {
  vector<int> class_hist;
  create_class_hist(set, class_hist);

  // entropy
  double n_entropy_classes = 0;
  for(int i=0; i < class_hist.size(); i++){
    double p = static_cast<float>(class_hist[i]) / set.size();
    if (p > 0)
      n_entropy_classes += p * log(p);
  }
  if(num_classes > 0 )
    n_entropy_classes /= num_classes;


  double n_entropy_pos_vs_neg = 0;
  if (alpha > 0) {
    n_entropy_pos_vs_neg = alpha * log(alpha);
  }

  return (alpha * n_entropy_classes) + (1-alpha)*n_entropy_pos_vs_neg;

}

void FeatureSample::show() {
  cv::imshow("roi", get_feature_channel(0)(roi));
  cv::Mat face = get_feature_channel(0).clone();
  cv::rectangle(face, roi, cv::Scalar(255, 255, 255, 0));
  cv::imshow("img ", face);
  cv::waitKey(0);
}
} /* namespace features */
} /* namespace bodypose */
