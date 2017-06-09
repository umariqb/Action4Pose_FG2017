/*
 * feature_sample.hpp
 *
 *  Created on: Sep 30, 2013
 *      Author: mdantone
 */

#ifndef FEATURE_SAMPLE_HPP_
#define FEATURE_SAMPLE_HPP_

#include "cpp/learning/common/image_sample.hpp"
#include "cpp/learning/forest/sample.hpp"
#include "cpp/learning/forest/simple_split.hpp"
#include "cpp/vision/features/simple_feature.hpp"
namespace bodypose {
namespace features {




class FeatureSample : public learning::forest::Sample,
                      public learning::common::ImageSample {
public:
  typedef learning::forest::ThresholdSplit<vision::features::PixelComparisonFeature> Split;
//  typedef learning::forest::ThresholdSplit<vision::features::PixelValueFeatuers> Split;
  typedef learning::forest::ClassificationLeaf Leaf;

  FeatureSample():
    Sample(),learning::common::ImageSample() { }

  FeatureSample(const learning::common::Image* patch,
                cv::Rect roi,
                int label_ = 0,
                float confident_ = 0.0) :
    Sample(label_ >= 0),
    learning::common::ImageSample(patch, roi),
    label(label_),
    confident(confident_){
  }

  static void generateSplit(const std::vector<FeatureSample*>& data,
                            int patch_width,
                            int patch_height,
                            int min_child_size,
                            int split_mode,
                            int depth,
                            const std::vector<float> class_weights,
                            int split_id,
                            Split* split);

  static double entropy(const std::vector<FeatureSample*>& set,
      float alpha, int num_classes);

  static double evalSplit(const std::vector<FeatureSample*>& setA,
      const std::vector<FeatureSample*>& setB,
      const std::vector<float>& class_weights,
      int split_mode, int depth);

  int get_label() const {
    return label;
  }

  void set_label(int label_ ) {
    label = label_;
  }

  float get_confidents() const {
    return confident;
  }
  static void create_leaf(learning::forest::ClassificationLeaf& leaf,
                        const std::vector<FeatureSample*>& set,
                        const std::vector<float>& class_weights,
                        int leaf_id = 0) ;


  void show();

  virtual ~FeatureSample(){};

private:
  int label;
  float confident;
};


} /* namespace features */
} /* namespace bodypose */
#endif /* FEATURE_SAMPLE_HPP_ */
