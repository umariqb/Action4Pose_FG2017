/*
 * body_sample.hpp
 *
 *  Created on: Mar 7, 2013
 *      Author: mdantone
 */

#ifndef BODY_SAMPLE_HPP_
#define BODY_SAMPLE_HPP_

#include "cpp/learning/common/image_sample.hpp"
#include "cpp/learning/forest/sample.hpp"
#include "cpp/learning/forest/part_sample.hpp"
#include "cpp/learning/forest/svn_split.hpp"
#include "cpp/learning/forest/simple_split.hpp"
#include "cpp/vision/features/simple_feature.hpp"

class BodySample  :
   public learning::forest::PartSample,
   public learning::common::ImageSample {
public:

// typedef learning::forest::ThresholdSplit<learning::forest::SimplePixelFeature> Split;
  typedef learning::forest::SVNSplit<vision::features::SURFFeature> Split;

  typedef learning::forest::ClassAndRegressionLeaf Leaf;


  BodySample(){};

  int get_width();
  int get_height();

  BodySample(const learning::common::Image* patch,
      cv::Rect roi,
      bool is_pos = true ):
        learning::forest::PartSample(is_pos),
        learning::common::ImageSample(patch, roi){
  }

  BodySample(const learning::common::Image* patch, cv::Rect roi,
      cv::Point_<int> offset ):
        learning::forest::PartSample(offset),
        learning::common::ImageSample(patch, roi){
  }

  static void generateSplit(const std::vector<BodySample*>& data,
                int patch_width,
                int patch_height,
                int min_child_size,
                int split_mode,
                int depth,
                const std::vector<float> class_weights,
                int split_id,
                Split* split){
    generate_split( data, patch_width, patch_height,
        min_child_size, split_mode, depth,
        class_weights, split_id, split);
  }

  static void generate_split(const std::vector<BodySample*>& data,
                int patch_width,
                int patch_height,
                int min_child_size,
                int split_mode,
                int depth,
                const std::vector<float> class_weights,
                int split_id,
                learning::forest::ThresholdSplit<vision::features::PixelComparisonFeature>* split);

  static void generate_split(const std::vector<BodySample*>& data,
                int patch_width,
                int patch_height,
                int min_child_size,
                int split_mode, int depth,
                const std::vector<float> class_weights,
                int split_id,
                learning::forest::SVNSplit<vision::features::SURFFeature>* split);

  static void create_leaf(Leaf& leaf,
      const std::vector<BodySample*>& set,
      const std::vector<float>& class_weights,
      int leaf_id = 0);

  void extract(const vision::features::SURFFeature* feature,
      std::vector<float>& desc)  const;


  void show();

  virtual ~BodySample(){};
};

#endif /* BODY_SAMPLE_HPP_ */
