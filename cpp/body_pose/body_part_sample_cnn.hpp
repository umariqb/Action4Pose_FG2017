/*
 * body_part_sample_cnn.hpp
 *
 *  Created on: September 11, 2015
 *      Author: uiqbal
 *
 * Creating this file since CNN features are very large,
 * e.g., if we take the features from conv-1 of bvlc_reference_caffe_net
 * we will end up with 55x55x96 feature channels, and after resizing it to
 * input image size IMG_ROWSxIMG_COLSx96 which is already quite a lot to be
 * stored in memory for all training images. The situation gets even worse
 * when we will take features from multiple layers,
 * e.g., conv-1,conv-2,conv-3,conv4,conv5, will end up with 1376 feature channels
 * for a single image. Therefore, it's better to save only for the selected locations
 *
 */

#ifndef BODY_PART_SAMPLE_CNN_HPP_
#define BODY_PART_SAMPLE_CNN_HPP_


#include "cpp/body_pose/body_part_sample.hpp"

class BodyPartSampleCNN :public BodyPartSample{

public:
  virtual const std::vector<cv::Mat>& get_images() const {
    return images;
  }

private:
  std::vector<cv::Mat> images;
};

#endif // BODY_PART_SAMPLE_CNN_HPP_

