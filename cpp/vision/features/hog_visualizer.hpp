/*
 * hog_visualizer.hpp
 *
 *  Created on: Aug 22, 2013
 *      Author: mdantone
 */

#ifndef HOG_VISUALIZER_HPP_
#define HOG_VISUALIZER_HPP_
#include <opencv2/opencv.hpp>

namespace vision {
namespace features {

void visualize_hog_features(
    const cv::Mat& origImg,
    std::vector<float>& descriptorValues,
    int block_size = 16,
    int cell_size = 8,
    int n_bins = 9,
    float zoom_factor = 3);


} /* namespace features */
} /* namespace vision */
#endif /* HOG_VISUALIZER_HPP_ */
