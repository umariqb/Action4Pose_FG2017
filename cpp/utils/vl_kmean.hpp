/*
 * vl_kmean.hpp
 *
 *  Created on: Oct 23, 2013
 *      Author: mdantone
 */

#ifndef VL_KMEAN_HPP_
#define VL_KMEAN_HPP_

#include "opencv2/core/core.hpp"

namespace utils {
namespace kmean {


bool vl_kmean(const cv::Mat_<float>& features,
                int voc_size,
                cv::Mat_<float>& vocabulary,
                int max_num_iterations = 3);

bool vl_kmean(const cv::Mat_<float>& features,
              int voc_size,
              cv::Mat_<float>& vocabulary,
              std::vector<uint32_t>& labels,
              std::vector<float>& distances,
              int max_num_iterations = 3);


}
} /* namespace utils */
#endif /* VL_KMEAN_HPP_ */
