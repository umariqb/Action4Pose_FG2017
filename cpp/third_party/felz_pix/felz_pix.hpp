/*
 * felz_pix.hpp
 *
 *  Created on: Aug 7, 2013
 *      Author: lbossard
 */

#ifndef AWESOMENESS__THIRD_PARTY__FELZ_PIX_HPP_
#define AWESOMENESS__THIRD_PARTY__FELZ_PIX_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace awesomeness {
namespace third_party {
namespace felz_pix {


/**
 *
 * Segment an image
 *
 * Returns a color image representing the segmentation.

 * @param img image to segment
 * @param sigma to smooth the image
 * @param k constant for threshold function
 * @param min_size minimum component size (enforced by post-processing stage).
 * @param components_ image representing segmentation. each segment has its own id.
 * @return number of connected components
 */
int segment_image(
    const cv::Mat_<cv::Vec3b>& img,
    const float sigma,
    const float k,
    const int min_size,
    cv::Mat_<int32_t>* components_);




void superpixel_to_mean_rgb(const cv::Mat_<cv::Vec3b>& img,
                            const cv::Mat_<int32_t>& labels,
                            int num_components,
                            cv::Mat_<cv::Vec3b>& dest_img);

void superpixel_to_halton_rgb(const cv::Mat_<int32_t>& labels,
                             cv::Mat_<cv::Vec3b>& dest_img);


} /* namespace felz_pix */
} /* namespace third_party */
} /* namespace awesomeness */
#endif /* AWESOMENESS__THIRD_PARTY__FELZ_PIX_HPP_ */
