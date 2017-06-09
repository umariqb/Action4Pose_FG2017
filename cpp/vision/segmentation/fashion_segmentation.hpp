/*
 * fashion_segmentation.hpp
 *
 *  Created on: Feb 20, 2014
 *      Author: mdantone
 */

#ifndef FASHION_SEGMENTATION_HPP_
#define FASHION_SEGMENTATION_HPP_


#include "cpp/vision/segmentation/segmentation_param.pb.h"
#include <opencv2/opencv.hpp>


namespace vision {
namespace segmentation {

bool segment_fashion_image(const cv::Mat_<cv::Vec3b>& img,
                   const cv::Mat_<uchar>& input_mask,
                   const GrabCutParam param,
                   cv::Mat_<uchar>& mask,
                   bool debug = false);

bool segment_fashion_image( const cv::Mat_<cv::Vec3b>& img,
                    cv::Mat_<uchar>& input_mask,
                    const SuperPixelSegmentatinParam param,
                    cv::Mat_<uchar>& mask,
                    bool debug = false);

bool segment_fashion_image( const cv::Mat_<cv::Vec3b>& img,
                    const std::vector<cv::Point>& bodyparts,
                    const float bodyestimation_score,
                    const SuperPixelSegmentatinParam param,
                    cv::Mat_<uchar>& mask,
                    bool debug = false);




// static helper function

SuperPixelSegmentatinParam get_default_param();

void set_vertical_boarder_pixel(const cv::Mat_<int32_t>& components,
                                unsigned char value,
                                cv::Mat_<uchar>& mask);

void set_horizontal_boarder_pixel(const cv::Mat_<int32_t>& components,
                                unsigned char value,
                                cv::Mat_<uchar>& mask);

void set_skincolor_pixel(const cv::Mat_<cv::Vec3b>& img,
                         const cv::Mat_<int32_t>& components,
                         int num_component,
                         unsigned char value,
                         const SkinSegmentationParam param,
                         cv::Mat_<uchar>& mask);

void set_faces(const cv::Mat_<cv::Vec3b>& img,
               const cv::Mat_<int32_t>& components,
               int num_component,
               unsigned char value,
               const FaceSegmentationParam param,
               cv::Mat_<uchar>& mask,
               std::vector < cv::Rect >& faces_bboxes);


void set_component(const cv::Mat_<int32_t>& components,
                   int num_components,
                   unsigned char value,
                   cv::Mat_<uchar>& mask);

cv::Scalar get_mean_of_component(const cv::Mat input,
                                 const cv::Mat_<int32_t>& components,
                                 int component_id);



// helper function for body
bool torso_is_present(std::vector<cv::Point> part_loc, cv::Rect roi);
void set_legs(const std::vector<cv::Point >& part_loc,
              cv::Mat_<uchar>& mask,
              cv::Scalar forground = cv::Scalar(cv::GC_PR_FGD) );

void set_arm(const std::vector<cv::Point >& part_loc,
              cv::Mat_<uchar>& mask,
              cv::Scalar forground = cv::Scalar(cv::GC_PR_FGD) );

void set_torso(const std::vector<cv::Point >& part_loc,
              cv::Mat_<uchar>& mask,
              cv::Scalar forground = cv::Scalar(cv::GC_PR_FGD) );

} /* namespace segmentation */
} /* namespace vision */
#endif /* FASHION_SEGMENTATION_HPP_ */
