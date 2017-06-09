/*
 * fashion_segmentation.cpp
 *
 *  Created on: Feb 20, 2014
 *      Author: mdantone
 */

#include "fashion_segmentation.hpp"
#include "cpp/vision/geometry_utils.hpp"
#include "cpp/utils/thread_pool.hpp"
#include "cpp/utils/system_utils.hpp"
#include "cpp/vision/features/global/color_hist.hpp"
#include "cpp/third_party/felz_pix/felz_pix.hpp"
#include "cpp/fashion/skin/SkinSegmentation.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

using namespace std;
using namespace cv;

namespace vision {
namespace segmentation {

int compute_super_pixel(const cv::Mat_<cv::Vec3b>& img,
                        const SuperPixelParam param,
                        cv::Mat_<int32_t>& components) {
  float sigma = param.sigma();
  float c = param.c();
  int min_size = std::max(img.cols, img.rows) * param.min_size();
  int num_components = awesomeness::third_party::felz_pix::segment_image(img, sigma, c, min_size, &components);
  return num_components;
}
SuperPixelSegmentatinParam get_default_param() {
  SuperPixelSegmentatinParam param;
  param.mutable_super_pixel_param();
  param.mutable_gabcut_param();
  param.mutable_skin_param();
  param.mutable_face_param();
  param.mutable_body_pose_param();
  return param;
}

bool segment_fashion_image( const cv::Mat_<cv::Vec3b>& img,
                    const std::vector<cv::Point>& bodyparts,
                    const float bodyestimation_score,
                    const SuperPixelSegmentatinParam param,
                    cv::Mat_<uchar>& mask,
                    bool debug) {
  cv::Mat_<uchar> input_mask;
  if(param.has_body_pose_param() &&
     param.body_pose_param().min_body_pose_score() > bodyestimation_score) {

    input_mask = cv::Mat_<uchar>(img.rows, img.cols );
    input_mask.setTo(cv::Scalar(cv::GC_PR_BGD));


    if(param.body_pose_param().upperbody_is_forground() ) {
      set_torso(bodyparts, input_mask, Scalar(GC_PR_FGD) );
      set_arm(  bodyparts, input_mask, Scalar(GC_PR_FGD) );
    }else{
      set_torso(bodyparts, input_mask, Scalar(GC_PR_BGD) );
      set_arm(  bodyparts, input_mask, Scalar(GC_PR_BGD) );

    }

    if(param.body_pose_param().lowerbody_is_forground() ) {
      set_legs(bodyparts, input_mask, Scalar(GC_PR_FGD) );
    }else{
      set_legs(bodyparts, input_mask, Scalar(GC_PR_BGD) );
    }

  }




  return segment_fashion_image(img, input_mask, param, mask, debug);
}

bool segment_fashion_image( const cv::Mat_<cv::Vec3b>& img,
                    cv::Mat_<uchar>& input_mask,
                    const SuperPixelSegmentatinParam param,
                    cv::Mat_<uchar>& mask,
                    bool debug){

  // allocate the mask, if the mask was not preset.
  if(!input_mask.data) {
    input_mask = cv::Mat_<uchar>(img.rows, img.cols );
    input_mask.setTo(cv::Scalar(cv::GC_PR_FGD));
  }


  // performe superpixel
  CHECK(param.has_super_pixel_param());
  cv::Mat_<int32_t> components;
  int num_components = compute_super_pixel(img, param.super_pixel_param(), components);


  // adding boarder pixel to background
  if(param.horizontal_boarders_is_background()) {
    set_horizontal_boarder_pixel(components, cv::GC_PR_BGD, input_mask);
  }
  if(param.vertical_boarders_is_background()) {
    set_vertical_boarder_pixel(components, cv::GC_PR_BGD, input_mask);
  }


  // skin is usualy background
  if(param.has_skin_param() ) {
    set_skincolor_pixel(img, components, num_components,
                        cv::GC_PR_BGD, param.skin_param(),input_mask);
  }

  // face is background as well
  std::vector < cv::Rect > faces_bboxes;
  if(param.has_face_param()) {
    set_faces(img, components, num_components,
              cv::GC_PR_BGD,param.face_param(),
              input_mask, faces_bboxes);
  }

  CHECK(param.has_gabcut_param());
  segment_fashion_image(img, input_mask, param.gabcut_param(), mask, debug );

  if(debug) {
    // get superpixel img
    cv::Mat_<cv::Vec3b> sp_img, sp_img_halton;

    awesomeness::third_party::felz_pix::superpixel_to_mean_rgb(img, components, num_components, sp_img);
    awesomeness::third_party::felz_pix::superpixel_to_halton_rgb( components, sp_img_halton);

      // Generate output image
      cv::Mat foreground(img.size(),CV_8UC3,cv::Scalar(255,0,255));
      img.copyTo(foreground,mask); // bg pixels not copied

      if(param.has_face_param()) {
        LOG(INFO) << faces_bboxes.size() << " faces found";
        // draw face bbox
        for (unsigned int i = 0; i < faces_bboxes.size(); i++) {
          cv::rectangle(foreground, faces_bboxes[i], cv::Scalar(255, 255, 255, 0));
        }
      }

      cv::imshow("sp", sp_img);
      cv::imshow("halton", sp_img_halton);

      cv::imshow("mask", mask);

      cv::Mat mask_copy = input_mask.clone();
      mask_copy *= 100;
      cv::imshow("mask_input", mask_copy);

      cv::imshow("img", foreground);
      cv::imshow("org", img);

      cv::waitKey(0);
  }

  return true;

}

bool segment_fashion_image(const cv::Mat_<cv::Vec3b>& img,
                    const cv::Mat_<uchar>& input_mask,
                    const GrabCutParam param,
                    cv::Mat_<uchar>& mask,
                    bool debug) {

  // set 'probably' forground and background
  mask = input_mask.clone();


  // grabcut segmentation
  cv::Mat bg_m, fg_m;
  try {
    cv::grabCut(img,
      mask,             // segmentation result and initialization
      cv::Rect(0,0,0,0),// we dont use this rectangle
      bg_m, fg_m,       // models
      param.grabcut_rounds(),                // number of iterations
      cv::GC_INIT_WITH_MASK); // use mark
  } catch (...) {
    // sometimes it crashes, than we set it back to default.
    mask.setTo(cv::Scalar(255));
    LOG(INFO) << "error during segmentation";
    return false;
  }

  // Get the pixels marked as likely foreground
  for(int x = 0; x < mask.cols; x++) {
    for(int y = 0; y < mask.rows; y++) {
      cv::Point p(x,y);
      if(mask(p) == cv::GC_PR_FGD or
         mask(p) == cv::GC_FGD) {
        mask(p) = 255;
      }else{
        mask(p) = 0;
      }
    }
  }

  cv::Mat kernel(cv::Size(param.post_processing_kernel_size(),
                          param.post_processing_kernel_size()),
                          CV_8UC1);
  kernel.setTo(cv::Scalar(1));
  cv::erode(mask, mask, kernel);
  cv::dilate(mask, mask, kernel);

  return true;
}

void set_vertical_boarder_pixel(const cv::Mat_<int32_t>& components,
                                unsigned char value,
                                cv::Mat_<uchar>& mask){
  for(int i=0; i < components.rows; i++) {
    if(mask(i,0) != value ) {
      set_component(components, components.at<int32_t>(i,0), value, mask);
    }
    if(mask(i,components.cols-1) != value) {
      set_component(components, components.at<int32_t>(i,components.cols-1), value, mask);
    }
  }
}

void set_horizontal_boarder_pixel(const cv::Mat_<int32_t>& components,
                                unsigned char value,
                                cv::Mat_<uchar>& mask){
  for(int i=0; i < components.cols; i++) {
    if(mask(0,i) != value) {
      set_component(components, components.at<int32_t>(0,i), value, mask);
    }
    if(mask(components.rows-1, i) != value) {
      set_component(components, components.at<int32_t>(components.rows-1,i), value, mask);
    }
  }
}


void set_skincolor_pixel(const cv::Mat_<cv::Vec3b>& img,
                         const cv::Mat_<int32_t>& components,
                         int num_component,
                         unsigned char value,
                         const SkinSegmentationParam param,
                         cv::Mat_<uchar>& mask) {

  // compute skin probability map
  fashion::SkinSegmentation skin_seg;
  cv::Mat skin_prob;
  skin_seg.createSkinProbabilityMap(img, &skin_prob);

  // iterate over components and set the skin-components
  for(int i_comp =0; i_comp < num_component; ++i_comp) {
    cv::Scalar prob = get_mean_of_component(skin_prob, components, i_comp);

    if( (prob(0)/ 255.0) > param.skin_threshold()) {
      set_component(components,i_comp, value, mask);
    }
  }
}

void set_faces(const cv::Mat_<cv::Vec3b>& img,
               const cv::Mat_<int32_t>& components,
               int num_component,
               unsigned char value,
               const FaceSegmentationParam param,
               cv::Mat_<uchar>& mask,
               std::vector < cv::Rect >& faces_bboxes) {

  // image needs to be grayscale
  cv::Mat_<uchar> gray_mat;
  cv::cvtColor(img, gray_mat, CV_BGR2GRAY);

  // loaded cascade and setting parameters
  int min_feature_size = param.min_feature_size();
  int min_neighbors = param.min_neighbors();
  float search_scale_factor = param.search_scale_factor();
  std::string path_face_cascade = param.path_face_cascade();

  // detect faces
  cv::CascadeClassifier face_cascade;
  if(face_cascade.load(path_face_cascade)) {
    int flags = CV_HAAR_SCALE_IMAGE;
    CvSize minFeatureSize = cv::Size(min_feature_size, min_feature_size);

    face_cascade.detectMultiScale(gray_mat, faces_bboxes, search_scale_factor,
         min_neighbors, flags, minFeatureSize);
  }else{
    LOG(INFO) << "cascade not found: " << path_face_cascade;
    return;
  }

  // rescale bounding boxes
  for (unsigned int i = 0; i < faces_bboxes.size(); i++) {
    int scale_x = faces_bboxes[i].width  * param.rescale_factor_boxes();
    int scale_y = faces_bboxes[i].height * param.rescale_factor_boxes();

    cv::Rect roi = vision::geometry_utils::intersect(
        cv::Rect(faces_bboxes[i].x - scale_x, faces_bboxes[i].y - scale_y*2,
            faces_bboxes[i].width + scale_x * 2, faces_bboxes[i].height + scale_y * 2),
        cv::Rect(0, 0, img.cols, img.rows));
    faces_bboxes[i] = roi;
  }

  // setting the regions
  for (unsigned int i = 0; i < faces_bboxes.size(); i++) {
    cv::Mat_<int32_t> roi_components = components(faces_bboxes[i]);
    cv::Mat_<uchar> roi_mask = mask(faces_bboxes[i]);
    for(int x = 0; x < roi_components.cols; x++) {
      for(int y = 0; y < roi_components.rows; y++) {
        cv::Point p(x,y);
        if(roi_mask(p) != value) {
          int32_t component_id = roi_components(p);
          set_component(components,component_id, value, mask);
        }
      }
    }
//    mask(faces_bboxes[i]).setTo(value);
  }
}




void set_component(const cv::Mat_<int32_t>& components,
                   int component_id,
                   unsigned char value,
                   cv::Mat_<uchar>& mask) {
  cv::Mat_<uchar> m;
  cv::compare(components, component_id, m,cv::CMP_EQ);
  mask.setTo(cv::GC_PR_BGD, m);
}

cv::Scalar get_mean_of_component(const cv::Mat input,
                                 const cv::Mat_<int32_t>& components,
                                 int component_id) {
  cv::Mat_<uchar> m;
  cv::compare(components, component_id, m,cv::CMP_EQ);
  return cv::mean(input, m);
}



bool torso_is_present(vector<Point> part_loc, Rect roi) {
  for(int i=0; i < part_loc.size(); i++) {
    if(!roi.contains(part_loc[i])) {
      return false;
    }
  }
  return true;
}

void set_legs(const vector<Point >& part_loc,
              Mat_<uchar>& mask,
              Scalar forground){

  vector<pair<int, int> > joint_index;
  joint_index.push_back( make_pair(4,11));
  joint_index.push_back( make_pair(11,12));
  joint_index.push_back( make_pair(3,9));
  joint_index.push_back( make_pair(9,10));

  for(int i=0; i < joint_index.size(); i++) {
    const Point& a = part_loc[joint_index[i].first];
    const Point& b = part_loc[joint_index[i].second];
    if(a.x >0 && b.x > 0 && a.y >0 && b.y > 0 && a.x < mask.cols && b.x < mask.cols ) {
      line(mask, a, b, forground, 3);
    }
  }
}

void set_arm(const vector<Point >& part_loc,
              Mat_<uchar>& mask,
              Scalar forground){

  vector<pair<int, int> > joint_index;
  joint_index.push_back(make_pair(2,7));
  joint_index.push_back(make_pair(7,8));
  joint_index.push_back(make_pair(1,5));
  joint_index.push_back(make_pair(5,6));

  for(int i=0; i < joint_index.size(); i++) {
    const Point& a = part_loc[joint_index[i].first];
    const Point& b = part_loc[joint_index[i].second];
    if(a.x >0 && b.x > 0 && a.y >0 && b.y > 0 && a.x < mask.cols && b.x < mask.cols ) {
      line(mask, a, b, forground, 3);
    }
  }
}

void set_torso(const std::vector<cv::Point >& part_loc,
               Mat_<uchar>& mask,
               Scalar forground){

  vector<Point> torso_corners(4);
  torso_corners[0] = part_loc[1];
  torso_corners[1] = part_loc[3];
  torso_corners[2] = part_loc[4];
  torso_corners[3] = part_loc[2];
  if (torso_is_present(torso_corners, Rect(0,0, mask.cols, mask.rows)) ) {
    const Point_<int>* ppt[1] = {&torso_corners[0]};
    int npt[] = {4};
    fillPoly(mask, ppt, npt,  1, forground );
  }else{
    LOG(INFO) << "torso is not present.";
  }

}





} /* namespace segmentation */
} /* namespace vision */
