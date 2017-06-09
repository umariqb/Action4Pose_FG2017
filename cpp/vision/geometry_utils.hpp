/*
 * geometry_utils.hpp
 *
 * 	Created on: Apr 26, 2013
 *      Author: gandrada
 */

#ifndef VISION_GEOMETRY_UTILS_HPP_
#define VISION_GEOMETRY_UTILS_HPP_

#include <opencv2/opencv.hpp>

namespace vision {
namespace geometry_utils {

cv::Rect intersect(const cv::Rect& r1, const cv::Rect& r2);

// return true if r2 is within r1, otherwise 0;
bool check_intersection(const cv::Rect& r1, const cv::Rect& r2);


// calculate the euclidean distance between two points
inline float dist(const cv::Point& a, const cv::Point&b) {
  return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

cv::Rect get_bbox_containing_points(const std::vector<cv::Point>& points );


int intersection_area(const cv::Rect detection, const cv::Rect ground_truth);
int union_area(const cv::Rect detection, const cv::Rect ground_truth);
float intersection_over_union(const cv::Rect detection, const cv::Rect ground_truth);


} // namespace geometry_utils
} // namespace vision

#endif /* VISION_GEOMETRY_UTILS_HPP_ */
