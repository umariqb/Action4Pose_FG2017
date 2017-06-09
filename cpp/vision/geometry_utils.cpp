/*
 * geometry_utils.cpp
 *
 * 	Created on: Apr 26, 2013
 *      Author: gandrada
 *  Modified on: Jul 29, 2015
 *      Modified by: uiqbal
 */

#include "cpp/vision/geometry_utils.hpp"

using namespace std;
using namespace cv;

namespace vision {
namespace geometry_utils {

Rect intersect(const Rect& r1, const Rect& r2) {
  Rect intersection;

  // find overlapping region
  intersection.x = (r1.x < r2.x) ? r2.x : r1.x;
  intersection.y = (r1.y < r2.y) ? r2.y : r1.y;
  intersection.width = (r1.x + r1.width < r2.x + r2.width) ? r1.x + r1.width
      : r2.x + r2.width;
  intersection.width -= intersection.x;
  intersection.height = (r1.y + r1.height < r2.y + r2.height) ? r1.y
      + r1.height : r2.y + r2.height;
  intersection.height -= intersection.y;

  // check for non-overlapping regions
  if ((intersection.width <= 0) || (intersection.height <= 0)) {
    intersection = cvRect(0, 0, 0, 0);
  }
  return intersection;
}

bool check_intersection(const cv::Rect& r1, const cv::Rect& r2) {
  Rect intersection = intersect(r1,r2);
  if(intersection.width == r2.width && intersection.height == r2.height) {
    return true;
  }else{
    return false;
  }
}


Rect get_bbox_containing_points(const vector<Point>& points ) {
  Point min_p = points[0];
  Point max_p = points[0];

  for(unsigned int i=1; i < points.size(); i++) {
    min_p.x = std::min(points[i].x, min_p.x);
    min_p.y = std::min(points[i].y, min_p.y);
    max_p.x = std::max(points[i].x, max_p.x);
    max_p.y = std::max(points[i].y, max_p.y);
  }
  Rect bbox = Rect(min_p.x, min_p.y, max_p.x-min_p.x, max_p.y-min_p.y );
  return bbox;
}


int intersection_area(const Rect detection, const Rect ground_truth)
{
    int detection_x2 = detection.x + detection.width;
    int detection_y2 = detection.y + detection.height;
    int ground_truth_x2 = ground_truth.x + ground_truth.width;
    int ground_truth_y2 = ground_truth.y + ground_truth.height;
    //first calculate the boundaries of the intersection of the rectangles
    int intersection_x = std::max(detection.x, ground_truth.x); //rightmost left
    int intersection_y = std::max(detection.y, ground_truth.y); //bottommost top
    int intersection_x2 = std::min(detection_x2, ground_truth_x2); //leftmost right
    int intersection_y2 = std::min(detection_y2, ground_truth_y2); //topmost bottom
    //then calculate the width and height of the intersection rect
    int intersection_width = intersection_x2 - intersection_x + 1;
    int intersection_height = intersection_y2 - intersection_y + 1;
    //if there is no overlap then return zero
    if ((intersection_width <= 0) || (intersection_height <= 0)) return 0;
    //otherwise calculate the intersection
    int intersection_area = intersection_width*intersection_height;

    return intersection_area;
}

int union_area(const Rect detection, const Rect ground_truth)
{
    int i_area = intersection_area(detection, ground_truth);
    int u_area = (detection.width+1)*(detection.height+1) + (ground_truth.width+1)*(ground_truth.height+1) - i_area;
    return u_area;
}

int union_area(const Rect detection, const Rect ground_truth, int i_area)
{
    int u_area = (detection.width+1)*(detection.height+1) + (ground_truth.width+1)*(ground_truth.height+1) - i_area;
    return u_area;
}


float intersection_over_union(const Rect detection, const Rect ground_truth)
{
    int inter_area = intersection_area(detection, ground_truth);

    //now calculate the union
    int u_area = union_area(detection, ground_truth, inter_area);

    //calculate the intersection over union and use as threshold as per VOC documentation
    float overlap = static_cast<float>(inter_area)/static_cast<float>(u_area);

    return overlap;
}


} // namespace geometry_utils
} // namespace vision

