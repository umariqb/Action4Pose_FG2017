/*!
 * optical_flow_utils.cpp
 *
 * Created on: July 18, 2014
 *
 *  @Author by: uiqbal
 *  Date:06.06.2014
 */

#ifndef OPTICAL_FLOW_UTILS_H
#define OPTICAL_FLOW_UTILS_H
#include <opencv2/opencv.hpp>

namespace vision {

void compute_optical_flow(std::vector<cv::Mat>& frames, std::vector<cv::Mat>& optical_flow, std::string save_path = "");
bool load_optical_flow(std::string path, std::vector<cv::Mat>& optical_flow);
bool save_optical_flow(std::vector<cv::Mat>& optical_flow, std::string save_path);

} /* namespace vision*/
#endif // OPTICAL_FLOW_UTILS_H
