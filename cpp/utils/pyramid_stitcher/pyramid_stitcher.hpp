#ifndef PYRAMID_STITCHER_H
#define PYRAMID_STITCHER_H

#include "cpp/utils/pyramid_stitcher/patchwork.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

// location of a scale within a stitched feature pyramid
// can be used for data in image space or feature decscriptor space
class ScaleLocation{
    public:
        int xMin;
        int xMax;
        int yMin;
        int yMax;
        int width;
        int height;

        int planeID;
        //int scaleIdx; //TODO?
};


//image -> multiscale pyramid -> stitch to same-sized planes for Caffe convnet
Patchwork stitch_pyramid(string file, int img_minWidth, int img_minHeight,
                         int padding, int nscales, int planeDim, cv::Scalar mean_pixel = cv::Scalar(0,0,0));

Patchwork stitch_pyramid(cv::Mat, int img_minWidth, int img_minHeight,
                         int padding, int nscales, int planeDim, cv::Scalar mean_pixel = cv::Scalar(0,0,0));

Patchwork stitch_pyramid(cv::Mat image, int img_minWidth, int img_minHeight,
                         int padding, int nscales, int planeWidth, int planeHeight, cv::Scalar mean_pixel = cv::Scalar(0,0,0));

Patchwork stitch_pyramid(string file, int img_minWidth, int img_minHeight,
                         int padding, int nscales, int planeWidth, int planeHeight, cv::Scalar mean_pixel = cv::Scalar(0,0,0));



// coordinates for unstitching the feature descriptors from planes.
//      sorted in descending order of size.
//        (well, Patchwork sorts in descending order of size, and that survives here.)
vector<ScaleLocation> unstitch_pyramid_locations(Patchwork &patchwork,
                                                 int sbin);


#endif
//
