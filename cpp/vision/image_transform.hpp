/*
 * image_transform.hpp
 *
 *  Created on: Sep 12, 2012
 *      Author: lbossard
 */

#ifndef VISION__IMAGE_TRANSFORM_HPP_
#define VISION__IMAGE_TRANSFORM_HPP_

#include <string>

#include <opencv2/core/core.hpp>

namespace vision
{
namespace image_transform
{

struct ImageTransform
{
	virtual void operator()(cv::Mat& image) const = 0;
    virtual void operator()(cv::Mat& image, cv::Mat_<uchar>& mask) const = 0;
    virtual std::string name() const = 0;
    virtual ~ImageTransform(){};
};

struct Identity : public ImageTransform
{
	virtual void operator()(cv::Mat& image) const {};
    virtual void operator()(cv::Mat& image, cv::Mat_<uchar>& mask) const {};
    virtual std::string name() const { return "";};
};


struct Flipper : public ImageTransform
{
    virtual void operator()(cv::Mat& m) const;
    virtual void operator()(cv::Mat& image, cv::Mat_<uchar>& mask) const;
    virtual std::string name() const;
};

struct CenterRotate : public ImageTransform
{
    CenterRotate(double angle);
    virtual void operator()(cv::Mat& m) const;
    virtual void operator()(cv::Mat& image, cv::Mat_<uchar>& mask) const;
    virtual std::string name() const;
    double angle;
};

} /* namespace image_transform */
} /* namespace vision */
#endif /* VISION__IMAGE_TRANSFORM_HPP_ */
