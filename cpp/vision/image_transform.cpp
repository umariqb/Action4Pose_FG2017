/*
 * image_transforms.cpp
 *
 *  Created on: Sep 12, 2012
 *      Author: lbossard
 */

#include "image_transform.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>

namespace vision
{
namespace image_transform
{
////////////////////////////////////////////////////////////////////////////////
// Flip
/*virtual*/ void Flipper::operator()(cv::Mat& m) const
{
	const int flipcode = 1;
	cv::Mat dst;
	cv::flip(m, dst, flipcode);
	m = dst.clone();
}

/*virtual*/ void Flipper::operator()(cv::Mat& image, cv::Mat_<uchar>& mask) const
{
	(*this)(image);
	(*this)(mask);
}

/*virtual*/ std::string Flipper::name() const
{
	return "_flip";
}

////////////////////////////////////////////////////////////////////////////////
// Rotate
CenterRotate::CenterRotate(double angle) :
		angle(angle)
{
}

/*virtual*/void CenterRotate::operator()(cv::Mat& m) const
{
	// we dont wanna lose something -> comput bounding box after rotation
	double radiants = angle / 180. * boost::math::constants::pi<double>();
	double sin_r = std::fabs(std::sin(radiants));
	double cos_r = std::fabs(std::cos(radiants));
	cv::Mat dst(int(m.cols * sin_r + m.rows * cos_r),
			int(m.rows * sin_r + m.cols * cos_r), 0);

	cv::Point center = cv::Point(m.cols / 2, m.rows / 2);
	cv::Mat_<double> rot_mat = cv::getRotationMatrix2D(center, angle, 1.);
	rot_mat(1, 2) += (dst.rows - m.rows) / 2;
	rot_mat(0, 2) += (dst.cols - m.cols) / 2;
	cv::warpAffine(m, dst, rot_mat, dst.size());
	m = dst.clone();
}

/*virtual*/void CenterRotate::operator()(cv::Mat& image, cv::Mat_<uchar>& mask) const
{
	if (!mask.data)
	{
		mask.create(image.rows, image.cols);
		mask = 255;
	}
	(*this)(image);
	(*this)(mask);
}

/*virtual*/std::string CenterRotate::name() const
{
	return "_r" + boost::lexical_cast<std::string>(angle);
}

} /* namespace image_transform */
} /* namespace vision */
