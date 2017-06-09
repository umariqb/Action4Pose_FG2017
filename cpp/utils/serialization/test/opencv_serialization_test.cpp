/*
 * opencv_serialization_test.cpp
 *
 *  Created on: Nov 4, 2011
 *      Author: lbossard
 */

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <opencv2/core/core.hpp>

#include "cpp/utils/serialization/opencv_serialization.hpp"

#include "cpp/third_party/gtest/gtest.h"


template <typename T, typename U>
void write_read(const T& in, U& out)
{
    std::stringstream buf;
    // serialize
    {
        boost::archive::binary_oarchive output_archive(buf);
        output_archive << in;
    }
    // deserialize
    {
        boost::archive::binary_iarchive input_archive(buf);
        input_archive >> out;
    }
}

TEST(OpenCvSerializationTest, TestRect)
{
    cv::Rect r(2,3,7,5);
    cv::Rect r2;
    write_read(r,r2);
    ASSERT_EQ(r, r2);
}

TEST(OpenCvSerializationTest, TestRectF)
{
    cv::Rect_<float> r(2.3, 2.9, 3.1, 3.7);
    cv::Rect_<float> r2;
    write_read(r,r2);
    ASSERT_EQ(r, r2);
}

TEST(OpenCvSerializationTest, TestMat)
{
    cv::Mat m = cv::Mat::eye(3,3,CV_32S);
    cv::Mat m2;
    write_read(m, m2);
    ASSERT_TRUE(
        std::equal(
                m.begin<int>(),
                m.end<int>(),
                m2.begin<int>()));
}

TEST(OpenCvSerializationTest, TestMat_ti)
{
    cv::Mat_<int> m = cv::Mat_<int>::eye(3,3);
    cv::Mat_<int> m2;
    write_read(m, m2);
    ASSERT_TRUE(
        std::equal(
                m.begin(),
                m.end(),
                m2.begin()));
}

TEST(OpenCvSerializationTest, TestMat_tf)
{
    cv::Mat_<float> m = cv::Mat_<float>::eye(3,3);
    cv::Mat_<float> m2;
    write_read(m, m2);
    ASSERT_TRUE(
        std::equal(
                m.begin(),
                m.end(),
                m2.begin()));
}

TEST(OpenCvSerializationTest, TestMat_non_matching)
{
    cv::Mat_<float> m = cv::Mat_<float>::eye(3,3);
    cv::Mat_<int> m2;
    write_read(m, m2);
    ASSERT_TRUE(m2.empty());
    ASSERT_TRUE(m2.data == NULL);
}

TEST(OpenCvSerializationTest, TestMat_partial)
{
    cv::Mat_<int> m = cv::Mat_<int>::eye(20,20);
    cv::Rect r(4,0,5,5);
    cv::Mat_<int> m_part = m(r);

    cv::Mat_<int> m2;
    write_read(m_part, m2);
    ASSERT_TRUE(
            std::equal(
                    m_part.begin(),
                    m_part.end(),
                    m2.begin()));
}
