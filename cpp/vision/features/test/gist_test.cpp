/*
 * gist_test.cpp
 *
 *  Created on: Sep 11, 2012
 *      Author: lbossard
 */

#include "cpp/third_party/gtest/gtest.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cpp/vision/features/global/gist.hpp"


const float delta = 0.000001f;

TEST(GistTest, TestGistColor)
{
    cv::Mat image = cv::imread("src/cpp/third_party/lear_gist/ar.ppm");
    ASSERT_TRUE(image.data != NULL);

    cv::Mat_<float> descriptors;
    vision::features::global::Gist::extractColor(image, descriptors, 4, 8, 8, 4);

    // values taken from the README of the lear_gist package
    ASSERT_EQ(1, descriptors.rows);
    ASSERT_EQ(960, descriptors.cols);

    ASSERT_NEAR(0.057863228f, descriptors(0), delta); // 0.0579
    ASSERT_NEAR(0.19255628f, descriptors(1), delta);  // 0.1926
    ASSERT_NEAR(0.093314603f, descriptors(2), delta); // 0.0933
    ASSERT_NEAR(0.066224582f, descriptors(3), delta); // 0.0662

    ASSERT_NEAR(0.056342714f, descriptors(957), delta); // 0.0563
    ASSERT_NEAR(0.057511147f, descriptors(958), delta); // 0.0575
    ASSERT_NEAR(0.06395784f, descriptors(959), delta);  // 0.0640
}


TEST(GistTest, TestGistBw)
{
    cv::Mat image = cv::imread("src/cpp/third_party/lear_gist/ar.ppm");
    cv::cvtColor(image,  image, CV_BGR2GRAY);

    cv::Mat_<float> descriptors;
    vision::features::global::Gist::extract(image, descriptors, 4, 8, 8, 4);

    ASSERT_EQ(1, descriptors.rows);
    ASSERT_EQ(320, descriptors.cols);

    ASSERT_NEAR(0.0515042f, descriptors(0), delta);
    ASSERT_NEAR(0.1560174f, descriptors(1), delta);
    ASSERT_NEAR(0.0628004f, descriptors(2), delta);
    ASSERT_NEAR(0.0548037f, descriptors(3), delta);

    ASSERT_NEAR(0.0782973f, descriptors(317), delta);
    ASSERT_NEAR(0.0641783f, descriptors(318), delta);
    ASSERT_NEAR(0.0665550f, descriptors(319), delta);
}

TEST(GistTest, TestGistColorNoncontinous)
{
    cv::Mat image = cv::imread("src/cpp/third_party/lear_gist/ar.ppm");
    cv::Mat image_bb(image(cv::Rect(2,3,22,22)));
    ASSERT_FALSE(image_bb.isContinuous());

    cv::Mat_<float> descriptors;
    vision::features::global::Gist::extractColor(image_bb, descriptors, 4, 8, 8, 4);

    // values taken from the README of the lear_gist package
    ASSERT_EQ(1, descriptors.rows);
    ASSERT_EQ(960, descriptors.cols);

    ASSERT_NEAR(0.0892997f, descriptors(0), delta);
    ASSERT_NEAR(0.19574021f, descriptors(1), delta);
    ASSERT_NEAR(0.17091835f, descriptors(2), delta);
    ASSERT_NEAR(0.1067882f, descriptors(3), delta);

    ASSERT_NEAR(0.077924281, descriptors(957), delta);
    ASSERT_NEAR(0.021825379f, descriptors(958), delta);
    ASSERT_NEAR(0.046658281f, descriptors(959), delta);
}

TEST(GistTest, TestGistBwNoncontinous)
{
    cv::Mat image = cv::imread("src/cpp/third_party/lear_gist/ar.ppm");
    cv::cvtColor(image,  image, CV_BGR2GRAY);
    cv::Mat image_bb(image(cv::Rect(2,3,22,22)));
    ASSERT_FALSE(image_bb.isContinuous());

    cv::Mat_<float> descriptors;
    vision::features::global::Gist::extract(image_bb, descriptors, 4, 8, 8, 4);

    ASSERT_EQ(1, descriptors.rows);
    ASSERT_EQ(320, descriptors.cols);

    ASSERT_NEAR(0.0740971863f, descriptors(0), delta);
    ASSERT_NEAR(0.163565114f, descriptors(1), delta);
    ASSERT_NEAR(0.140128613f, descriptors(2), delta);
    ASSERT_NEAR(0.0764200464f, descriptors(3), delta);

    ASSERT_NEAR(0.0488275997f, descriptors(317), delta);
    ASSERT_NEAR(0.0227536801f, descriptors(318), delta);
    ASSERT_NEAR(0.0377165712f, descriptors(319), delta);
}


TEST(GistTest, TestGistColorTooSmall)
{
    cv::Mat image = cv::imread("src/cpp/third_party/lear_gist/ar.ppm");
    cv::Mat image_bb(image(cv::Rect(0,0,7,7)));
    ASSERT_FALSE(image_bb.isContinuous());

    cv::Mat_<float> descriptors;
    vision::features::global::Gist::extractColor(image_bb, descriptors, 4, 8, 8, 4);

    // values taken from the README of the lear_gist package
    ASSERT_EQ(0, descriptors.rows);
    ASSERT_EQ(0, descriptors.cols);
}

TEST(GistTest, TestGistBwTooSmall)
{
    cv::Mat image = cv::imread("src/cpp/third_party/lear_gist/ar.ppm");
    cv::cvtColor(image,  image, CV_BGR2GRAY);
    cv::Mat image_bb(image(cv::Rect(0,0,7,7)));
    ASSERT_FALSE(image_bb.isContinuous());

    cv::Mat_<float> descriptors;
    vision::features::global::Gist::extract(image_bb, descriptors, 4, 8, 8, 4);

    ASSERT_EQ(0, descriptors.rows);
    ASSERT_EQ(0, descriptors.cols);

}
