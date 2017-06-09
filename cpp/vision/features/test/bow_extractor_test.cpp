/*
 * BowExtractorTest.cpp
 *
 *  Created on: Oct 14, 2011
 *      Author: lbossard
 */



#include "cpp/third_party/gtest/gtest.h"

#include "cpp/vision/features/bow_extractor.hpp"

TEST(BowTest, TestMatching)
{
    cv::Mat_<float> m =	cv::Mat_<float>::zeros(4, 2);
    m(0,0) = 10;
    m(1,1) = 10;
    m(2,0) = -10;
    m(3,1) = -10;

    cv::Mat_<float> f =	cv::Mat_<float>::zeros(5, 2);
    f(0,0) = 9;
    f(1,1) = 9;
    f(2,0) = -9;
    f(3,1) = -9;
    f(4,0) = -9;
    f(4,1) = -8;

    vision::features::BowExtractor d(m);
    std::vector<vision::features::BowExtractor::WordId> vw;
    d.match(f,vw);
    EXPECT_EQ(0, vw[0]);
    EXPECT_EQ(1, vw[1]);
    EXPECT_EQ(2, vw[2]);
    EXPECT_EQ(3, vw[3]);
    EXPECT_EQ(2, vw[4]);
}
TEST(BowTest, TestHistoCountComputation)
{
    ASSERT_EQ( 1,  vision::features::BowExtractor::getHistogramCount(1));
    ASSERT_EQ( 5,  vision::features::BowExtractor::getHistogramCount(2));
    ASSERT_EQ(21,  vision::features::BowExtractor::getHistogramCount(3));
    ASSERT_EQ(85,  vision::features::BowExtractor::getHistogramCount(4));
}


TEST(BowTest, TestPooling)
{
    std::vector<vision::features::BowExtractor::WordId> words;
    words.push_back(0);
    words.push_back(0);
    words.push_back(0);
    words.push_back(0);
    words.push_back(0);
    words.push_back(0);
    words.push_back(0);
    words.push_back(0);
    words.push_back(0);
    words.push_back(1);
    words.push_back(2);
    words.push_back(3);
    words.push_back(3);

    std::vector<cv::Point> locations;
    // test outside
    locations.push_back(cv::Point(2,0));
    locations.push_back(cv::Point(0,2));
    locations.push_back(cv::Point(13, 1));
    locations.push_back(cv::Point(1, 13));
    // test on boundaries (are considered to be inside)
    locations.push_back(cv::Point(2,1));
    locations.push_back(cv::Point(1,2));
    locations.push_back(cv::Point(12,9));
    locations.push_back(cv::Point(9,12));
    // test each bow
    locations.push_back(cv::Point(2,2));
    locations.push_back(cv::Point(3,2));
    locations.push_back(cv::Point(4,2));
    locations.push_back(cv::Point(5,2));
    // test multi occurrence
    locations.push_back(cv::Point(6,2));


    const cv::Rect rect(1,1,12,12);
    const unsigned int max_word_count = 4;
    int expected_histo[] = {5,1,1,2};

    cv::Mat_<int> histogram;
    vision::features::BowExtractor::createHistogram(words, locations, rect, cv::Mat_<uchar>(),max_word_count, histogram);
    for (unsigned int i = 0; i < max_word_count; ++i)
    {
        ASSERT_EQ(expected_histo[i], histogram(i)) << "histograms not equal at " << i << std::endl;
    }
}

TEST(BowTest, TestPoolingWithSpatialPyramid)
{
    std::vector<vision::features::BowExtractor::WordId> words;
    words.push_back(0);
    words.push_back(1);
    words.push_back(2);
    words.push_back(3);
    words.push_back(4);

    std::vector<cv::Point> locations;
    locations.push_back(cv::Point(0,1));
    locations.push_back(cv::Point(8, 1));
    locations.push_back(cv::Point(1, 8));
    locations.push_back(cv::Point(8, 8));
    locations.push_back(cv::Point(15, 15));

    const cv::Rect rect(0,0,16,16);
    const unsigned int max_word_count = 5;
    const unsigned int level = 3;
    const unsigned int histo_count = vision::features::BowExtractor::getHistogramCount(level);

    cv::Mat_<int> histogram;
    vision::features::BowExtractor::sumPool(words, locations, rect, cv::Mat_<uchar>(), max_word_count, level, histogram);

    int expected_histo[] = {
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 0, 1, 1,
            1, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 1 };

    histogram = histogram.reshape(0, max_word_count * histo_count);
    for (unsigned int i = 0; i < max_word_count * histo_count; ++i)
    {
        ASSERT_EQ(expected_histo[i], histogram(i)) << "histograms not equal at " << i << std::endl;
    }

}




