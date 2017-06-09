/*
 * bow_histogram_test.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: lbossard
 */




#include "cpp/third_party/gtest/gtest.h"

#include "cpp/vision/features/bow_histogram.hpp"



 vision::features::BowHistogram get_test_histo()
{
    cv::Mat_<int> histo(1,42);
    for (int i = 0; i < 21; ++i)
    {
        histo(2*i) = i;
        histo(2*i+1) = i;
    }
    return vision::features::BowHistogram(histo, vision::features::feature_type::None, 2);
}

TEST(BowHistogramTest, PartTest)
{
    vision::features::BowHistogram bh = get_test_histo();

    ASSERT_TRUE(bh.bow_count() == 2);
    ASSERT_TRUE(bh.spm_levels() == 3 );
    ASSERT_TRUE(bh.part_count() == 21);
    for (unsigned int i = 0; i < bh.part_count(); ++i)
    {
        ASSERT_TRUE(bh.histogram_part(i)(0) == i);
        ASSERT_TRUE(bh.histogram_part(i)(1) == i);
    }
}


TEST(BowHistogramTest, ViewTest)
{
     vision::features::BowHistogram bh = get_test_histo();

    ASSERT_TRUE(bh.bow_count() == 2);
    ASSERT_TRUE(bh.spm_levels() == 3 );
    ASSERT_TRUE(bh.part_count() == 21);
    int h_count = 0;
    for (unsigned int l = 0; l < bh.spm_levels(); ++l)
    {
        vision::features::BowHistogram::Histogram h = bh.histogram_view(l);
        std::cout << "h.rows == " <<  h.rows << std::endl;
        int expected = std::pow(std::pow(2, l), 2);
        ASSERT_TRUE( h.rows == expected);
        ASSERT_TRUE( h.cols == 2);
        for (unsigned int r = 0; r < h.rows; ++r)
        {
            std::cout << l << "/" <<  r << " " << h.row(r) << std::endl;
            ASSERT_TRUE(h.row(r)(0) == h_count);
            ASSERT_TRUE(h.row(r)(1) == h_count);
            ++h_count;
        }
    }
}
