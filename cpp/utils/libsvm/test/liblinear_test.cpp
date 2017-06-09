/*
 * liblinear_test.cpp
 *
 *  Created on: Feb 6, 2012
 *      Author: lbossard
 */

#include "cpp/utils/libsvm/liblinear.hpp"
namespace ll = utils::liblinear;

#include <sstream>

#include "cpp/third_party/gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>



std::string libsvm_document = "1 2:1.1123   3:.123\t4:1\n2\t  5:11\t6:.0003\n";
TEST(LibLinearUtilsTest, TestReadProblemFromStream)
{
    std::stringstream sstr(libsvm_document);
    utils::liblinear::ProblemHolder p;
    ASSERT_TRUE(p.read(sstr));

    ASSERT_EQ(2, p.labels().size());
    ASSERT_EQ(2, p.features().size());
    for (unsigned int i=0; i < p.labels().size(); ++i)
    {
        ASSERT_EQ(p.labels()[i], i+1);
    }
}

TEST(LibLinearUtilsTest, TestReadAlloc)
{
    std::stringstream sstr(libsvm_document);
    utils::liblinear::ProblemHolder p;
    p.allocate(1, 1);


    // testing the dyn allocation of the features and the labels
    cv::Mat_<int> feature1 = (cv::Mat_<int>(1,8) << 0,1,2,3,0,4,0,5);
    cv::Mat_<double> feature2 = (cv::Mat_<double>(1,8) << 6,0,0,7,8,9,0,10);
    cv::Mat_<double> feature3 = (cv::Mat_<double>(1,8) << 11,0,0,12,13,14,0,15);


    ASSERT_TRUE(p.push_problem<int>(feature1, 1));
    ASSERT_TRUE(p.push_problem<double>(feature2, 2));
    ASSERT_TRUE(p.push_problem<double>(feature3, 2));

    ASSERT_TRUE(p.features().size() >= 3);
    ASSERT_TRUE(p.feature_nodes().size() >= 18);
    ASSERT_TRUE(p.features()[0] == &p.feature_nodes()[0]);
    ASSERT_TRUE(p.features()[1] == &p.feature_nodes()[6]);
    ASSERT_TRUE(p.features()[2] == &p.feature_nodes()[12]);
    ASSERT_TRUE(p.feature_nodes()[0].value == 1);
    ASSERT_TRUE(p.feature_nodes()[1].value == 2);
    ASSERT_TRUE(p.feature_nodes()[2].value == 3);
    ASSERT_TRUE(p.feature_nodes()[3].value == 4);
    ASSERT_TRUE(p.feature_nodes()[4].value == 5);
    ASSERT_TRUE(p.feature_nodes()[5].index == -1);
    ASSERT_TRUE(p.feature_nodes()[6].value == 6);
    ASSERT_TRUE(p.feature_nodes()[7].value == 7);
    ASSERT_TRUE(p.feature_nodes()[8].value == 8);
    ASSERT_TRUE(p.feature_nodes()[9].value == 9);
    ASSERT_TRUE(p.feature_nodes()[10].value == 10);
    ASSERT_TRUE(p.feature_nodes()[11].index == -1);
    ASSERT_TRUE(p.feature_nodes()[12].value == 11);
    ASSERT_TRUE(p.feature_nodes()[13].value == 12);
    ASSERT_TRUE(p.feature_nodes()[14].value == 13);
    ASSERT_TRUE(p.feature_nodes()[15].value == 14);
    ASSERT_TRUE(p.feature_nodes()[16].value == 15);
    ASSERT_TRUE(p.feature_nodes()[17].index == -1);
}

TEST(LibLinearUtilsTest, TestRealloc)
{
    std::stringstream sstr(libsvm_document);
    utils::liblinear::ProblemHolder p;
    p.allocate(2, 1);

    cv::Mat_<int> feature1 = (cv::Mat_<int>(1,8) << 0,1,2,3,0,4,0,5);
    cv::Mat_<double> feature2 = (cv::Mat_<double>(1,8) << 6,0,0,7,8,9,0,10);

    ASSERT_TRUE(p.push_problem<int>(feature1, 1));
    ASSERT_TRUE(p.push_problem<double>(feature2, 2));

    ASSERT_TRUE(p.features().size() == 2);
    ASSERT_TRUE(p.feature_nodes().size() >= 12);
    ASSERT_TRUE(p.features()[0] == &p.feature_nodes()[0]);
    ASSERT_TRUE(p.features()[1] == &p.feature_nodes()[6]);
    ASSERT_TRUE(p.feature_nodes()[0].value == 1);
    ASSERT_TRUE(p.feature_nodes()[1].value == 2);
    ASSERT_TRUE(p.feature_nodes()[2].value == 3);
    ASSERT_TRUE(p.feature_nodes()[3].value == 4);
    ASSERT_TRUE(p.feature_nodes()[4].value == 5);
    ASSERT_TRUE(p.feature_nodes()[5].index == -1);
    ASSERT_TRUE(p.feature_nodes()[6].value == 6);
    ASSERT_TRUE(p.feature_nodes()[7].value == 7);
    ASSERT_TRUE(p.feature_nodes()[8].value == 8);
    ASSERT_TRUE(p.feature_nodes()[9].value == 9);
    ASSERT_TRUE(p.feature_nodes()[10].value == 10);
    ASSERT_TRUE(p.feature_nodes()[11].index == -1);

}

TEST(LibLinearUtilsTest, TestNoRealloc)
{
    std::stringstream sstr(libsvm_document);
    utils::liblinear::ProblemHolder p;
    p.allocate(2, 10);
    const std::size_t feature_count = p.feature_nodes().size();

    cv::Mat_<int> feature1 = (cv::Mat_<int>(1,8) << 0,1,2,3,0,4,0,5);
    cv::Mat_<double> feature2 = (cv::Mat_<double>(1,8) << 6,0,0,7,8,9,0,10);

    ASSERT_TRUE(p.push_problem<int>(feature1, 1));
    ASSERT_TRUE(p.push_problem<double>(feature2, 2));
    ASSERT_TRUE(p.feature_nodes().size() == feature_count);
}

std::string libsvm_linear_toy_problem =
        "1 1:1\n"
        "1 1:2\n"
        "1 1:3\n"
        "1 2:2\n"
        "1 2:3\n"
        "0 2:1\n"
        "0 3:1\n"
        "0 3:2\n"
        "0 4:2\n"
        "0 4:3\n";
TEST(LibLinearUtilsTest, TestTrain)
{
    std::stringstream sstr(libsvm_linear_toy_problem);
    utils::liblinear::ProblemHolder p;
    ASSERT_TRUE(p.read(sstr));

    utils::liblinear::LinearHolder m;
    m.train(p, utils::liblinear::solver_type::L2R_L2LOSS_SVC, 1);

    cv::Mat_<int> feature1 = (cv::Mat_<int>(1,4) << 0,3,0,0);
    ASSERT_TRUE( 1 == m.predict(feature1));
    cv::Mat_<int> feature2 = (cv::Mat_<int>(1,4) << 0,0,3,0);
    ASSERT_TRUE( 0 == m.predict(feature2));
}


cv::Mat_<double> libsvm_linear_toy_problem_x =
        (cv::Mat_<double>(10,4) <<
                1, 0, 0, 0,
                2, 0, 0, 0,
                3, 0, 0, 0,
                0, 2, 0, 0,
                0, 3, 0, 0,
                0, 0, 1, 0,
                0, 0, 2, 0,
                0, 0, 3, 0,
                0, 0, 0, 2,
                0, 0, 0, 3
                );
TEST(LibLinearUtilsTest, TestTrain2)
{
    utils::liblinear::ProblemHolder p;
    const int sample_count = libsvm_linear_toy_problem_x.rows;
    const int problem_size = 100;
    p.allocate(problem_size, 10);
    for (int i = 0; i < problem_size; ++i)
    {
        int idx = i % libsvm_linear_toy_problem_x.rows;
        ASSERT_TRUE(
            p.push_problem(
                libsvm_linear_toy_problem_x.row(idx),
                idx < sample_count/2));
    }


    utils::liblinear::LinearHolder m;
    double C = 1;
    m.train(p, (utils::liblinear::solver_type::T)2, C);

    cv::Mat_<int> feature1 = (cv::Mat_<int>(1,4) << 0,3,0,0);
    ASSERT_TRUE( 1 == m.predict(feature1));
    cv::Mat_<int> feature2 = (cv::Mat_<int>(1,4) << 0,0,3,0);
    ASSERT_TRUE( 0 == m.predict(feature2));
}

TEST(LibLinearUtilsTest, TestSerialize)
{
    std::stringstream buf;
    utils::liblinear::LinearHolder lh;

    // create model
    {
        utils::liblinear::ProblemHolder p;
        std::stringstream sstr(libsvm_linear_toy_problem);
        ASSERT_TRUE(p.read(sstr));
        lh.train(p, utils::liblinear::solver_type::L2R_L2LOSS_SVC, 1);
    }

    // serialize
    {
        boost::archive::binary_oarchive output_archive(buf);
        output_archive << lh;
    }

    // deserialize
    utils::liblinear::LinearHolder lh2;
    {

        boost::archive::binary_iarchive input_archive(buf);
        input_archive >> lh2;
    }
    const ll::LinearModel& m1 = lh.get_model();
    const ll::LinearModel& m2 = lh2.get_model();
    ASSERT_TRUE(m1.bias == m2.bias);
    ASSERT_TRUE(m1.nr_class == m2.nr_class);
    ASSERT_TRUE(m1.nr_feature == m2.nr_feature);
    ASSERT_TRUE(m1.param.solver_type == m2.param.solver_type);

    // check labels
    for (int i = 0; i <  m1.nr_class; ++i)
    {
        ASSERT_TRUE(m1.label[i] == m2.label[i]);
    }

    // check weighs (only two classes -> only one w vector)
    for (int i = 0; i <  m1.nr_feature; ++i)
    {
        ASSERT_TRUE(m1.w[i] == m2.w[i]);
    }
}
