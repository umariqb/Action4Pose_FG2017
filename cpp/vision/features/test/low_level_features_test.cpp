/*
 * FeaturesTest.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: lbossard
 */



#include "cpp/third_party/gtest/gtest.h"


#include <opencv2/highgui/highgui.hpp>
#include "cpp/vision/features/low_level_features.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"

const cv::Mat colors_image = cv::imread("src/cpp/vision/features/test/colors.bmp");
const cv::Mat colors_big_image = cv::imread("src/cpp/vision/features/test/colors_big.bmp");
const cv::Mat blocks_image = cv::imread("src/cpp/vision/features/test/blocks.bmp");
const cv::Mat hog_patch = cv::imread("src/cpp/vision/features/test/patch.bmp");

float _expected_surfs_mem_colors_big[] = {
    0, 0, 0, 0, 0, 0, 0, 0, -0.073242232, 0, 0.073242232, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.11385763, -0.11385763, 0.14524946, 0.14524946, 0.15595104, 0.29861718, 0.38751754, 0.45846498, 0, 0.069398969, 0, 0.069398969, -0.014490715, -0.014644572, 0.018780539, 0.017128153, 0.22323816, 0.22260392, 0.41128892, 0.4045555, 0.031391837, 0.031391837, 0.031391837, 0.031391837, 0, 0, 0, 0, 0.0081416192, 0.0082417605, 0.011933585, 0.011729317, 0.0025400266, 0.0025400266, 0.0025400266, 0.0025400266, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0080258893, -0.0076631121, 0.011019788, 0.010421714, 0.011672395, -0.012164967, 0.011672395, 0.012164967, 0, 0, 0, 0, 0, 0, 0, 0, -0.003973858, 0.0036783144, 0.003973858, 0.0036783144, 0.28986162, -0.28873125, 0.38386437, 0.37878621, 0.11547251, -0.11547251, 0.11547251, 0.11547251, 0, 0, 0, 0, 0, 0, 0, 0, -0.032182802, 0.032182802, 0.032182802, 0.032182802, 0.18419124, -0.36014816, 0.34807971, 0.42742565, 0, -0.080704078, 0, 0.080704078, 0, 0, 0, 0, 0, 0, 0, 0, -0.037504423, 0, 0.037504423, 0, 0, 0, 0, 0,
    0.020953741, 0, 0.020953741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.14506987, 0, 0.14506987, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3695738, -0.31437728, 0.3795397, 0.32623523, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.043091729, -0.043091729, 0.067883171, 0.067883171, 0.33347413, -0.3324298, 0.34533206, 0.34239572, 0, 0, 0, 0, 0, 0, 0, 0
};
const int feature_indexes[] = {0, 23, 43};
const cv::Mat_<float> expected_surfs_colors_big(3, 64, _expected_surfs_mem_colors_big);

TEST(FeatureTest, TestSurf)
{
    vision::features::Surf extractor;
    std::vector<cv::Point> locations;
    cv::Mat_<float> descriptors = extractor.denseExtract(colors_big_image, locations);
    ASSERT_EQ(576, locations.size());
    ASSERT_EQ(locations.size(), descriptors.rows);
    ASSERT_EQ(descriptors.cols, expected_surfs_colors_big.cols);
    for (unsigned int r_idx = 0; r_idx < 3; ++r_idx)
    {
      int r = feature_indexes[r_idx];
      for (unsigned int c = 0; c < expected_surfs_colors_big.cols; ++c)
      {
        ASSERT_NEAR(expected_surfs_colors_big(r_idx,c), descriptors(r,c), 0.0000001) <<  "histograms not equal at " << r << "/" << c << std::endl;
      }
    }
}

//const cv::Mat nan_img = cv::imread("/home/lbossard/g/mit-indoor/train_images/cloister/image.jpg");
TEST(FeatureTest, TestRootSift)
{
    vision::features::RootSift extractor;
    std::vector<cv::Point> locations;
    cv::Mat_<float> descriptors = extractor.denseExtract(colors_big_image, locations);
//    cv::Mat_<float> descriptors = extractor.denseExtract(nan_img, locations);
    ASSERT_EQ(0, cv::countNonZero(descriptors != descriptors));
    ASSERT_EQ(400, locations.size());
    ASSERT_EQ(locations.size(), descriptors.rows);
    ASSERT_EQ(descriptors.cols, 128);

}

TEST(FeatureTest, TestSurfMultiScale)
{
    vision::features::Surf extractor;
    std::vector<cv::Point> locations;
    cv::Mat_<float> descriptors = extractor.denseExtractMultiscale(colors_big_image, locations, -1, 3, .5);
    return;
    //std::cout << descriptors << std::endl;
    ASSERT_EQ(4612, locations.size());
    ASSERT_EQ(locations.size(), descriptors.rows);
    ASSERT_EQ(descriptors.cols, expected_surfs_colors_big.cols);

}

TEST(FeatureTest, TestRootSurf)
{
  vision::features::RootSurf extractor;
  std::vector<cv::Point> locations;
  cv::Mat_<float> descriptors = extractor.denseExtract(colors_big_image, locations);
  ASSERT_EQ(576, locations.size());
  ASSERT_EQ(locations.size(), descriptors.rows);
  ASSERT_EQ(descriptors.cols, expected_surfs_colors_big.cols);

  // root normalization
  cv::Mat_<float> normalized_surf;
  normalized_surf.create(expected_surfs_colors_big.rows, expected_surfs_colors_big.cols);
  for (int r = 0; r < expected_surfs_colors_big.rows; ++r){
    for (int c = 0; c < expected_surfs_colors_big.cols; ++c){
      if (expected_surfs_colors_big(r,c) < 0){
        normalized_surf(r,c) = -std::sqrt(-expected_surfs_colors_big(r,c));
      }
      else {
        normalized_surf(r,c) = std::sqrt(expected_surfs_colors_big(r,c));
      }
    }
    const double l2_norm = cv::norm(normalized_surf.row(r), cv::NORM_L2);
    if (l2_norm == 0.){
      continue;
    }
    normalized_surf.row(r) /= l2_norm;
  }

  for (unsigned int r_idx = 0; r_idx < 3; ++r_idx){
    int r = feature_indexes[r_idx];
    std::cout << "l1=" << cv::norm(descriptors.row(r), cv::NORM_L1) << ", l2=" << cv::norm(descriptors.row(r), cv::NORM_L2) << std::endl;
    for (unsigned int c = 0; c < expected_surfs_colors_big.cols; ++c){
      ASSERT_FALSE(std::isnan(descriptors(r,c)));
      ASSERT_NEAR(normalized_surf(r_idx,c), descriptors(r,c), 0.0000001) << "rootSurf not equal at " << r << "/" << c << std::endl;
    }
  }
}

TEST(FeatureTest, TestColor)
{
    std::vector<cv::Point> locations;
    vision::features::Color extractor;
    cv::Mat_<float> descriptors = extractor.denseExtract(colors_image, locations);
    ASSERT_EQ(3, descriptors.cols);
    ASSERT_EQ(100, descriptors.rows);
}

TEST(FeatureTest, GridExtraction)
{
    std::vector<cv::Point> locations;

    cv::Size image(16,16);
    cv::Size desc_size(16,16);
    cv::Mat_<uchar> mask =  cv::Mat_<uchar>(image);
    mask.setTo(1);
    vision::features::LowLevelFeatureExtractor::generateGridLocations(
        image,
        cv::Size(5,5),
        desc_size.height/2,
        desc_size.width/2,
        locations, mask);

    ASSERT_EQ(1, locations.size());
    ASSERT_EQ(desc_size.width/2, locations[0].x);
    ASSERT_EQ(desc_size.height/2, locations[0].y);


    // check mask
    mask.setTo(0);

    locations.clear();
    vision::features::LowLevelFeatureExtractor::generateGridLocations(
        image,
        cv::Size(5,5),
        desc_size.height/2,
        desc_size.width/2,
        locations, mask);
    ASSERT_EQ(0, locations.size());


    mask.setTo(1);
    locations.clear();
    image = cv::Size(16+5,16+5);
    vision::features::LowLevelFeatureExtractor::generateGridLocations(
        image,
        cv::Size(5,5),
        desc_size.height/2,
        desc_size.width/2,
        locations, mask);
    ASSERT_EQ(4, locations.size());

}

TEST(FeatureTest, TestColorExtract)
{
    vision::features::Color extractor;
    cv::Mat_<char> mask = cv::Mat_<char>::ones(colors_image.size());

    std::vector<cv::Point> locations1;
    cv::Mat_<float> features1 = extractor.denseExtract(colors_image, locations1, false);

    std::vector<cv::Point> locations2;
    cv::Mat_<float> features2 = extractor.denseExtractMasked(colors_image, locations2, mask);

//    std::cout<<features1 << std::endl;
//    std::cout<<features2<<std::endl;

    ASSERT_EQ(locations1, locations2);
    ASSERT_EQ(features1.rows, features2.rows);
    ASSERT_EQ(features1.cols, features2.cols);
    for (unsigned int c = 0; c < features1.cols; ++c)
    {
        for (unsigned int r = 0; r < features1.rows; ++r)
        {
            ASSERT_EQ(features1(r,c), features2(r,c)) << "rootsurf not equal at " << r << "/" << c << std::endl;
        }
    }
}


TEST(FeatureTest, TestColorExtractMasked)
{
    vision::features::Color extractor;
    cv::Mat_<char> mask = cv::Mat_<char>::zeros(colors_image.size());
    cv::Rect roi(0,1,2,3);
    mask(roi) = 1;


    std::vector<cv::Point> locations1;
    cv::Mat_<float> features1 = extractor.denseExtract(colors_image(roi), locations1, false);

    std::vector<cv::Point> locations2;
    cv::Mat_<float> features2 = extractor.denseExtractMasked(colors_image, locations2, mask);

//    std::cout<<features1 << std::endl;
//    std::cout<<features2<<std::endl;

    ASSERT_EQ(locations1.size(), locations2.size());
    ASSERT_EQ(features1.rows, features2.rows);
    ASSERT_EQ(features1.cols, features2.cols);
    for (unsigned int c = 0; c < features1.cols; ++c)
    {
        for (unsigned int r = 0; r < features1.rows; ++r)
        {
            ASSERT_EQ(features1(r,c), features2(r,c)) << "histograms not equal at " << r << "/" << c << std::endl;
        }
    }
}


TEST(FeatureTest, TestLbp)
{
    float expected_features[10] = {
            255, 0, 228, 0, 78,
            0 , 0 , 250, 57, 190
    };

    vision::features::Lbp extractor;
    std::vector<cv::Point> locations;
    cv::Mat_<float> features = extractor.denseExtract(blocks_image, locations);
//    std::cout << features << std::endl;

    ASSERT_EQ(1, features.cols);
    ASSERT_EQ(10, features.rows);
    for (int i = 0; i < 10; ++i)
    {
        ASSERT_EQ(expected_features[i], features(i));
    }

}

TEST(FeatureTest, TestLbpHisto)
{
    vision::features::Lbp extractor;
    cv::Mat_<int> histo;
    extractor.denseExtractHistogram(blocks_image, histo);

    for (int i = 0; i < 255; ++i)
    {
        if (i == 0)
        {
            ASSERT_EQ(4, histo(i));
        }
        else if (i == 255 || i == 228 || i == 78 || i == 250 || i == 57 || i == 190)
        {
            ASSERT_EQ(1, histo(i));
        }
        else
        {
            ASSERT_EQ(0, histo(i));
        }
    }
}
TEST(FeatureTest, TestSSDBinMap)
{
    const int expected_bin_map[] =
     {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, 15, 15, 19, 19, 23, 23, 23, 27, 27, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, 11, 15, 15, 15, 19, 19, 23, 23, 23, 27, 27, 27, 31, -1, -1, -1, -1,
      -1, -1, -1, 11, 11, 15, 15, 15, 19, 19, 23, 23, 23, 27, 27, 27, 31, 31, -1, -1, -1,
      -1, -1, 11, 11, 11, 11, 15, 15, 15, 19, 23, 23, 27, 27, 27, 31, 31, 31, 31, -1, -1,
      -1, -1,  7,  7, 11, 11, 11, 14, 14, 18, 22, 22, 26, 26, 31, 31, 31, 35, 35, -1, -1,
      -1,  7,  7,  7,  7, 11, 10, 10, 14, 18, 22, 22, 26, 30, 30, 31, 35, 35, 35, 35, -1,
      -1,  7,  7,  7,  7,  6, 10, 10, 14, 13, 21, 25, 26, 30, 30, 34, 35, 35, 35, 35, -1,
      -1,  3,  3,  3,  7,  6,  6,  6,  9, 13, 21, 25, 29, 34, 34, 34, 35, 39, 39, 39, -1,
      -1,  3,  3,  3,  3,  2,  2,  5,  5,  8, 20, 28, 33, 33, 38, 38, 39, 39, 39, 39, -1,
       3,  3,  3,  3,  3,  2,  2,  1,  1,  0, -1, 40, 41, 41, 42, 42, 43, 43, 43, 43, 43,
      -1, 79, 79, 79, 79, 78, 78, 73, 73, 68, 60, 48, 45, 45, 42, 42, 43, 43, 43, 43, -1,
      -1, 79, 79, 79, 75, 74, 74, 74, 69, 65, 61, 53, 49, 46, 46, 46, 47, 43, 43, 43, -1,
      -1, 75, 75, 75, 75, 74, 70, 70, 66, 65, 61, 53, 54, 50, 50, 46, 47, 47, 47, 47, -1,
      -1, 75, 75, 75, 75, 71, 70, 70, 66, 62, 62, 58, 54, 50, 50, 51, 47, 47, 47, 47, -1,
      -1, -1, 75, 75, 71, 71, 71, 66, 66, 62, 62, 58, 54, 54, 51, 51, 51, 47, 47, -1, -1,
      -1, -1, 71, 71, 71, 71, 67, 67, 67, 63, 63, 59, 55, 55, 55, 51, 51, 51, 51, -1, -1,
      -1, -1, -1, 71, 71, 67, 67, 67, 63, 63, 63, 59, 59, 55, 55, 55, 51, 51, -1, -1, -1,
      -1, -1, -1, -1, 71, 67, 67, 67, 63, 63, 63, 59, 59, 55, 55, 55, 51, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, 67, 67, 63, 63, 63, 59, 59, 55, 55, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 63, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    vision::features::SelfSimilarity extractor(
                3, //patch_size
                10, // window_radius
                4, // radius_bin_count
                20, // angle_bin_count
                1.	// var_noise
                );
    cv::Mat_<int> bin_map = extractor.getBinMap().reshape(0, 21*21);
    for (unsigned int i = 0; i < 21*21; ++i)
    {
        ASSERT_EQ(expected_bin_map[i], bin_map(i)) << "bin_map not equal at " << i << std::endl;
    }
}


TEST(FeatureTest, TestSSD)
{
    float expected_descriptor[] =
            { 1, 0.85740393, 0.85740393, 0.73514146, 0.65502822, 0.56162381, 0.92596108,
            0.73514146, 0.73514146, 0.92596108, 0.73514146, 0.73514146,
            0.73514146, 0.65502822, 0.56162381, 1, 0.85740393, 0.85740393 };
//    { 0.99999988, 0.80724025, 0.80724025, 0.64196724, 0.53367114, 0.40740806,
//            0.89991498, 0.64196724, 0.64196724, 0.89991498, 0.64196724,
//            0.64196724, 0.64196724, 0.53367114, 0.40740806, 0.99999988,
//            0.80724025, 0.80724025 };

    vision::features::SelfSimilarity extractor(
             5,  //patch_size
             20,  // window_radius
             3,  // radius_bin_count
             6,  // angle_bin_count
             25*3*36, // var_noise
             1   // auto noise radius
            );
    cv::Mat_<float> descriptor;
    extractor.extract(colors_big_image, cv::Point(50,50), descriptor);
    std::cout << descriptor << std::endl;
    ASSERT_EQ(1, descriptor.rows);
    ASSERT_EQ(3*6, descriptor.cols);
    for (unsigned int i = 0; i < 18; ++i)
    {
        ASSERT_NEAR(expected_descriptor[i], descriptor(i), .0000000001);
    }
}

TEST(FeatureTest, FelzenHogDim){
  const uint32_t width = 23;
  const uint32_t height = 32;
  uint32_t cell_size = 8;
  const uint32_t num_orientations = 9;

  uint32_t hog_dims = 3 * num_orientations  + 4; //Felzenhog
  uint32_t dims = vision::features::FelzenHogExtracor(num_orientations, cell_size).descriptor_length(width, height);


  uint32_t hogWidth = (width + cell_size/2) / cell_size;
  uint32_t hogHeight = (height + cell_size/2) / cell_size;
  uint32_t dims2 = hogHeight * hogWidth * hog_dims;
  ASSERT_EQ(dims, dims2);


}

TEST(FeatureTest, FelzenHog){
  static const float _test[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0002897, 0.0141328, 0.0007033, 0, 0, 0, 0, 0, 0.0023338, 0.0096986, 0, 0, 0, 0.0301018, 0.2735339, 0.0034476, 0, 0, 0, 0, 0, 0.0284162, 0.3469166, 0, 0, 0, 0.0892251, 0.3871273, 0, 0, 0, 0, 0, 0, 0.0286927, 0.3953008, 0, 0, 0, 0.0960848, 0.4000000, 0, 0, 0, 0, 0, 0, 0.0371489, 0.4000000, 0, 0, 0, 0.0962340, 0.4000000, 0, 0, 0.0007822, 0.0116524, 0, 0, 0.0685323, 0.4000000, 0, 0, 0, 0.0962340, 0.4000000, 0, 0, 0.0033663, 0.0446936, 0, 0, 0.0847847, 0.4000000, 0, 0, 0, 0.0962332, 0.4000000, 0, 0, 0, 0, 0, 0, 0.0565744, 0.4000000, 0, 0, 0, 0.0964564, 0.4000000, 0, 0, 0, 0, 0, 0, 0.0333950, 0.4000000, 0, 0, 0, 0.0750069, 0.3955509, 0.0012449, 0, 0, 0, 0, 0, 0.0329464, 0.4000000, 0, 0, 0, 0.0126740, 0.1876385, 0.0038188, 0, 0, 0, 0, 0, 0.0231542, 0.2264243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  static const cv::Mat_<float> test(1,sizeof(test)/sizeof(float), const_cast<float*>(_test));


  vision::features::FelzenHogExtracor extractor;
  cv::Mat_<float> descriptors;
  extractor.extract(colors_big_image, &descriptors);
  ASSERT_EQ(13*13*31, descriptors.cols);

  for (int c = 0; c < test.cols; ++c){
    ASSERT_NEAR(test(c), descriptors(c), 0.0000001);
  }

}

TEST(FeatureTest, FelzenHog2){
  cv::Mat_<float> test;
  utils::serialization::read_binary_archive("src/cpp/vision/features/test/patch.fhog", test);

  vision::features::FelzenHogExtracor extractor;
  cv::Mat_<float> descriptors;
  extractor.extract(hog_patch, &descriptors);
  ASSERT_EQ(8*8*31, descriptors.cols);

  for (int c = 0; c < test.cols; ++c){
    ASSERT_NEAR(test(c), descriptors(c), 0.0000001) << c;
  }

}

TEST(FeatureTest, FelzenHogVis){
  cv::Mat_<uchar> test = cv::imread("src/cpp/vision/features/test/patch_hog.pgm", cv::IMREAD_GRAYSCALE);
//  cv::imshow("foo", test);
//  cv::waitKey();

  vision::features::FelzenHogExtracor extractor;
  cv::Mat_<float> descriptors;
  extractor.extract(hog_patch, &descriptors);
  cv::Mat_<uchar> vis;
  extractor.visualize(descriptors, &vis);
  ASSERT_EQ(168, vis.rows);
  ASSERT_EQ(168, vis.cols);
  std::cout << vis.row(0) << std::endl;
  std::cout << test.row(0) << std::endl;
//    cv::imshow("test", test);
//    cv::imshow("foo", vis);
//    cv::waitKey();
  // vlfeat does the normalization wrong and thus produces a float image with
  // values bigger 1. we normalize this to 0...255, thus have discrepancies
  // for some pixels
  ASSERT_EQ(cv::countNonZero( (test - vis) != 0), 9789);

}


