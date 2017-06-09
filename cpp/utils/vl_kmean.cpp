/*
 * vl_kmean.cpp
 *
 *  Created on: Oct 23, 2013
 *      Author: mdantone
 */

#include "vl_kmean.hpp"
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

extern "C" {
#include "cpp/third_party/vlfeat/vl/kmeans.h"
}

namespace utils {
namespace kmean {


VlKMeans* kmean(const cv::Mat_<float>& features,
                int voc_size,
                int max_num_iterations) {

  CHECK_GT(voc_size, 1);

  VlKMeans* kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
  vl_kmeans_set_algorithm (kmeans, VlKMeansLloyd) ;

  // Initialize the cluster centers by randomly sampling the data
  vl_kmeans_init_centers_with_rand_data(kmeans, features[0], features.cols, features.rows, voc_size);
  // vl_kmeans_init_centers_plus_plus(kmeans, all_descriptors[0], all_descriptors.cols, all_descriptors.rows, vocab_size);


  // Run at most 5 iterations of cluster refinement using Lloyd algorithm
  vl_kmeans_set_max_num_iterations(kmeans, max_num_iterations) ;
  vl_kmeans_refine_centers(kmeans, features[0], features.rows) ;


  CHECK_EQ(vl_kmeans_get_dimension(kmeans), features.cols );
  CHECK_EQ(vl_kmeans_get_num_centers(kmeans), voc_size );

  return kmeans;
}


void copy_vocabulary(VlKMeans* kmeans,
                     cv::Mat_<float>& vocabulary) {

  int descriptor_dimension = vl_kmeans_get_dimension(kmeans);
  int voc_size= vl_kmeans_get_num_centers(kmeans);

  const float * centers = (float const *) vl_kmeans_get_centers(kmeans) ;
  vocabulary = cv::Mat_<float>(voc_size, descriptor_dimension );
  vocabulary.setTo(-1.0);
  for (int c = 0 ; c < voc_size; ++c) {
    for (int d = 0 ; d < descriptor_dimension ; ++d) {
      vocabulary.at<float>(c,d) = *centers;
      ++centers;
    }
  }
}

bool vl_kmean(const cv::Mat_<float>& features,
              int voc_size,
              cv::Mat_<float>& vocabulary,
              std::vector<uint32_t>& labels,
              std::vector<float>& distances,
              int max_num_iterations) {

  VlKMeans* kmeans = kmean(features, voc_size, max_num_iterations);

  labels.resize(features.rows);
  distances.resize(features.rows);
  vl_kmeans_quantize(kmeans, labels.data(), distances.data(),  features[0], features.rows) ;

  copy_vocabulary(kmeans, vocabulary);

  vl_kmeans_delete(kmeans);
  kmeans = NULL;
  return true;
}

bool vl_kmean(const cv::Mat_<float>& features,
              int voc_size,
              cv::Mat_<float>& vocabulary,
              int max_num_iterations) {

  VlKMeans* kmeans = kmean(features, voc_size, max_num_iterations);

  // copy vocabulary.
  copy_vocabulary(kmeans, vocabulary);

  vl_kmeans_delete(kmeans);
  kmeans = NULL;

  return true;
}

} /* kmean */
} /* namespace utils */
