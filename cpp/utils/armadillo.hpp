/*
 * armadillo.hpp
 *
 *  Created on: Sep 6, 2013
 *      Author: lbossard
 */

#ifndef UTILS__ARMADILLO_HPP_
#define UTILS__ARMADILLO_HPP_

#include <armadillo>
#include "cpp/utils/armadillo.hpp"

#include <opencv2/core/core.hpp>

namespace utils {
namespace armadillo {

template <typename U, typename V>
void to_opencv(
    const arma::Mat<U>& arma_mat,
    cv::Mat_<V>* cv_mat){

  const cv::Mat_<U> arma_wrapper(
      arma_mat.n_cols,
      arma_mat.n_rows,
      const_cast<U*>(arma_mat.memptr()));

  // check, if types match
  typedef typename arma::Mat<U>::elem_type ArmaElementType;
  if (cv::DataType<ArmaElementType>::type == cv_mat->type()){
    cv::transpose(arma_wrapper, *cv_mat);
  }
  else {
    arma_wrapper.convertTo(*cv_mat, cv_mat->type());
    cv::transpose(*cv_mat, *cv_mat);
  }
}


template <typename U>
void from_opencv(
    const cv::Mat_<U>& cv_mat,
    arma::Mat<U>* arma_mat
    ){
  const arma::Mat<U> cv_wrapper(
      const_cast<U*>(cv_mat[0]),
      cv_mat.cols,
      cv_mat.rows,
      /*copy_aux_mem*/ false, /*strict*/ true);

   *arma_mat = cv_wrapper.t();
}


template <typename U, typename V>
void from_opencv(
    const cv::Mat_<U>& cv_mat,
    arma::Mat<V>* arma_mat
    ){
  const arma::Mat<U> cv_wrapper(
      const_cast<U*>(cv_mat[0]),
      cv_mat.cols,
      cv_mat.rows,
      /*copy_aux_mem*/ false, /*strict*/ true);
  typedef arma::Mat<V> ArmaMatType;
  *arma_mat = arma::conv_to<ArmaMatType>::from(cv_wrapper.t());
}


}
}
#endif /* UTILS__ARMADILLO_HPP_ */
