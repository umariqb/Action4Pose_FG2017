/** Armadillo based exact and very fast k nearest neighbor search
 *  \file:    armahelper.h
 *  \author   Dr. Christian Wengert
 *            christian.wengert@ieee.org
 *
 *
 */

#ifndef _ARMAKNN_H
#define	_ARMAKNN_H

#include <armadillo>

void knn_L2( const arma::fmat &queryVectors,
             const arma::fmat &referenceVectors,
               arma::fmat &distanceMap,
               arma::umat &indexMap,
                   const unsigned int k);


#endif	/* _ARMAKNN_H */

