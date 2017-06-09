/*
 * roc_graphs.hpp
 *
 *  Created on: May 24, 2013
 *      Author: Andrada Georgescu
 */

#ifndef LEARNING_COMMON_ROC_GRAPHS_
#define LEARNING_COMMON_ROC_GRAPHS_

#include <utility>
#include <vector>
#include <cstddef>


namespace learning {
namespace common {

double area_under_curve(
		std::vector<std::pair<double, bool> >& scores,
		std::vector<std::pair<double, double> >* roc_points = NULL);

// Finds ROC point where the hit-rate is at least dmin.
// Returns the other point on the curve fmin and the cooresponding score.
double find_roc_point(
		std::vector<std::pair<double, bool> >& scores,
		double dmin, double* fpoint, double* dpoint);

} // namespace common
} // namespace learning

#endif /* LEARNING_COMMON_ROC_GRAPHS_ */
