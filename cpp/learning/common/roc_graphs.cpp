/*
 * roc_graphs.cpp
 *
 *  Created on: May 24, 2013
 *      Author: Andrada Georgescu
 */

#include "roc_graphs.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace learning {
namespace common {

using std::vector;

static double trap_area(double x1, double x2, double y1, double y2) {
	const double base = fabs(x1 - x2);
	const double height = (y1 + y2) / 2;
	return base * height;
}

double area_under_curve(vector<std::pair<double, bool> >& scores,
												vector<std::pair<double, double> >* roc_points) {
	size_t pos_count = 0;
	// Sort probabilities.
	for (size_t i = 0; i < scores.size(); ++i) {
		if (scores[i].second) {
			++pos_count;
		}
	}
	sort(scores.begin(), scores.end(), std::greater<std::pair<double, bool> >());

	const size_t neg_count = scores.size() - pos_count;
	double fp = 0, tp = 0, fp_prev = 0, tp_prev = 0;
	double area = 0;
	double f_prev = std::numeric_limits<double>::min();

	for (size_t i = 0; i < scores.size(); ++i) {
		if (scores[i].first != f_prev) {
			// Update area.
			area += trap_area(fp, fp_prev, tp, tp_prev);

			// Add point to ROC graph.
			if (roc_points != NULL) {
				roc_points->push_back(std::make_pair(fp / neg_count, tp / pos_count));
			}

			// Update last point.
			f_prev = scores[i].first;
			fp_prev = fp;
			tp_prev = tp;
		}


		if (scores[i].second) {
			++tp;
		} else {
			++fp;
		}
	}
	// Add last area segment.
	area += trap_area(fp_prev, neg_count, tp_prev, pos_count);
	// Scale from PxN onto the unit space.
	area /= pos_count * neg_count;

	// Add last point to ROC graph.
	if (roc_points != NULL) {
		roc_points->push_back(std::make_pair(fp / neg_count, tp / pos_count));
	}

	return area;
}

double find_roc_point(
		std::vector<std::pair<double, bool> >& scores,
		double dmin, double* fpoint, double* dpoint) {
	size_t pos_count = 0;
	// Sort probabilities.
	for (size_t i = 0; i < scores.size(); ++i) {
		if (scores[i].second) {
			++pos_count;
		}
	}
	sort(scores.begin(), scores.end(), std::greater<std::pair<double, bool> >());

	const size_t neg_count = scores.size() - pos_count;
	double fp = 0, tp = 0, fp_prev = 0, tp_prev = 0;
	double f_prev = std::numeric_limits<double>::min();

	for (size_t i = 0; i < scores.size(); ++i) {
		if (scores[i].first != f_prev) {
			if (tp / pos_count >= dmin) {
				*fpoint = fp / neg_count;
				*dpoint = tp / pos_count;
				return scores[i].first;
			}

			// Update last point.
			f_prev = scores[i].first;
			fp_prev = fp;
			tp_prev = tp;
		}


		if (scores[i].second) {
			++tp;
		} else {
			++fp;
		}
	}

	*fpoint = fp / neg_count;
	*dpoint = tp / pos_count;
	return scores.back().first;
}

} // namespace common
} // namespace learning
