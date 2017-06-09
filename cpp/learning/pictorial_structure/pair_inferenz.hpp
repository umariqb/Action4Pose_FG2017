/*
 * pair_inferenz.hpp
 *
 *  Created on: Apr 2, 2013
 *      Author: mdantone
 */

#ifndef PAIR_INFERENZ_HPP_
#define PAIR_INFERENZ_HPP_


#include "cpp/learning/pictorial_structure/model.hpp"

namespace learning
{
namespace ps
{

class PairInferenz {
public:

  PairInferenz(Model* m) : model(m), _inferenz_score(0)  {
    int n_parts = model->get_num_parts();
    scores.resize(n_parts);
    score_src_x.resize(n_parts);
    score_src_y.resize(n_parts);

  };

  float compute_min(int part_id = 0);

  void compute_arg_min( std::vector<cv::Point_<int> >& min_locations,
      std::vector<int>& min_orientations,
      int part_id ) const;

  void compute_arg_min( std::vector<cv::Point_<int> >& min_locations) const;

  float get_score() const {
    return _inferenz_score;
  }

private:
  Model* model;
  float _inferenz_score;
  std::vector<int> best_orientation;
  std::vector<std::vector<cv::Mat_<float> > > scores;
  std::vector<std::vector<cv::Mat_<int> > > score_src_x;
  std::vector<std::vector<cv::Mat_<int> > > score_src_y;

};


float pair_inferenz_multiple(std::vector<Model> models,
     const std::vector<cv::Mat_<float> >& apperance_scores,
     std::vector<cv::Point_<int> >& min_locations, bool debug = false,
     const cv::Mat& img = cv::Mat());

} /* namespace ps */
} /* namespace learning */

#endif /* PAIR_INFERENZ_HPP_ */
