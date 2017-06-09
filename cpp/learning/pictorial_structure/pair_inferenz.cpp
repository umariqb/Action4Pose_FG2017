/*
 * inferenz.cpp
 *
 *  Created on: Mar 24, 2013
 *      Author: mdantone
 */
#include "cpp/learning/pictorial_structure/pair_inferenz.hpp"
#include "cpp/learning/pictorial_structure/utils.hpp"
#include "cpp/learning/pictorial_structure/math_util.hpp"

#include "cpp/utils/thread_pool.hpp"
#include "cpp/utils/system_utils.hpp"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace learning
{
namespace ps
{

  float PairInferenz::compute_min(int part_id) {
    CHECK_GT(scores.size(), part_id);
    const Part* part = model->get_part(part_id);
    vector<int> child_ids = model->get_children_ids(part_id);
    bool is_root = !model->has_parent(part_id);
    bool is_leaf = (child_ids.size() == 0);

    // recursive call.
    for( int i = 0; i < child_ids.size(); i++){
      compute_min(child_ids[i]);
    }

    if(!is_root) {

      // allocate storage.
      int n_orientations = part->num_orientations();
      scores[part_id].resize(n_orientations);
      score_src_x[part_id].resize(n_orientations);
      score_src_y[part_id].resize(n_orientations);

      if(is_leaf) {
        // the leaf has no child-score to combine
        // we just do the distance transform for each rotation / cluster
        Mat_<float> score_tmp = part->get_appeareance_score_copy();
        for( int i=0; i < n_orientations; i++) {
          const Point& shift  = part->get_offset(i);
          const Quadratic& fx = part->get_cost_function_x(i);
          const Quadratic& fy = part->get_cost_function_y(i);
          DistanceTransform<float> td;
          td.compute(score_tmp, fx, fy, shift,
              scores[part_id][i],
              score_src_x[part_id][i],
              score_src_y[part_id][i]);
        }

      }else{
        // the normal case. the part has a child(s) and a parent.
        // combine min-score of childs and add appeareance score
        for( int i=0; i < n_orientations; i++) {

          Mat_<float> score_tmp = part->get_appeareance_score_copy();
          for( int j = 0; j < child_ids.size(); j++){
            int child_id = child_ids[j];
            Mat_<float> min_score_chid;
            const Part* child = model->get_part(child_id);
            vector<vector<float> > pairwise_weights = child->get_pairwise_weights();
            learning::ps::utils::reduce_min(scores[child_id], pairwise_weights[i], min_score_chid);
            add(score_tmp, min_score_chid, score_tmp);
          }

          const Point& shift  = part->get_offset(i);
          const Quadratic& fx = part->get_cost_function_x(i);
          const Quadratic& fy = part->get_cost_function_y(i);
          DistanceTransform<float> td;
          td.compute(score_tmp, fx, fy, shift,
              scores[part_id][i],
              score_src_x[part_id][i],
              score_src_y[part_id][i]);
        }
      }
    }else{

      // combine min-score of childs and add appeareance score
      // the root has only one orientation
      CHECK_EQ(part_id, 0);
      scores[part_id].resize(1);
      score_src_x[part_id].clear();
      score_src_y[part_id].clear();
      scores[part_id][0] = part->get_appeareance_score_copy();

      for( int i = 0; i < child_ids.size(); i++){
        int child_id = child_ids[i];
        Mat_<float> min_score_chid;

        const Part* child = model->get_part(child_id);
        vector<vector<float> > pairwise_weights = child->get_pairwise_weights();
        learning::ps::utils::reduce_min(scores[child_id], pairwise_weights[0], min_score_chid);
        add(scores[part_id][0], min_score_chid, scores[part_id][0]);
      }

    }

    // check if root
    if( is_root ) {
      double min;
      minMaxLoc(scores[0][0], &min, 0 , 0, 0);
      _inferenz_score =  static_cast<float>(min);
      return static_cast<float>(min);
    }else{
      return 0;
    }
  }


  void PairInferenz::compute_arg_min( vector<Point_<int> >& min_locations) const {
    vector<int> min_orientations(min_locations.size(), -1);
    compute_arg_min(min_locations, min_orientations, 0);

//    LOG(INFO) << "min_orientations " << ::utils::VectorToString(min_orientations);
  }

  void PairInferenz::compute_arg_min( vector<Point_<int> >& min_locations,
      vector<int>& min_orientations,
      int part_id) const {

    CHECK_EQ(min_locations.size(), scores.size());
    int parent_id = model->get_parent_id(part_id);
    // the anker point is the location of the parent
    // and for the root the anker point is the maximum of the score mat
    Point_<int> min_loc;
    if(model->has_parent(part_id)) {
      Point anker;
      anker = min_locations[ parent_id ];

      const Part* part = model->get_part(part_id);
      vector<vector<float> > pairwise_weights = part->get_pairwise_weights();

      // search best orientation
      int n_orientations = score_src_x[part_id].size();
      float min_v = boost::numeric::bounds<double>::highest();
      int part_orientation = min_orientations[parent_id];
      CHECK_GT(part_orientation, -1);
      for(int i=0; i < n_orientations; i++) {
        float w = pairwise_weights[part_orientation][i];
        if(w >=0 && min_v > scores[part_id][i](anker)+w) {
          min_loc.x = score_src_x[part_id][i](anker);
          min_loc.y = score_src_y[part_id][i](anker);
          min_v = scores[part_id][i](anker)+w;
          min_orientations[part_id] = i;
        }
      }
    }else{
      CHECK_EQ(part_id, 0); // root must be 0.
      minMaxLoc(scores[part_id][0], 0, 0, &min_loc, 0);
      min_orientations[part_id] = 0;

    }

    min_locations[part_id] = min_loc;

    // recursive call
    vector<int> child_ids = model->get_children_ids(part_id);
    for(int i=0; i < child_ids.size(); i++) {
      compute_arg_min( min_locations, min_orientations, child_ids[i]);
    }
  }

  float pair_inferenz_multiple(std::vector<Model> models,
       const std::vector<cv::Mat_<float> >& apperance_scores,
       std::vector<cv::Point_<int> >& min_locations, bool debug, const Mat& img) {

     // seting voting maps;
     vector<PairInferenz> solvers;
     for(int j=0; j < models.size(); j++) {
       models[j].set_voting_maps(apperance_scores, -1);
       solvers.push_back(PairInferenz(&models[j]) );

     }

     // multithreaded inferenz
     int num_threads = ::utils::system::get_available_logical_cpus();
     if(num_threads > 1 && models.size() > 1 && !debug) {
       boost::thread_pool::executor e(num_threads);
       for(int i=0; i < solvers.size(); i++) {
         e.submit(boost::bind(&PairInferenz::compute_min, &solvers[i], 0 ));
       }
       e.join_all();

     // singlethreaded inferenz
     }else{
       for(int i=0; i < solvers.size(); i++) {
         solvers[i].compute_min();
       }
     }

     // check min
     float min_inferenz = boost::numeric::bounds<double>::highest();
     int min_index = 0;
     for(int i=0; i < solvers.size(); i++) {
       float inferenz = solvers[i].get_score();
       if( inferenz < min_inferenz ) {
         min_inferenz = inferenz;
         min_index = i;
       }
     }


     if(debug) {
       for(int i=0; i < solvers.size(); i++) {
         LOG(INFO) << i << " -> " << solvers[i].get_score();
         solvers[i].compute_arg_min(min_locations);
         Mat p = img.clone();
         plot( p, min_locations);
       }

       LOG(INFO) << "best: " << min_index << " " <<  min_inferenz;

     }

     solvers[min_index].compute_arg_min(min_locations);
     return min_inferenz;
   }

} /* namespace ps */
} /* namespace learning */
