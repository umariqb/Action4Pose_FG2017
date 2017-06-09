/*!
*
* Class implements pictorial structure inference to obtain
* n best poses from an image following the article:
*
* D. Park, D. Ramanan. "N-Best Maximal Decoders for Part Models"
* International Conference on Computer Vision (ICCV) Barcelona, Spain,
* November 2011.
*
* @Author: uiqbal
* @Date: 06.06.2014
*
*/

#include "cpp/learning/pictorial_structure/nbest_inferenz.hpp"
#include "cpp/learning/pictorial_structure/utils.hpp"
#include "cpp/learning/pictorial_structure/math_util.hpp"
#include "cpp/learning/pictorial_structure/learn_model_parameter.hpp"
#include "cpp/learning/forest/param.hpp"

#include "cpp/utils/thread_pool.hpp"
#include "cpp/utils/system_utils.hpp"
#include "cpp/utils/string_utils.hpp"

#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


using namespace std;
using namespace cv;
namespace learning
{
namespace ps
{
  NBestInferenz::~NBestInferenz()
  {
    //dtor
  }

  /*!
  * Function takes the cost image and computes minimum
  * values by doing non minimum/maximum supression.
  */
  bool NBestInferenz::nms_loop(cv::Mat_<float>& cost_img,
                               std::vector<cv::Point_<int> >& min_locs,
                               float threshold, int patch_size, int max_count){
      patch_size = 15;
      cv::Mat_<float> cost = cost_img.clone();
      //normalize(cost, cost, 0, 1, CV_MINMAX);

      for(int i=0; i<max_count; i++){
        // get minimum
        Point_<int> min_index;
        double min_val = 0;
        minMaxLoc(cost, &min_val, 0, &min_index, 0);

        if(min_val > threshold){
          break;
        }
        min_locs.push_back(min_index);

        Rect box(min_index.x - patch_size/2, min_index.y - patch_size/2, patch_size, patch_size);
        Rect inter = utils::intersect(box, Rect(0,0,cost.cols,cost.rows));
        cost(inter).setTo(cv::Scalar(0));

//        cout<<min_val<<"\t"<<min_index.x<<"\t"<<min_index.y<<"\n";
//        imshow("cost", 1-cost);
//        waitKey(0);
      }
      return true;
  }


  /*!
  * Function changes the root in the graph and
  * shifts the parents accordingly.
  *
  * @param:
  *    part_id: is the part which we want to make root
  */
  bool NBestInferenz::reorder_parts(int new_root, std::vector<int>& new_parents, body_pose::BodyPoseTypes pose_type)
  {
    std::vector<int> parents;
    get_joint_constalation(parents, pose_type);
    get_joint_constalation(new_parents, pose_type);
    int par = parents[new_root];

    new_parents[par] = new_root;
    new_parents[new_root] = new_root;

    while(par != 0){
      int par_par = parents[par];
      new_parents[par_par] = par;
      par = par_par;
    }
    return true;
  }

  bool NBestInferenz::compute_arg_mins_mm(int part_id,
                                std::vector<cv::Point_<int> >& min_locs,
                                std::vector<int>& old_parents,
                                std::vector<int>& new_parents)
  {

    cv::Point_<int> par_loc = min_locs[part_id];

    // getting the childerns of part_id
    std::vector<int> child_ids;
    for(unsigned int i=0; i < new_parents.size(); i++){
      if(new_parents[i] == part_id && i != part_id){
        child_ids.push_back(i);
      }
    }

    // recursive call
    for(unsigned int i=0; i < child_ids.size(); i++){
      int cid = child_ids[i];
      cv::Point_<int> min_loc;
      if(old_parents[cid] == new_parents[cid]){
        min_loc.x = score_src_x[cid](par_loc);
        min_loc.y = score_src_y[cid](par_loc);
      }
      else{
        min_loc.x = score_src_x_d[part_id](par_loc);
        min_loc.y = score_src_y_d[part_id](par_loc);
      }
      min_locs[cid] = min_loc;
      compute_arg_mins_mm(child_ids[i], min_locs, old_parents, new_parents);
    }
    return true;
  }

  bool NBestInferenz::compute_diverse_poses(float threshold,
          bool save_partwise_costs, body_pose::BodyPoseTypes pose_type,
          std::vector<int>& nbest_part_ids)
  {
    CHECK(min_marginals_computed);
    unsigned int num_parts = min_marginals.size();
    std::vector<int> parents;
    get_joint_constalation(parents, pose_type);


    for(unsigned int pIdx=0; pIdx<nbest_part_ids.size(); pIdx++){

      unsigned int part_id = nbest_part_ids[pIdx];
      // extract several minimas
      std::vector<cv::Point> min_locs;
      nms_loop(min_marginals[part_id], min_locs, threshold);

      // make current part as root and reorder parents
      std::vector<int> new_parents;
      reorder_parts(part_id, new_parents, pose_type);
      CHECK_EQ(num_parts, new_parents.size());

      // compute argmins for each minima
      for(unsigned int i=0; i<min_locs.size(); i++){

        std::vector<cv::Point_<int> > parts(num_parts);
        parts[part_id] = min_locs[i];
        compute_arg_mins_mm(part_id, parts, parents, new_parents);

        std::vector<double> partwise_costs;
        if(save_partwise_costs){
          // saving contribution of each part to the cost function
          partwise_costs.resize(num_parts,0);

          // getting appearance and deformation cost for each part
          for(unsigned int j=0; j < parts.size(); j++){
            int parent_id = parents[j];
            double deformation_cost = scores[j](parts[parent_id]) -
                                  scores_without_dt[j](parts[j]);

//            LOG(INFO)<<scores[j].at<float>(parts[parent_id]) -
//                                  scores_without_dt[j].at<float>(parts[j]);

            //LOG(INFO)<<scores[j](parts[parent_id])<<"\t\t"<<scores_without_dt[j](parts[j]);
            //double appearance_cost = scores_without_dt[j](parts[j]);
            Part* part = model->get_part(part_id);

            const cv::Mat* app_score = part->get_appeareance_score();
            double appearance_cost = app_score->at<float>(parts[j]);

            // overall contribution of the part
            partwise_costs[j] = deformation_cost + appearance_cost;
            //cout<<partwise_costs[j]<<" ";
          }
          //cout<<endl;
        }
        Pose ps(parts, min_marginals[part_id](min_locs[i]), part_id, partwise_costs);
        poses.push_back(ps);

      }
    }
    return true;
  }

  bool NBestInferenz::normalize_min_marginals(){

    unsigned int num_parts = min_marginals.size();
    for(unsigned int i=0; i<num_parts; i++){
      scores[i] /= static_cast<float>(num_parts);
      scores_without_dt[i] /= static_cast<float>(num_parts);
      min_marginals[i] /= static_cast<float>(num_parts);
    }
  }

  bool NBestInferenz::compute_min_marginals(){

    CHECK(min_computed);
    CHECK(scores.size());
    CHECK(score_src_x_d.size());
    CHECK(score_src_y_d.size());

    unsigned int nParts = scores.size();
    cv::Mat_<float> score_tmp;
    cv::Mat_<float> score_tmp_dt;

    //min marginals for root are already available
    min_marginals[0] = scores[0];

    // computing min_marginals for all other parts
    for(unsigned part_id=1; part_id < nParts; ++part_id){
      const Part* part = model->get_part(part_id);
      int parent_id = model->get_parent_id(part_id);
      score_tmp = min_marginals[parent_id] - scores[part_id];
      part->transform_d(score_tmp,
          score_tmp_dt,
          score_src_x_d[part_id],
          score_src_y_d[part_id]);

      min_marginals[part_id] = score_tmp_dt + scores_without_dt[part_id];
    }

    min_marginals_computed = true;
    return min_marginals_computed;
  }

  bool NBestInferenz::get_poses(std::vector<Pose>& out_poses){
    out_poses.insert(out_poses.end(), poses.begin(), poses.end());
    return true;
  }

  bool NBestInferenz::get_poses(std::vector<std::vector<cv::Point_<int> > >& out_poses){

    int out_poses_size = out_poses.size();
    out_poses.resize(out_poses_size+poses.size());

    for(unsigned int i=0; i<poses.size(); i++){
      out_poses[i]=poses[i-out_poses_size].parts;
    }
    return true;
  }

  bool eliminate_overlapping_poses(std::vector<NBestInferenz::Pose>& poses
                                   ,std::vector<NBestInferenz::Pose>& nms_poses,
                                   double threshold, int upper_body_size)
  {
    CHECK(poses.size());
    nms_poses.clear();
    vector<bool> is_suppressed(poses.size(), false);

    for(unsigned int i=0; i < poses.size(); i++){
      if(is_suppressed[i]){
        continue;
      }
      for(unsigned int j=i+1; j < poses.size(); j++){
        if(is_suppressed[j]){
          continue;
        }

        NBestInferenz::Pose& pose_a = poses[i];
        NBestInferenz::Pose& pose_b = poses[j];

        // pose is overlapping if all parts overlap
        int ov_parts = 0;
        for(unsigned int n=0; n<pose_a.parts.size(); n++){
          double dist = sqrt(pow((pose_a.parts[n].x - pose_b.parts[n].x), 2) +
                          pow((pose_a.parts[n].y - pose_b.parts[n].y), 2));

          if(dist <= threshold*upper_body_size){
            ov_parts++;
          }
        }

        if(ov_parts == pose_a.parts.size()){
          if(pose_a.inferenz_score >= pose_b.inferenz_score){
            is_suppressed[i] = true;
            break;
          }
          else{
            is_suppressed[j] = true;
          }
        }
      }
    }

    for(unsigned int i=0; i<poses.size(); i++){
      if(!is_suppressed[i]){
        nms_poses.push_back(poses[i]);
      }
    }
    return true;
  }

  /*!***************************************************
  *       Function to get nbest maximum decoders
  ******************************************************/
  float inferenz_nbest_max_decoder(std::vector<Model> models,
      const std::vector<cv::Mat_<float> >& apperance_scores,
        std::vector<NBestInferenz::Pose>& final_poses,
          body_pose::BodyPoseTypes pose_type,
          std::vector<int>& nbest_part_ids,
          bool debug, const Mat& img, float weight,
          unsigned int N, bool save_partwise_costs)
  {

    float threshold = 0.1;

    CHECK(models.size());
    // seting voting maps;
    vector<NBestInferenz> solvers;
    for(int j=0; j < models.size(); j++) {
      models[j].set_voting_maps(apperance_scores, -1);
      solvers.push_back(NBestInferenz(&models[j]) );
    }

    // multithreaded inferenz
    int num_threads = ::utils::system::get_available_logical_cpus();
    if(num_threads > 1 && models.size() > 1 && !debug) {

      LOG(INFO)<<"MULTITHREADIUNG";
      // dynamic programming first pass
      boost::thread_pool::executor e(num_threads);
      for(unsigned int i=0; i < solvers.size(); i++) {
        e.submit(boost::bind(&NBestInferenz::compute_min, &solvers[i], 0 ));
      }
      e.join_all();

      // dynamic programming 2nd pass
      // computing min marginals for each part
      boost::thread_pool::executor f(num_threads);
      for(unsigned int i=0; i < solvers.size(); i++){
        f.submit(boost::bind(&NBestInferenz::compute_min_marginals, &solvers[i]));
      }
      f.join_all();

      boost::thread_pool::executor g(num_threads);
      for(unsigned int i=0; i < solvers.size(); i++){
        g.submit(boost::bind(&NBestInferenz::normalize_min_marginals, &solvers[i]));
      }
      g.join_all();

      //computing nbest poses using each model
      boost::thread_pool::executor h(num_threads);
      for(unsigned int i=0; i < solvers.size(); i++){
        //h.submit(boost::bind(&NBestInferenz::compute_diverse_poses, &solvers[i], threshold, save_partwise_costs, pose_type));
      }
      h.join_all();

    // singlethreaded inferenz
    }else{
      for(unsigned int i=0; i < solvers.size(); i++) {
        // dynamic programming first pass
        solvers[i].compute_min();
        // dynamic programming 2nd pass
        // computing min marginals for each part
        solvers[i].compute_min_marginals();
        //normalize min_marginals between 0 and 1
        solvers[i].normalize_min_marginals();
        // computing nbest poses
        solvers[i].compute_diverse_poses(threshold, save_partwise_costs, pose_type, nbest_part_ids);
      }
    }

    // obtain poses from all models
    std::vector<NBestInferenz::Pose> poses, poses_cleaned;
    for(unsigned int i=0; i<solvers.size(); i++){
      solvers[i].get_poses(poses);
    }

    // eliminate overlappig poses
    eliminate_overlapping_poses(poses, poses_cleaned,0.05);

    if(debug){
      LOG(INFO)<<"Before elim. : "<<poses.size()<<" After elim. : "<<poses_cleaned.size();
    }

    //sort with respect to cost
    std::sort(poses_cleaned.begin(), poses_cleaned.end(), NBestInferenz::by_inferenz_score());

    int available_n = std::min(static_cast<unsigned int>(poses_cleaned.size()), N);
    final_poses.insert(final_poses.end(), poses_cleaned.begin(),
                        poses_cleaned.begin()+available_n);

    LOG(INFO)<<"Total detected poses = "<<final_poses.size();


    return 0;
  }

  float inferenz_nbest_max_decoder(std::vector<Model> models,
      const std::vector<cv::Mat_<float> >& apperance_scores,
        std::vector<std::vector<cv::Point_<int> > >& min_locations,
          body_pose::BodyPoseTypes pose_type,
          std::vector<int>& nbest_part_ids,
          bool debug, const Mat& img, float weight,
          unsigned int N, bool save_partwise_costs)
  {

    std::vector<NBestInferenz::Pose> poses;

    inferenz_nbest_max_decoder(models, apperance_scores, poses,pose_type, nbest_part_ids, debug, img, weight, N, save_partwise_costs);

    int n_poses = std::min(N, static_cast<unsigned int>(poses.size()));
    min_locations.clear();
    min_locations.resize(n_poses);

    for(unsigned int i=0; i<n_poses; i++){
      min_locations[i]=poses[i].parts;

      if(debug){
        LOG(INFO)<<"Part_id = "<<poses[i].init_part_id<<"  Score = "<<poses[i].inferenz_score;
      }
    }
    return 0;
  }

  bool load_nbest_poses(std::string path, std::vector<NBestInferenz::Pose>& poses)
  {
    std::ifstream ifs(path.c_str());

    if(!ifs){
      LOG(INFO)<<"file not found.";
    }
    else{
      try{
        boost::archive::text_iarchive ia(ifs);
        ia>>poses;
        LOG(INFO)<<"Poses loaded";
        return true;
      }
      catch(boost::archive::archive_exception& ex){
        LOG(INFO)<<"Reload Tree: Archive exception during deserializiation: "
              <<ex.what();
          LOG(INFO)<<"not able to load poses from: "<<path;
      }
    }
    return false;
  }

  bool save_nbest_poses(std::vector<NBestInferenz::Pose>& poses, std::string path){
    try{
      std::ofstream ofs(path.c_str());
      if(ofs==0){
      LOG(INFO)<<"Error: Cannot open the given path to save detected poses.";
      return false;
      }
      boost::archive::text_oarchive oa(ofs);
      oa<<poses;
      ofs.flush();
      ofs.close();
      LOG(INFO)<<"Poses saved at :"<<path;
      return true;
    }
    catch(boost::archive::archive_exception& ex){
      LOG(INFO)<<"Archive exception during deserialization:" <<std::endl;
      LOG(INFO)<< ex.what() << std::endl;
      LOG(INFO)<< "it was file: "<<path;
    }
    return true;
  }

} /* namespace ps */
} /* namespace learning */

