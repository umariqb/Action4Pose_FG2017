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

#ifndef NBESTINFERENZ_H
#define NBESTINFERENZ_H

#include "cpp/learning/pictorial_structure/inferenz.hpp"
#include "cpp/learning/pictorial_structure/model.hpp"
#include "cpp/body_pose/body_pose_types.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/algorithm/string.hpp>

namespace learning
{
namespace ps
{

class NBestInferenz : public Inferenz
{
  public:

    struct Pose{
      std::vector<cv::Point_<int> > parts;
      double inferenz_score; // total cost for the part
      double init_part_id; //id of the root part
      std::vector<double> part_wise_costs;

      Pose():parts(), inferenz_score(0), init_part_id(0), part_wise_costs(){}

      Pose(std::vector<cv::Point_<int> > parts_, double inferenz_score_):
                        parts(parts_), inferenz_score(inferenz_score_), init_part_id(0), part_wise_costs(){}

      Pose(std::vector<cv::Point_<int> > parts_, double inferenz_score_, double init_part_id_):
                        parts(parts_), inferenz_score(inferenz_score_), init_part_id(init_part_id_), part_wise_costs(){}

      Pose(std::vector<cv::Point_<int> > parts_, double inferenz_score_, double init_part_id_, std::vector<double> part_wise_costs_):
                        parts(parts_), inferenz_score(inferenz_score_), init_part_id(init_part_id_), part_wise_costs(part_wise_costs_){}

      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & parts;
        ar & inferenz_score;
        ar & init_part_id;
        ar & part_wise_costs;
      }
    };

    struct by_inferenz_score {
      bool operator()( Pose const &a, Pose const &b) {
        return a.inferenz_score < b.inferenz_score;
      }
    };

    NBestInferenz(Model* m): Inferenz(m),
                             min_marginals_computed(false),
                             poses_computed(false)
    {
      int n_parts = Inferenz::model->get_num_parts();
    // min marginals for each part
    min_marginals.resize(n_parts);
    score_src_x_d.resize(n_parts);
    score_src_y_d.resize(n_parts);

    };

    virtual ~NBestInferenz();
    bool compute_min_marginals();
    bool compute_diverse_poses(float threshold, bool save_partwise_costs,
                                body_pose::BodyPoseTypes pose_type,
                                std::vector<int>& nbest_part_ids);
    bool nms_loop(cv::Mat_<float>& cost_img,
                             std::vector<cv::Point_<int> >& min_locs,
                             float threshold, int patch_size = 2,
                             int max_count = 1000);

    bool normalize_min_marginals();

    bool compute_arg_mins_mm(int part_id,
                              std::vector<cv::Point_<int> >& min_locs,
                              std::vector<int>& old_parents,
                              std::vector<int>& new_parents);

    bool reorder_parts(int new_root, std::vector<int>& new_parents,  body_pose::BodyPoseTypes pose_type);

    bool get_poses(std::vector<std::vector<cv::Point_<int> > >& out_poses);
    bool get_poses(std::vector<Pose>& out_poses);
  protected:

  private:
    bool min_marginals_computed;
    bool poses_computed;
    std::vector<cv::Mat_<float> > min_marginals;
    std::vector<cv::Mat_<int> > score_src_x_d;
    std::vector<cv::Mat_<int> > score_src_y_d;
    std::vector<Pose> poses;
};

bool eliminate_overlapping_poses(std::vector<NBestInferenz::Pose>& poses
                                 ,std::vector<NBestInferenz::Pose>& nms_poses,
                                 double threshold = 0.02,
                                 int upper_body_size = 50);


float inferenz_nbest_max_decoder(std::vector<Model> models,
    const std::vector<cv::Mat_<float> >& apperance_scores,
      std::vector<NBestInferenz::Pose>& final_poses,
        body_pose::BodyPoseTypes pose_type,
        std::vector<int>& nbest_part_ids, bool debug = false,
        const cv::Mat& img = cv::Mat(), float weight = 0.0,
        unsigned int N = 100, bool save_partwise_costs = false);

float inferenz_nbest_max_decoder(std::vector<Model> models,
    const std::vector<cv::Mat_<float> >& apperance_scores,
    std::vector<std::vector<cv::Point_<int> > >& min_locations,
     body_pose::BodyPoseTypes pose_type,
    std::vector<int>& nbest_part_ids, bool debug = false,
    const cv::Mat& img = cv::Mat(), float weight = 0.0,
    unsigned int N = 100, bool save_partwise_costs = false);

bool load_nbest_poses(std::string path, std::vector<NBestInferenz::Pose>& poses);
bool save_nbest_poses(std::vector<NBestInferenz::Pose>& poses, std::string path);


} /* namespace ps */
} /* namespace learning */

#endif // NBESTINFERENZ_H
