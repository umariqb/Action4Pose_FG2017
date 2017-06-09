/*
 * create_clusters.cpp
 *
 *  Created on: Feb 27, 2013
 *      Author: mdantone
 */


#include <opencv2/opencv.hpp>

#include <istream>
#include <boost/filesystem/path.hpp>
#include <boost/assign/std/vector.hpp>
#include "cpp/utils/file_utils.hpp"
#include "cpp/body_pose/utils.hpp"
#include "body_clustering.hpp"
#include "cpp/body_pose/body_pose_types.hpp"

using namespace boost::assign;
using namespace std;
using namespace cv;
using namespace body_pose::clustering;

int main(int argc, char** argv) {

  body_pose::BodyPoseTypes pose_type = body_pose::FULL_BODY_J13;

  srand(1);
  vector<int> parents;
  get_joint_constalation(parents, pose_type);

  vector<Annotation> org_annotations;
  string index_file;

  index_file = "/home/ibal_109/work/Datasets/Fashion_Pose/Train/image_index_train_rescaled.txt";
  load_annotations(org_annotations, index_file);

  LOG(INFO) << org_annotations.size() << " found.";
  vector<Annotation> cleaned_annotations;
  clean_annotations(org_annotations, cleaned_annotations);
  CHECK_GT( cleaned_annotations.size(), 0);
  LOG(INFO) << cleaned_annotations.size() << " cleaned.";

  string path = "/home/ibal_109/work/2014/Pose_Estimation_Code_v1.1/clusters/FashionPose/";

  ClusterMethod method = PART_POSE;
  method = GLOBAL_POSE;
//  method = GIST_APPEARANCE;
//  method = BOW_SPM;

  vector<int> cluster_sizes;
  cluster_sizes += 50, 100;

  vector<int> labels;
  labels += 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 17, 19;

  for(unsigned int aIdx = 0; aIdx < labels.size(); aIdx++){
      string action_dir(boost::str(boost::format("%1%/action_class_%2%") %path %labels[aIdx]));
      std::vector<int> actionLabel(1, labels[aIdx]);

      std::vector<Annotation> annotations;

      getAnnotationsWithClusterId(cleaned_annotations, annotations, actionLabel, pose_type, true);
      LOG(INFO)<<"Selected "<<annotations.size()<<" annotations for clustering";

      if(!boost::filesystem::exists(action_dir.c_str())) {
        CHECK(boost::filesystem::create_directory(action_dir));
      }

      for(int j = 0; j < cluster_sizes.size(); j++) {
        int n_clusters = cluster_sizes[j];

        vector<Mat> clusters;
        // try to load clusters
        if(load_clusters(action_dir, n_clusters, method, clusters, pose_type) ) {
          CHECK_EQ(clusters.size(), parents.size());

          std::vector<int> part_indices;
          bool visualize = true;
          if(visualize) {
            if(method == PART_POSE) {
              for(int i=0; i < parents.size(); i++) {
                part_indices.push_back(i);
                part_indices.push_back( parents[i] );

                visualize_part_clusters(annotations, method, part_indices, clusters[i], pose_type);
              }
            }else{
              visualize_part_clusters(annotations, method, part_indices, clusters[0], pose_type);
            }
          }

          bool save = false;
          if(save){
            if(method != PART_POSE) {

              vector<Annotation> clustered_annotations;
              assigne_to_clusters (annotations, part_indices, clusters[0],
                  method, clustered_annotations, pose_type);
              CHECK( clustered_annotations.size() > 0 );
              save_clustrered_annotations(action_dir, index_file, n_clusters, method,
                  clustered_annotations, pose_type);
            }
            // store annotations
          }

          bool print = true;
          if(print){

          }
        }else{
          // create clusters and store them
          std::vector<int> part_indices;
          if(method == PART_POSE) {
            for(int i=0; i < parents.size(); i++) {
              part_indices.push_back(i);
              part_indices.push_back( parents[i] );
              Mat cluster;
              cluster_annotations(annotations, method, n_clusters, cluster, pose_type, part_indices);
              clusters.push_back(cluster);
            }
          }else{
            Mat cluster;
            cluster_annotations(annotations, method, n_clusters, cluster, pose_type, part_indices);

            for(int i=0; i < parents.size(); i++) {
              clusters.push_back(cluster);
            }
          }
          save_clusters(action_dir, n_clusters, method, clusters, pose_type);
        }
      }
  }
  return 0;
}

