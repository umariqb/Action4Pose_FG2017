#ifndef NBEST_BODY_POSE_UTILS_H
#define NBEST_BODY_POSE_UTILS_H

#include <istream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/format.hpp>
#include <boost/random.hpp>

#include "cpp/utils/timing.hpp"
#include "cpp/learning/forest/forest.hpp"
#include "cpp/learning/pictorial_structure/learn_model_parameter.hpp"
#include "cpp/learning/pictorial_structure/pair_inferenz.hpp"
#include "cpp/learning/pictorial_structure/nbest_inferenz.hpp"

#include "cpp/learning/pictorial_structure/utils.hpp"

#include "cpp/body_pose/body_part_sample.hpp"
#include "cpp/body_pose/utils.hpp"
#include "cpp/utils/thread_pool.hpp"
#include "cpp/body_pose/clustering/body_clustering.hpp"
#include "cpp/vision/features/feature_channels/feature_channel_factory.hpp"
#include "cpp/body_pose/body_pose_types.hpp"
#include "cpp/third_party/MixingBodyPartSequences/mixing_body_part_sequences.hpp"
#include "cpp/vision/optical_flow_utils.hpp"
#include "cpp/vision/image_utils.hpp"

using namespace boost::assign;
using namespace std;
using namespace cv;
using namespace learning::forest::utils;
using namespace learning::forest;
using namespace learning::ps;


bool detect_nbest_rescaled(cv::Mat& img_org, Annotation& ann, std::vector<NBestInferenz::Pose>& poses,
                            float scale,
                            vector<Forest<BodyPartSample> >& forests,
                            vector<Forest<BodyPartSample> >& context_forests,
                            vector<Model>& models,
                            body_pose::BodyPoseTypes pose_type,
                            int nbest_num,
                            std::vector<int>& nbest_part_ids,
                            FeatureChannelFactory& fcf,
                            learning::forest::ForestParam& param,
                            bool use_context = false, bool do_classification = false);

bool detect_nbest_multiscale(cv::Mat& img_org, Annotation& ann,
                            std::vector<NBestInferenz::Pose>& poses,
                            vector<Forest<BodyPartSample> > forests,
                            vector<Forest<BodyPartSample> > context_forests,
                            vector<Model>& models,
                            learning::common::utils::PeakDetectionParams& pyramid_param,
                            int max_side_length,
                            body_pose::BodyPoseTypes pose_type,
                            int nbest_num,
                            std::vector<int>& nbest_part_ids,
                            FeatureChannelFactory& fcf,
                            learning::forest::ForestParam& param,
                            bool use_context = false, bool do_classification = false);

#endif // NBEST_BODY_POSE_UTILS_H
