#include <istream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/format.hpp>
#include <boost/random.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <libconfig.h++>

#include "cpp/learning/forest/forest.hpp"
#include "cpp/learning/pictorial_structure/learn_model_parameter.hpp"
#include "cpp/learning/pictorial_structure/inferenz.hpp"

#include "cpp/body_pose/body_part_sample.hpp"
#include "cpp/body_pose/utils.hpp"
#include "cpp/body_pose/clustering/body_clustering.hpp"
#include "cpp/body_pose/body_pose_types.hpp"
#include "cpp/vision/image_utils.hpp"
#include "cpp/vision/features/cnn/cnn_features.hpp"
#include "cpp/utils/pyramid_stitcher/pyramid_stitcher.hpp"

using namespace std;
using namespace cv;

using boost::shared_ptr;
using namespace boost::assign;

using namespace learning::forest::utils;
using namespace learning::forest;
using namespace learning::ps;
using namespace libconfig;

using learning::forest::utils::get_voting_map_cnn;

typedef BodyPartSample S;

bool run_pose_estimation(std::string config_file)
{
    /// load parameters from config file
    Config config;
    try
    {
        config.readFile(config_file.c_str());
    }
    catch(const FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        return(EXIT_FAILURE);
    }
    catch(const ParseException &pex)
    {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
        return(EXIT_FAILURE);
    }
    string dataset_name = config.lookup("dataset_name");
    int num_actions     = config.lookup("num_actions");
    bool save           = config.lookup("save");
    string cache        = config.lookup("cache");
    std::string experiment_name = config.lookup("experiment_name");
    int num_threads     = config.lookup("num_threads");

    string forest_path  = config.lookup("forest_path");
    int num_trees       = config.lookup("num_trees");
    int norm_size       = config.lookup("norm_size");

    int p_type          = config.lookup("pose_type");
    body_pose::BodyPoseTypes pose_type = static_cast<body_pose::BodyPoseTypes>(p_type);

    int num_clusters    = config.lookup("ps_num_clusters");
    JointParameter joint_param;
    joint_param.joint_type          = learning::ps::CLUSTER_GAUSS;
    joint_param.num_rotations       = config.lookup("ps_num_rotations");//boost::lexical_cast<int>(argv[2]);
    joint_param.use_weights         = config.lookup("ps_use_weights");
    joint_param.weight_alpha        = config.lookup("ps_weight_alpha");
    joint_param.zero_sum_weights    = config.lookup("ps_zero_sum_weights");
    joint_param.min_sampels         = config.lookup("ps_min_samples");
    vector<JointParameter> joints_param((int)pose_type, joint_param);

    string train_file   = config.lookup("train_file");
    string test_file    = config.lookup("test_file");
    int start_idx       = config.lookup("start_index");
    int end_idx         = config.lookup("end_index");


    LOG(INFO)<<"Experiment name: "<<experiment_name;
    LOG(INFO)<<"Train file: "<<train_file;
    LOG(INFO)<<"Test file: "<<test_file;

    // loading test images
    LOG(INFO)<<"Loading annotation...";
    vector<Annotation> annotations;
    CHECK(load_annotations(annotations, test_file));
    LOG(INFO)<<"Found "<<annotations.size()<<" test images";

    if(end_idx <= 0){
        end_idx = annotations.size();
    }


    forest_path += "/"+dataset_name+"/";
    LOG(INFO)<<"Path to forest: "<<forest_path;
    vector<Forest<S> >  forests;
    load_forests(forest_path, forests, num_trees);

    ForestParam param;
    param = forests[0].getParam();
    param.norm_size = norm_size;
    param.img_index_path = test_file;

    /// CNN params
    const Setting &root = config.getRoot();
    Setting &cnn_params = root["cnn_params"];
    string pretrained_net_proto = (const char*)cnn_params["pretrained_net_proto"];
    string feature_extraction_proto = (const char*)cnn_params["feature_extraction_proto"];
    Scalar mean_pixel;
    mean_pixel.val[0] = (double)cnn_params["mean_pixel"][0][0];
    mean_pixel.val[1] = (double)cnn_params["mean_pixel"][0][1];
    mean_pixel.val[2] = (double)cnn_params["mean_pixel"][0][2];

    vision::features::CNNFeatures cnn_feat_extractor(pretrained_net_proto, feature_extraction_proto, "", true);
    cnn_feat_extractor.set_mean_pixel(mean_pixel);
    cv::Size cnn_geometry = cnn_feat_extractor.get_input_geometry();
    vector<string> feature_names;
    feature_names += "conv2_2","conv3_3","conv4_3","conv5_3";

    std::vector<int> actionLabels(boost::counting_iterator<int>( 0 ),
                        boost::counting_iterator<int>( num_actions ));

    /// create PS models for every action class
    std::vector<std::vector<Model> > models(actionLabels.size());
    body_pose::clustering::ClusterMethod method = body_pose::clustering::GLOBAL_POSE;
    for(unsigned int mIdx = 0; mIdx < actionLabels.size(); mIdx++){
      vector<int> labels(1,actionLabels[mIdx]);
      learning::ps::get_body_model_mixtures(train_file, method, num_clusters, "",
                                        models[mIdx], pose_type, labels, joints_param,
                                        param.norm_size, false);
      LOG(INFO) <<models[mIdx].size() << ": model(s) created for action class"<<actionLabels[mIdx];
    }

    std::vector<std::vector<std::vector<double> > > learnt_weights(actionLabels.size());
    std::string weights_path = config.lookup("app_sharing_weights");
    for(unsigned int i=0; i<actionLabels.size(); i++){
      std::string weights_file = boost::str(boost::format("%1%/%2%/class_%3%.txt")
                                                    %weights_path %dataset_name %i);
      load_learnt_weights(weights_file, learnt_weights[i]);
    }

    std::string action_probs_path = config.lookup("action_probs_file");
    std::string action_probs_file = boost::str(boost::format("%1%/%2%/action_probs.txt")
                                                    %action_probs_path %dataset_name);
    cv::Mat action_probs;
    readMatAsciiWithoutHeader(action_probs_file,action_probs, 1020, num_actions);

    ofstream outFile;
    string file_name(boost::str(boost::format("%s/%s_m%d_%sX_t%d_stIdx%06d_endIdx%06d.txt")
                                        %cache %experiment_name %num_clusters
                                        %joint_param.to_string() %num_trees %start_idx %end_idx));

    if(save) {
      LOG(INFO) <<  "file_name: " << file_name << endl;
      outFile.open(file_name.c_str(), ios::out);
    }

    int padding = 16;
    int nScales = 5;
    int img_minWidth = 50; //just a test
    int img_minHeight = 50;
    int upsample_factor = 1;
    float scale_factor = 0.85;

    int videoId = -1;
    std::vector<int> estActionLabels;
    std::vector<int> estActionLabelsIdx;
    vector<Model> selected_models;

    vector<int> part_ids;
    boost::progress_display show_progress(annotations.size());

    for (int i = 0; i < end_idx; ++i)
    {

      Mat img = imread(annotations[i].url,1);

      std::vector<std::string> strs;
      boost::split(strs, annotations[i].url, boost::is_any_of("/"));
      std::vector<std::string> strs1;
      boost::split(strs1, strs.back(), boost::is_any_of("."));
      int frame_index = boost::lexical_cast<int>(strs1.front());

      if(frame_index == 1){
        videoId++;

        estActionLabels.clear();
        estActionLabelsIdx.clear();
        selected_models.clear();

        cv::Mat probs_i = action_probs.row(videoId).clone();
        normalize(probs_i, probs_i, 1, 0, NORM_L1);

        Point maxLoc;
        double max_prob = 0;
        minMaxLoc(probs_i, NULL, &max_prob, NULL, &maxLoc);
        probs_i.at<double>(maxLoc.y, maxLoc.x) = -999;

        estActionLabels.push_back(actionLabels[maxLoc.x]);
        estActionLabelsIdx.push_back(maxLoc.x);

        for(unsigned int nIdx = 0; nIdx < models[maxLoc.x].size(); nIdx++){
          selected_models.push_back(models[maxLoc.x][nIdx]);
          selected_models[selected_models.size()-1].set_weight(static_cast<float>(max_prob));
        }
      }

      if(i < start_idx)
        continue;

      Patchwork patchwork = stitch_pyramid(img, img_minWidth, img_minHeight,
                                    padding, nScales, cnn_geometry.width,
                                    cnn_geometry.height, mean_pixel);

      int n_planes = patchwork.planes_.size();
      std::vector< std::vector<cv::Mat> > cnn_features(n_planes);

      for(int p=0; p<n_planes;p++){
        cnn_feat_extractor.extract(patchwork.planes_[p], feature_names, cnn_features[p], true);
      }

      vector<ScaleLocation> scaleLocations =  unstitch_pyramid_locations(patchwork, 1);
      CHECK_EQ(scaleLocations.size(), nScales);

      vector<cv::Point> minimas_rescaled;
      float min_score = std::numeric_limits<float>::max();
      for(int i_scale =0; i_scale < nScales; i_scale++) {

        cv::Rect scale_bbox(scaleLocations[i_scale].xMin + padding,
                            scaleLocations[i_scale].yMin + padding,
                            scaleLocations[i_scale].width - 2*padding,
                            scaleLocations[i_scale].height - 2*padding);
        int plane_id = scaleLocations[i_scale].planeID;

        Image sample;
        for(unsigned int fIdx=0; fIdx<cnn_features[plane_id].size(); fIdx++){
          sample.add_feature_channel(cnn_features[plane_id][fIdx](scale_bbox));
//          cv::Mat tmp = cnn_features[plane_id][fIdx](scale_bbox);
//          imshow("cnn_features", tmp);
//          waitKey(0);
        }
        sample.set_global_attr_label(estActionLabelsIdx[0]);

        vector<Mat_<float> > voting_maps;
        eval_forests_cnn(forests,sample, voting_maps, 1, true, num_threads, true, learnt_weights[estActionLabelsIdx[0]]);
        for(int j=0; j < voting_maps.size(); j++) {
          normalize(voting_maps[j], voting_maps[j], 0, 1, CV_MINMAX);
//           imshow("X", voting_maps[j]);
//           waitKey(0);
        }

        vector<Point> minimas;
        minimas.resize(annotations[i].parts.size(), Point(-1,-1));

        float score = learning::ps::inferenz_multiple(selected_models, voting_maps, minimas, false, img);

        if(score < min_score){
          min_score = score;
          double scale = upsample_factor*pow(scale_factor, i_scale);

          minimas_rescaled = minimas;
          for(int j=0; j < minimas_rescaled.size(); j++) {
            minimas_rescaled[j].x /= scale;
            minimas_rescaled[j].y /= scale;
          }
        }
     }
     if(save){
      outFile << annotations[i].url <<" 0 0 0 0 0 "<< annotations[i].parts.size() << " ";
      assert( minimas_rescaled.size() == annotations[i].parts.size() );
      for( int j=0; j < minimas_rescaled.size(); j++){
        Point pt = minimas_rescaled[j];
        outFile << pt.x  << " " << pt.y << " ";
      }
      outFile<<"\n";
      outFile.flush();

    }else{
      plot(img, minimas_rescaled, pose_type);
    }
  }

  LOG(INFO) << "DONE";
  return true;
}

int main(int argc, char** argv)
{
    string config_file = "./config_file.txt";
    if(argc > 1){
        config_file = argv[1];
    }
    CHECK(run_pose_estimation(config_file));
    return 0;
}
