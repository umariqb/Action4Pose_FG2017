#include "cpp/body_pose/nbest_body_pose_utils.hpp"

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
                            bool use_context, bool do_classification){

    cv::Mat image;
    if (img_org.data == NULL) {
      cerr << "could not load " << ann.url << endl;
      return false;
    }else{
      if( abs(scale - 1) > 0.05 ){
        cv::resize(img_org, image, Size(img_org.cols * scale, img_org.rows * scale), 0, 0);
      }else{
        scale = 1;
        image = img_org;
      }
    }

    Rect extracted_region(0,0,0,0);
    int x1 = ann.bbox.x;
    int y1 = ann.bbox.y;
    int x2 = x1 + ann.bbox.width;
    int y2 = y1 + ann.bbox.height;
    int offset_x = static_cast<int>(0.27*((x2-x1)+(y2-y1)));
    int offset_y = static_cast<int>(0.4*((x2-x1)+(y2-y1)));

    if(true){
      extract_roi(image, ann, image, offset_x, offset_y, &extracted_region);
    }

    Image sample(image, param.features, fcf, false);

    if(use_context) {
      CHECK_GT(context_forests.size(), 0);
      vector<cv::Mat_<float> > voting_maps;
      eval_forests(context_forests, sample, voting_maps, 1, false );
      for(int t=0; t< voting_maps.size(); t++ ) {
        Mat v_8C;
        voting_maps[t].convertTo(v_8C, CV_8UC1, 255);
        sample.add_feature_channel(v_8C);
      }
    }

    // get voting maps;
    vector<Mat_<float> > voting_maps;
    if(do_classification) {
      Rect roi(0,0, sample.width(), sample.height());
      get_multiple_forground_map(sample, forests, voting_maps, roi, 2);
    }else{
      eval_forests(forests,sample, voting_maps, 1, true);
      for(int t=0; t < voting_maps.size(); t++) {
        normalize(voting_maps[t], voting_maps[t], 0, 1, CV_MINMAX);
      }
    }

    learning::ps::inferenz_nbest_max_decoder(models, voting_maps, poses, pose_type, nbest_part_ids, false, image ,0 ,nbest_num, true);
    if(false){
      for(unsigned int pose_id=0; pose_id < poses.size(); pose_id++){
        plot(image, poses[pose_id].parts , pose_type);
      }
    }

    for(unsigned int pose_id=0; pose_id<poses.size(); pose_id++){
      for(unsigned int part_id=0; part_id<poses[pose_id].parts.size(); part_id++){
        poses[pose_id].parts[part_id].x += extracted_region.x;
        poses[pose_id].parts[part_id].y += extracted_region.y;
        poses[pose_id].parts[part_id].x /= scale;
        poses[pose_id].parts[part_id].y /= scale;
      }
    }
}

bool detect_nbest_multiscale(cv::Mat& img_org,
                             Annotation& ann, std::vector<NBestInferenz::Pose>& poses,
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
                            bool use_context, bool do_classification)
{

    cv::Mat image;
    Rect extracted_region(0,0,0,0);
    int x1 = ann.bbox.x;
    int y1 = ann.bbox.y;
    int x2 = x1 + ann.bbox.width;
    int y2 = y1 + ann.bbox.height;
    int offset_x = static_cast<int>(0.27*((x2-x1)+(y2-y1)));
    int offset_y = static_cast<int>(0.4*((x2-x1)+(y2-y1)));

    if(true){
      extract_roi(img_org, ann, image, offset_x, offset_y, &extracted_region);
    }
    else{
      image = img_org;
    }

    vector<Mat> pyramid;
    float initial_scale = vision::image_utils::build_image_pyramid(image,
        pyramid_param.scale_count,
        pyramid_param.scale_factor, pyramid,
        max_side_length);

    vector<Image> image_samples;
    create_image_sample_mt(pyramid, param.features, image_samples);

    float min_score = 0;

    std::vector<NBestInferenz::Pose> detected_poses, cleaned_poses;

    for(int i_scale =0; i_scale < pyramid_param.scale_count; i_scale++) {

      if(use_context) {
        CHECK_GT(context_forests.size(), 0);
        vector<cv::Mat_<float> > voting_maps;
        eval_forests(context_forests, image_samples[i_scale], voting_maps, 1, false );
        for(int j=0; j< voting_maps.size(); j++ ) {
          Mat v_8C;
          voting_maps[j].convertTo(v_8C, CV_8UC1, 255);
          image_samples[i_scale].add_feature_channel(v_8C);
        }
      }

      // get voting maps;
      vector<Mat_<float> > voting_maps;
      if(false) {
        Rect roi(0,0, image_samples[i_scale].width(), image_samples[i_scale].height());
        get_multiple_forground_map(image_samples[i_scale], forests, voting_maps, roi, 2);

      }else{
        eval_forests(forests, image_samples[i_scale], voting_maps, 2, true);
        for(int j=0; j < voting_maps.size(); j++) {
          normalize(voting_maps[j], voting_maps[j], 0, 1, CV_MINMAX);
        }
      }

      std::vector<NBestInferenz::Pose> poses_for_each_scale;
      learning::ps::inferenz_nbest_max_decoder(models, voting_maps, poses_for_each_scale, pose_type, nbest_part_ids, false, image ,0 ,nbest_num, true);

      double scale = initial_scale*pow(pyramid_param.scale_factor, i_scale);
      for(unsigned int pose_id=0; pose_id<poses_for_each_scale.size(); pose_id++){
        for(unsigned int part_id=0; part_id<poses_for_each_scale[pose_id].parts.size(); part_id++){
          poses_for_each_scale[pose_id].parts[part_id].x /= scale;
          poses_for_each_scale[pose_id].parts[part_id].y /= scale;
          poses_for_each_scale[pose_id].parts[part_id].x += extracted_region.x;
          poses_for_each_scale[pose_id].parts[part_id].y += extracted_region.y;
        }
      }

      detected_poses.insert(detected_poses.end(), poses_for_each_scale.begin(), poses_for_each_scale.end());
    }

    // eliminate overlappig poses
    eliminate_overlapping_poses(detected_poses, cleaned_poses,0.05);

    //sort with respect to cost
    std::sort(cleaned_poses.begin(), cleaned_poses.end(), NBestInferenz::by_inferenz_score());

    int available_n = std::min(static_cast<int>(cleaned_poses.size()), nbest_num);
    poses.insert(poses.end(), cleaned_poses.begin(),
                        cleaned_poses.begin()+available_n);

}

