/*
 * performance_metrics.cpp
 *
 * Implementations of various performance metrics
*
 *  Created on: April 25, 2013
 *      Author: Umar Iqbal
 *
 */
#include <math.h>
#include <vector>
#include "cpp/body_pose/performance_metrics.h"


double get_shoulder_hip_dist(Annotation gtAnn, body_pose::BodyPoseTypes pose_type)
{

  double dist = 0;
  switch((int)pose_type){
    case body_pose::FULL_BODY_J14:{
       dist = sqrt(pow((gtAnn.parts[3].x - gtAnn.parts[4].x), 2) +
                        pow((gtAnn.parts[3].y - gtAnn.parts[4].y), 2));
       break;
    }

    case body_pose::FULL_BODY_J13:{
      dist = sqrt(pow((gtAnn.parts[2].x - gtAnn.parts[3].x), 2) +
                  pow((gtAnn.parts[2].y - gtAnn.parts[3].y), 2));
    }
  }

  return dist;
}

double calAccuracyJLE(std::vector<Annotation>& groundTruth,
                       std::vector<Annotation>& estimated,
                       double errThresh,
                       body_pose::BodyPoseTypes pose_type,
                       EVAL_METRICS metric)
{

  double accuracy = 0;
  int trueDetections = 0;

  if(groundTruth.size() != estimated.size()){
    LOG(INFO)<<"Number of annotations are not equal in ground truth and estimated index files";
    exit(-1);
  }

  if(groundTruth.size() < 1 || estimated.size() < 1){
    LOG(INFO)<<"No annotation available";
    exit(-1);
  }

  std::vector<float> actionClassJLE(20);
  std::vector<float> actionClassTotal(20);

  int nAnn = groundTruth.size();
  int totalDetections = 0;
//
//  for(unsigned idx = 0; idx < 2180; idx++){
  for(unsigned idx = 0; idx < nAnn; idx++){
    Annotation gtAnn = groundTruth[idx];
    Annotation estAnn = estimated[idx];

    double upperBodySize = 0;
    switch(metric){
      case APK:
        upperBodySize = std::max(gtAnn.bbox.height, gtAnn.bbox.width);
        break;
      case JLE:
        upperBodySize = get_upperbody_size(gtAnn, pose_type);
        break;
      case PCKT:
        upperBodySize = get_shoulder_hip_dist(gtAnn, pose_type);
        break;
      default:
        LOG(INFO)<<"Unknown Error Metric Provided";
    };
    CHECK(upperBodySize);


    for(unsigned pIdx = 0; pIdx < gtAnn.parts.size(); pIdx++){

      if(gtAnn.parts[pIdx].x < 0 || gtAnn.parts[pIdx].y < 0){
        continue;
      }
      totalDetections++;
      actionClassTotal[gtAnn.cluster_id]++;

      double dist = sqrt(pow((gtAnn.parts[pIdx].x - estAnn.parts[pIdx].x), 2) +
                        pow((gtAnn.parts[pIdx].y - estAnn.parts[pIdx].y), 2));

      if(dist <= (errThresh * upperBodySize)){
        trueDetections++;
        actionClassJLE[gtAnn.cluster_id]++;
      }
    }
  }

  accuracy = (trueDetections / static_cast<double>(totalDetections))*100;
  LOG(INFO)<<"Accuracy = "<<accuracy;

//  LOG(INFO)<<"Class wise accuracies:";
//
//  for(unsigned int idx = 0; idx < actionClassJLE.size(); idx++){
//    LOG(INFO)<<"Class-"<<idx<<" = "<<(actionClassJLE[idx]/std::max(float(1), actionClassTotal[idx]))*100;
//  }
//
  return accuracy;
}

bool calAccuracyJLE_PartWise(std::vector<Annotation>& groundTruth,
                       std::vector<Annotation>& estimated,
                       double errThresh, std::vector<double>& accuracies,
                       body_pose::BodyPoseTypes pose_type,
                       EVAL_METRICS metric)
{

  double accuracy = 0;
  int trueDetections = 0;

  if(groundTruth.size() != estimated.size()){
    LOG(INFO)<<"Number of annotations are not equal in ground truth and estimated index files";
    exit(-1);
  }

  if(groundTruth.size() < 1 || estimated.size() < 1){
    LOG(INFO)<<"No annotation available";
    exit(-1);
  }


  for(unsigned pIdx = 0; pIdx < groundTruth[0].parts.size(); pIdx++){
    int trueDetections = 0;
    int totalDetections = 0;

    for(unsigned idx = 0; idx < groundTruth.size(); idx++){
      Annotation gtAnn = groundTruth[idx];
      Annotation estAnn = estimated[idx];

      if(gtAnn.parts[pIdx].x < 0 || gtAnn.parts[pIdx].y < 0){
        continue;
      }

      totalDetections++;

      double upperBodySize = 0;
      switch(metric){
        case APK:
          upperBodySize = std::max(gtAnn.bbox.height, gtAnn.bbox.width);
          break;
        case JLE:
          upperBodySize = get_upperbody_size(gtAnn, pose_type);
          break;
        case PCKT:
          upperBodySize = get_shoulder_hip_dist(gtAnn, pose_type);
          break;
        default:
          LOG(INFO)<<"Unknown Error Metric Provided";
      };

      CHECK(upperBodySize);

      double dist = sqrt(pow((gtAnn.parts[pIdx].x - estAnn.parts[pIdx].x), 2) +
                        pow((gtAnn.parts[pIdx].y - estAnn.parts[pIdx].y), 2));

      if(dist <= (errThresh * upperBodySize)){
        trueDetections++;
      }
    }
    accuracy = (trueDetections / static_cast<double>(totalDetections))*100;
    accuracies.push_back(accuracy);
    LOG(INFO)<<"Part id-"<<pIdx<<" = "<<accuracy;
  }


  return true;;
}

bool calAccuracy_error_ClassWise(std::vector<Annotation>& groundTruth,
                       std::vector<Annotation>& estimated,
                       std::vector<double>& errors,
                       body_pose::BodyPoseTypes pose_type)
{

  double accuracy = 0;
  int trueDetections = 0;

  if(groundTruth.size() != estimated.size()){
    LOG(INFO)<<"Number of annotations are not equal in ground truth and estimated index files";
    exit(-1);
  }

  if(groundTruth.size() < 1 || estimated.size() < 1){
    LOG(INFO)<<"No annotation available";
    exit(-1);
  }

  std::vector<int> actionClassTotal;
  errors.clear();

  for(unsigned pIdx = 0; pIdx < groundTruth[0].parts.size(); pIdx++){
  //for(unsigned pIdx = 0; pIdx < 238; pIdx++){
    int trueDetections = 0;
    int totalDetections = 0;

   double error = 0;
   for(unsigned idx = 0; idx < groundTruth.size(); idx++){
  // for(unsigned idx = 0; idx < 5000; idx++){

      Annotation gtAnn = groundTruth[idx];
      Annotation estAnn = estimated[idx];

      if(gtAnn.parts[pIdx].x < 0 || gtAnn.parts[pIdx].y < 0){
        continue;
      }

      totalDetections++;
      double upperBodySize = get_upperbody_size(gtAnn, pose_type);

//      LOG(INFO)<<"GT = " << gtAnn.parts[pIdx].x <<"\t"<<gtAnn.parts[pIdx].y;
//      LOG(INFO)<<"EST = " << estAnn.parts[pIdx].x<<"\t"<<estAnn.parts[pIdx].y;

      double dist = sqrt(pow((gtAnn.parts[pIdx].x - estAnn.parts[pIdx].x), 2) +
                        pow((gtAnn.parts[pIdx].y - estAnn.parts[pIdx].y), 2));

      error += dist;

      if(errors.size() < estAnn.cluster_id){
        errors.resize(estAnn.cluster_id, 0);
      }
      if(actionClassTotal.size() < estAnn.cluster_id){
        actionClassTotal.resize(estAnn.cluster_id, 0);
      }

      errors[estAnn.cluster_id-1] += dist;
      actionClassTotal[estAnn.cluster_id-1]++;
    }
  }

  for(unsigned int i=0; i<errors.size(); i++){
    errors[i] /= static_cast<float>(actionClassTotal[i]);
    LOG(INFO)<<"Class-"<<i<<" = "<<errors[i];
  }

  double error_avg = 0;
  for(unsigned int i=0; i<errors.size(); i++){
    error_avg += errors[i];
  }
  error_avg /= errors.size();
  LOG(INFO)<<"Average = "<<error_avg;

  return true;;
}

bool calAccuracy_error_PartWise(std::vector<Annotation>& groundTruth,
                       std::vector<Annotation>& estimated,
                       std::vector<double>& errors,
                       body_pose::BodyPoseTypes pose_type)
{

  double accuracy = 0;
  int trueDetections = 0;

//  if(groundTruth.size() != estimated.size()){
//    LOG(INFO)<<"Number of annotations are not equal in ground truth and estimated index files";
//    exit(-1);
//  }
//
//  if(groundTruth.size() < 1 || estimated.size() < 1){
//    LOG(INFO)<<"No annotation available";
//    exit(-1);
//  }


  for(unsigned pIdx = 0; pIdx < groundTruth[0].parts.size(); pIdx++){
  //for(unsigned pIdx = 0; pIdx < 238; pIdx++){
    int trueDetections = 0;
    int totalDetections = 0;

   double error = 0;
   for(unsigned idx = 0; idx < groundTruth.size(); idx++){
   //for(unsigned idx = 0; idx <510; idx++){

      Annotation gtAnn = groundTruth[idx];
      Annotation estAnn = estimated[idx];

      if(gtAnn.parts[pIdx].x < 0 || gtAnn.parts[pIdx].y < 0){
        continue;
      }

      totalDetections++;
      double upperBodySize = get_upperbody_size(gtAnn, pose_type);

//      LOG(INFO)<<"GT = " << gtAnn.parts[pIdx].x <<"\t"<<gtAnn.parts[pIdx].y;
//      LOG(INFO)<<"EST = " << estAnn.parts[pIdx].x<<"\t"<<estAnn.parts[pIdx].y;

      double dist = sqrt(pow((gtAnn.parts[pIdx].x - estAnn.parts[pIdx].x), 2) +
                        pow((gtAnn.parts[pIdx].y - estAnn.parts[pIdx].y), 2));

      error += dist;

    }
    errors.push_back(error / groundTruth.size());

    LOG(INFO)<<"Part id-"<<pIdx<<" = "<<errors[pIdx];
  }
  double error_avg = 0;
  for(unsigned int i=0; i<errors.size(); i++){
    error_avg += errors[i];
  }
  error_avg /= errors.size();
  LOG(INFO)<<"Average = "<<error_avg;

  return true;;
}


