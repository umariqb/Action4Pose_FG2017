#ifndef PERFORMANCE_METRICS_HH
#define PERFORMANCE_METRICS_HH

#include<vector>
#include<cpp/body_pose/common.hpp>
#include <glog/logging.h>


enum EVAL_METRICS {JLE=1, APK=2, PCKT=3, PCKH=4};

// computes the joint localization error
double calAccuracyJLE(std::vector<Annotation>& groundTruth,
                    std::vector<Annotation>& estimated,
                    double errThresh,
                    body_pose::BodyPoseTypes pose_type,
                    EVAL_METRICS metric);

bool calAccuracyJLE_PartWise(std::vector<Annotation>& groundTruth,
                       std::vector<Annotation>& estimated,
                       double errThresh, std::vector<double>& accuracies,
                       body_pose::BodyPoseTypes pose_type,
                       EVAL_METRICS metric);

bool calAccuracy_error_PartWise(std::vector<Annotation>& groundTruth,
                       std::vector<Annotation>& estimated,
                       std::vector<double>& errors,
                       body_pose::BodyPoseTypes pose_type);

bool calAccuracy_error_ClassWise(std::vector<Annotation>& groundTruth,
                       std::vector<Annotation>& estimated,
                       std::vector<double>& errors,
                       body_pose::BodyPoseTypes pose_type);


#endif
