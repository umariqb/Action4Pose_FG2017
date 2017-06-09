#include "cpp/vision/optical_flow_utils.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/utils/system_utils.hpp"
#include "cpp/utils/thread_pool.hpp"
#include <cpp/third_party/ldof/ldof.h>
#include <cpp/third_party/ldof/COpticFlowPart.h>

namespace vision {

bool load_optical_flow(std::string path, std::vector<cv::Mat>& optical_flow)
{
  std::ifstream ifs(path.c_str());

  if(!ifs){
    LOG(INFO)<<"file not found.";
  }
  else{
    try{
      boost::archive::text_iarchive ia(ifs);
      ia>>optical_flow;
      LOG(INFO)<<"Optical flow loaded";
      return true;
    }
    catch(boost::archive::archive_exception& ex){
      LOG(INFO)<<"Reload Tree: Archive exception during deserializiation: "
            <<ex.what();
      LOG(INFO)<<"not able to load optical flow from "<<path;
    }
  }
  return false;
}

bool save_optical_flow(std::vector<cv::Mat>& optical_flow, std::string save_path)
{
  try{
    std::ofstream ofs(save_path.c_str());
    if(ofs==0){
      LOG(INFO)<<"Error: Cannot open the given path to save Optical Flow";
      return false;
    }
    boost::archive::text_oarchive oa(ofs);
    oa<<optical_flow;
    ofs.flush();
    ofs.close();
    LOG(INFO)<<"Optical flow saved at :"<<save_path;
    return true;
  }
  catch(boost::archive::archive_exception& ex){
    LOG(INFO)<<"Archive exception during deserialization:" <<std::endl;
    LOG(INFO)<< ex.what() << std::endl;
    LOG(INFO)<< "it was file: "<<save_path;
  }
  return true;

}


void compute_optical_flow(std::vector<cv::Mat>& frames, std::vector<cv::Mat>& optical_flow, std::string save_path)
{
  optical_flow.resize(frames.size()-1);
  for(unsigned int i=0; i<frames.size()-1; i++){

//    unsigned char *f1 = (unsigned char*)(frames[i].data);
//    unsigned char *f2 = (unsigned char*)(frames[i+1].data);
//    CTensor<float> img1(frames[i].cols, frames[i].rows,3,f1);
//    CTensor<float> img2(frames[i+1].cols, frames[i+1].rows,3,f2);

    //calcOpticalFlowFarneback(cur_frame, next_frame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    cv::calcOpticalFlowSF(frames[i], frames[i+1], optical_flow[i], 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
  }

  if(!(save_path=="")){
    save_optical_flow(optical_flow, save_path);
  }
}

} /* namespace vision*/
