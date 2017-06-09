#include "cpp/learning/forest/votmap_max_coords.hpp"
#include <fstream>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <opencv2/opencv.hpp>
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include <glog/logging.h>



namespace learning {
namespace forest {

bool save_votmap_max_coords(VotMapMaxCoords& coords, std::string& fname){
  try{
    std::ofstream ofs(fname.c_str());
    if(ofs==0){
    LOG(INFO)<<"Error: Cannot open the given path to save voting map maximas."<<fname;
    return false;
    }
    boost::archive::text_oarchive oa(ofs);
    oa<<coords;
    ofs.flush();
    ofs.close();
    LOG(INFO)<<"Voting map maximas saved at :"<<fname;
    return true;
  }
  catch(boost::archive::archive_exception& ex){
    LOG(INFO)<<"Archive exception during deserialization:" <<std::endl;
    LOG(INFO)<< ex.what() << std::endl;
    LOG(INFO)<< "it was file: "<<fname;
  }
  return true;
}

bool save_votmap_max_coords(std::vector<VotMapMaxCoords>& coords, std::string& fname)
{
  try{
    std::ofstream ofs(fname.c_str());
    if(ofs==0){
    LOG(INFO)<<"Error: Cannot open the given path to save voting map maximas:"<<fname;
    return false;
    }
    boost::archive::text_oarchive oa(ofs);
    oa<<coords;
    ofs.flush();
    ofs.close();
    LOG(INFO)<<"Voting map maximas saved at :"<<fname;
    return true;
  }
  catch(boost::archive::archive_exception& ex){
    LOG(INFO)<<"Archive exception during deserialization:" <<std::endl;
    LOG(INFO)<< ex.what() << std::endl;
    LOG(INFO)<< "it was file: "<<fname;
  }
  return true;
}


bool load_votmax_max_coords(std::string& fname, VotMapMaxCoords& coords){
  std::ifstream ifs(fname.c_str());

  if(!ifs){
    LOG(INFO)<<"file not found.";
  }
  else{
    try{
      boost::archive::text_iarchive ia(ifs);
      ia>>coords;
      LOG(INFO)<<"Voting map maximas loaded";
      return true;
    }
    catch(boost::archive::archive_exception& ex){
      LOG(INFO)<<"Reload Tree: Archive exception during deserializiation: "
            <<ex.what();
      LOG(INFO)<<"not able to load voting map maximas from "<<fname;
    }
  }
  return false;
}

bool load_votmax_max_coords(std::string& fname, std::vector<VotMapMaxCoords>& coords){
  std::ifstream ifs(fname.c_str());
  if(!ifs){
    LOG(INFO)<<"file not found.";
  }
  else{
    try{
      boost::archive::text_iarchive ia(ifs);
      ia>>coords;
      LOG(INFO)<<"Voting map maximas loaded";
      return true;
    }
    catch(boost::archive::archive_exception& ex){
      LOG(INFO)<<"Reload Tree: Archive exception during deserializiation: "
            <<ex.what();
      LOG(INFO)<<"not able to load voting map maximas from "<<fname;
    }
  }
  return false;

}

bool save_class_wise_votmaps(std::vector<std::vector<std::vector<cv::Mat> > >& votmaps, std::string& fname){
  try{
    std::ofstream ofs(fname.c_str());
    if(ofs==0){
    LOG(INFO)<<"Error: Cannot open the given path to save voting map."<<fname;
    return false;
    }
    boost::archive::text_oarchive oa(ofs);
    oa<<votmaps;
    ofs.flush();
    ofs.close();
    LOG(INFO)<<"Voting map maximas saved at :"<<fname;
    return true;
  }
  catch(boost::archive::archive_exception& ex){
    LOG(INFO)<<"Archive exception during deserialization:" <<std::endl;
    LOG(INFO)<< ex.what() << std::endl;
    LOG(INFO)<< "it was file: "<<fname;
  }
  return true;
}

bool load_class_wise_votmaps(std::string& fname, std::vector<std::vector<std::vector<cv::Mat> > >& votmaps){
  std::ifstream ifs(fname.c_str());
  if(!ifs){
    LOG(INFO)<<"file not found.";
  }
  else{
    try{
      boost::archive::text_iarchive ia(ifs);
      ia>>votmaps;
      LOG(INFO)<<"Voting maps loaded";
      return true;
    }
    catch(boost::archive::archive_exception& ex){
      LOG(INFO)<<"Reload Tree: Archive exception during deserializiation: "
            <<ex.what();
      LOG(INFO)<<"not able to load voting maps from "<<fname;
    }
  }
  return false;

}


} // namespace learning
} // namespace forest
