/*
 * datas_set.cpp
 *
 *  Created on: Jan 10, 2014
 *      Author: mdantone
 */

#include "data_set.hpp"
#include <glog/logging.h>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/algorithm/string/replace.hpp>
namespace fs = boost::filesystem;

#include "cpp/third_party/json_spirit/json_spirit.h"
namespace j = json_spirit;


namespace learning {
namespace common {

int parse_data_set(const std::string& data_set_json_file, DataSet* data_set_){
  DataSet& data_set = *data_set_;
  data_set.clear();

  // load info from json
  j::Value value;
  {
    std::ifstream ifs(data_set_json_file.c_str());
    CHECK(ifs);
    CHECK(j::read(ifs, value)) << "Could not parse jsonfile";
  }
  // parse json to our datastructure
  const j::Object& dict = value.get_obj();

  int sample_count = 0;
  BOOST_FOREACH(const j::Pair& class_entry, dict){
    const std::string class_name = class_entry.name_;
    DataSet::mapped_type& img_ids = data_set[class_name];

    const j::Array& json_img_ids = class_entry.value_.get_array();
    img_ids.resize(json_img_ids.size());
    for (std::size_t i = 0; i < img_ids.size(); ++i) {
      img_ids[i] = json_img_ids[i].get_str();
//      img_ids[i] = img_ids[i].replace_extension("");
      ++sample_count;
    }
  }
  return sample_count;
}


} /* namespace common */
} /* namespace learning */
