/*
 * datas_set.hpp
 *
 *  Created on: Jan 10, 2014
 *      Author: mdantone
 */

#ifndef DATAS_SET_HPP_
#define DATAS_SET_HPP_

#include <boost/filesystem/path.hpp>
#include <map>
#include <string>

namespace learning {
namespace common {


  // TODO change it from string to template
  typedef std::map<std::string, std::vector<boost::filesystem::path> > DataSet;


  // parse the dataset and returns the number of items in the dataset
  int parse_data_set(const std::string& data_set_json_file,
                     DataSet* data_set);


} /* namespace common */
} /* namespace learning */
#endif /* DATAS_SET_HPP_ */
