/*
 * net.cpp
 *
 *  Created on: Feb 6, 2013
 *      Author: lbossard
 */

#include "net_utils.hpp"

#include <unistd.h>

#include <glog/logging.h>

namespace utils {
namespace net {

std::string get_host_name()
{

  char name[128];
  if (0 != gethostname(name, sizeof(name))){
    LOG(ERROR) << "gethostname() failed" << std::endl;
    return "";
  }
  return std::string(name);
}

} /* namespace net */
} /* namespace utils */
