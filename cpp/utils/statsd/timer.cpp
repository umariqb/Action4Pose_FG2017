/*
 * statsd_timer.cpp
 *
 *  Created on: Feb 18, 2014
 *      Author: lbossard
 */

#include "timer.hpp"

#include <glog/logging.h>

namespace awesomeness {
namespace utils {
namespace statsd {


Timer::Timer(Client& client, const std::string& bucket)
: _client(client),
  _bucket(bucket)
{

}
Timer::~Timer(){

  int time_millisecs = _timer.elapsed().user / 1000000;
  try {
    _client.timing(_bucket, time_millisecs);
  }
  catch (...){
    LOG(ERROR) << "error while with timer";
  }

}

} /* namespace statsd */
} /* namespace utils */
} /* namespace awesomeness */
