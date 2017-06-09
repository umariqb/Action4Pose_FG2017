/*
 * statsd_timer.hpp
 *
 *  Created on: Feb 18, 2014
 *      Author: lbossard
 */

#ifndef AWESOMENESS__UTILS__STATSD__TIMER_HPP_
#define AWESOMENESS__UTILS__STATSD__TIMER_HPP_

#include <boost/timer/timer.hpp>

#include "client.hpp"

namespace awesomeness {
namespace utils {
namespace statsd {

class Timer {
public:

  Timer(Client& client, const std::string& bucket);
  ~Timer();



private:
  Client& _client;
  const std::string& _bucket;
  boost::timer::cpu_timer _timer;

  Timer(const Timer&);
  Timer();

};


} /* namespace statsd */
} /* namespace utils */
} /* namespace awesomeness */
#endif /* AWESOMENESS__UTILS__STATSD__TIMER_HPP_ */
