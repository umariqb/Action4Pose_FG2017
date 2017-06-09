/*
 * test_statsd.cpp
 *
 *  Created on: 17.02.2014
 *      Author: luk
 */


#include <boost/thread.hpp>
#include "cpp/third_party/gtest/gtest.h"
#include "client.hpp"
#include "timer.hpp"

TEST(statsd_client, test_simple_dns) {

  awesomeness::utils::statsd::Client c("localhost", 8125);
  c.increment("foo");
  c.decrement("foo");
  c.gauge("fooo", 12);

}

TEST(statsd_client, test_simple) {

  awesomeness::utils::statsd::Client c("127.0.0.1", 8125);
  c.decrement("bar");
}

TEST(statsd_client, test_timer) {

  awesomeness::utils::statsd::Client c("127.0.0.1", 8125);
  {
    awesomeness::utils::statsd::Timer(c, "test.timer_test");
    boost::this_thread::sleep(boost::posix_time::milliseconds(10));
  }
}


