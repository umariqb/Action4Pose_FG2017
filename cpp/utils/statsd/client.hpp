/*
 * statsd_client.hpp
 *
 *  Created on: Feb 14, 2014
 *      Author: lbossard
 */

#ifndef AWESOMENESS__UTILS__STATSD__CLIENT_HPP_
#define AWESOMENESS__UTILS__STATSD__CLIENT_HPP_

#include <string>
#include <vector>

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/udp.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

namespace awesomeness {
namespace utils {
namespace statsd {


class Client {
public:
  explicit Client(const std::string& prefix="");
  Client(const std::string& hostname, int port, const std::string& prefix="");

  virtual ~Client();

  void set_prefix(const std::string& prefix);
  inline std::string prefix() const;

  void connect(const std::string& hostname, int port);

  void gauge(const std::string& bucket, int value, bool delta=false, float sample_rate=1);
  void set(const std::string& bucket, int value, float sample_rate=1);
  void timing(const std::string& bucket, int timing, float sampleRate=1);
  void increment(const std::string& bucket, int delta=1, float sampleRate=1);
  void decrement(const std::string& bucket, int delta=1, float sampleRate=1);


protected:
  void send_message(
      const std::string& bucket,
      const std::string& value,
      const std::string& msg_type,
      float sample_rate = 1);


private:

  Client(const Client& c);

  boost::asio::io_service _io_service;
  boost::asio::ip::udp::endpoint _endpoint;
  boost::asio::ip::udp::socket _socket;

  std::string _prefix;

  typedef boost::mt19937 RandomGenerator;
  RandomGenerator _rng;
  boost::random::uniform_real_distribution<> _rand;
};

////////////////////////////////////////////////////////////////////////////////
/*inline*/ std::string Client::prefix() const {
  return _prefix;
}
} /* namespace statsd */
} /* namespace utils */
} /* namespace awesomeness */
#endif /* AWESOMENESS__UTILS__STATSD__CLIENT_HPP_ */
