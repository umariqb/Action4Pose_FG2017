/*
 * client.cpp
 *
 *  Created on: Feb 14, 2014
 *      Author: lbossard
 */

#include "client.hpp"

#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <glog/logging.h>

#include <boost/timer/timer.hpp>

namespace awesomeness {
namespace utils {
namespace statsd {

Client::Client(const std::string& prefix)
:  _socket(_io_service),
   _prefix(prefix),
   _rand(0,1)
{

}

Client::Client(const std::string& hostname, int port, const std::string& prefix)
:  _socket(_io_service),
   _prefix(prefix),
   _rand(0, 1)
{

  try {
    connect(hostname, port);
  }
  catch (...){
    LOG(ERROR) << "StatsdClient can't connect to host " << hostname;
  }
}

Client::~Client() {
  boost::system::error_code err;
  _socket.close(err);
  if (err){
    LOG(ERROR) << "error while closing socket" << err;
  }
}

void Client::set_prefix(const std::string& prefix){
  _prefix = prefix;
}


void Client::connect(const std::string& hostname, int port){

  // if we're already connected: close connection first
  if (_socket.is_open()){
    LOG(INFO) << "closing open socket before connecting";
    _socket.close();
  }
  // resolve endpoint
  boost::asio::ip::udp::resolver resolver(_io_service);
  boost::asio::ip::udp::resolver::query query(
      boost::asio::ip::udp::v4(),
      hostname,
      boost::lexical_cast<std::string>(port)
  );
  _endpoint = *resolver.resolve(query);
  _socket.open(boost::asio::ip::udp::v4());
}

void Client::send_message(
    const std::string& bucket,
    const std::string& value,
    const std::string& msg_type,
    float sample_rate){


  // do the sampling
  if (sample_rate < 1.){
    CHECK(sample_rate > 0 && sample_rate < 1) << "sample_rate needs to be between 0 and 1";
    if (_rand(_rng) > sample_rate){
      return;
    }
  }

  // assemble message
  // <prefix>.<bucket>:<value>@<sample_rate>|<msg_type_id>
  std::string message;
  message.reserve(_prefix.size() + bucket.size() + value.size() + 10);

  if (_prefix.size() > 0){
    message.append(_prefix);
    message.append(".");
  }
  message.append(bucket);
  message.append(":");
  message.append(value);
  message.append("|");
  message.append(msg_type);
  if (sample_rate < 1){
    message.append("@");
    message.append(boost::lexical_cast<std::string>(sample_rate));
  }

  // finally send it to the server
  _socket.send_to(
        boost::asio::buffer(message.data(), message.size()),
        _endpoint);
}


void Client::gauge(const std::string& bucket, int value, bool delta, float sample_rate){
  if (!delta && value < 0){
    if (sample_rate < 1 && _rand(_rng) > sample_rate){
      return;
    }
    send_message(bucket, "0", "g");
    send_message(bucket, boost::lexical_cast<std::string>(value), "g");
  }
  else if (delta && value >= 0){
    send_message(bucket, "+" + boost::lexical_cast<std::string>(value), "g", sample_rate);
  }
  else {
    send_message(bucket, boost::lexical_cast<std::string>(value), "g", sample_rate);
  }
}

void Client::set(const std::string& bucket, int value, float sample_rate){
  send_message(bucket, boost::lexical_cast<std::string>(value), "s", sample_rate);
}

void Client::timing(const std::string& bucket, int timing, float sample_rate){
  send_message(bucket, boost::lexical_cast<std::string>(timing), "ms", sample_rate);
}

void Client::increment(const std::string& bucket, int delta, float sample_rate){
  send_message(bucket, boost::lexical_cast<std::string>(-1*delta), "c", sample_rate);
}

void Client::decrement(const std::string& bucket, int delta, float sample_rate){
  send_message(bucket, boost::lexical_cast<std::string>(delta), "c", sample_rate);
}

} /* namespace statsd */
} /* namespace utils */
} /* namespace awesomeness */
