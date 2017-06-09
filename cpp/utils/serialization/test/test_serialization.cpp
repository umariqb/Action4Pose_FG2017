/*
 * test_serialization.cpp
 *
 *  Created on: Aug 19, 2013
 *      Author: lbossard
 */

#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/third_party/gtest/gtest.h"

#include <boost/filesystem.hpp>


class EphemeralTmpfile {
public:
  EphemeralTmpfile(){
  }
  ~EphemeralTmpfile(){
    if (!_path.empty()){
      boost::system::error_code ec;
      boost::filesystem::remove(_path, ec);
      if (ec) {
        LOG(ERROR) << "error while trying to delete tmpfile: " << ec.value() << " " << ec.message();
      }
    }
  }

  boost::filesystem::path get(){
    if (_path.empty()){
      _path = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
    }
    return _path;

  }

private:
  boost::filesystem::path _path;
};

TEST(SerializationTest, RWSimple)
{
  std::string t = "abcd123";

  for (utils::serialization::Compression::T compression = utils::serialization::Compression::None;
      compression < utils::serialization::Compression::_unused;
      compression = (utils::serialization::Compression::T)((int)compression + 1)
  )
  {
    EphemeralTmpfile f;
    ASSERT_TRUE(utils::serialization::write_binary_archive(f.get().string(), t, compression));

    std::string read_t;
    ASSERT_TRUE(utils::serialization::read_binary_archive(f.get().string(), read_t, compression)) <<
        "failed:" <<  utils::serialization::Compression::to_string(compression);
    ASSERT_EQ(t, read_t);
  }
}
