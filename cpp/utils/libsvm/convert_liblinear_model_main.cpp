/*
 * convert_liblinear_model_main.cpp
 *
 * converts liblinear model from text format to our serialization format and
 * vice versa.
 *
 *  Created on: Feb 16, 2012
 *      Author: lbossard
 */




#include <gflags/gflags.h>

#include "liblinear.hpp"
#include "cpp/utils/file_utils.hpp"
#include "cpp/utils/serialization/serialization.hpp"


DEFINE_bool(txt2bin, true, "convert text to bin. otherwhise vice versa");
int main(int argc, char**argv)
{

    // get command line args
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (argc != 3)
    {
        std::cerr << argv[0] << "[options] src_model dst_model" << std::endl;
        return 1;
    }

    const std::string src_model_file = argv[1];
    const std::string dest_model_file = argv[2];
    const bool text_to_bin = FLAGS_txt2bin;

    if (text_to_bin)
    {
        // read model
        utils::liblinear::LinearHolder linear_model;
        if (!linear_model.read_from_text(src_model_file))
        {
            std::cerr << "could not load model from " << src_model_file << std::endl;
            return 1;
        }
        // serialize it
        if (! utils::serialization::write_binary_archive(dest_model_file, linear_model))
        {
            std::cerr << "could not save model to " << dest_model_file << std::endl;
            return 1;
        }
    }
    else
    {
        utils::liblinear::LinearHolder linear_model;
        // deserialize it
        if (! utils::serialization::read_binary_archive(src_model_file, linear_model))
        {
            std::cerr << "could not load model from " << src_model_file << std::endl;
            return 1;
        }

        // save it as text
        if (! linear_model.write_as_text(dest_model_file))
        {
            std::cerr << "could not save model to " << dest_model_file << std::endl;
            return 1;
        }
    }


    return 0;
}
