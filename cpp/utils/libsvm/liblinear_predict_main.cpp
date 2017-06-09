/*
 * liblinear_predict_main.cpp
 *
 *  Created on: Feb 6, 2012
 *      Author: lbossard
 */

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <gflags/gflags.h>

#include "liblinear.hpp"

int main(int argc, char**argv)
{

    // get command line args
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (argc != 4)
    {
        std::cerr << argv[0] << "[options] test_file model_file output_file" << std::endl;
        return 1;
    }

    std::string test_file = argv[1];
    std::string model_file = argv[2];
    std::string output_file = argv[3];

    // read model
    utils::liblinear::LinearHolder linear_model;
    if (!linear_model.read_from_text(model_file))
    {
        std::cerr << "could not load model from " << model_file << std::endl;
        return 1;
    }

    // read test file
    utils::liblinear::ProblemHolder test_data;
    {
        std::ifstream is(test_file.c_str());
        if (!is || !test_data.read(is))
        {
            std::cerr << "could not read problem from  " << test_file << std::endl;
            return 1;
        }
    }

    // classify
    unsigned int correct_results = 0;
    {
        std::ofstream os(output_file.c_str());
        if (!os)
        {
            std::cerr << "could not open output_file  " << output_file << std::endl;
            return 1;
        }
        std::vector<double> values;

        typedef utils::liblinear::FeatureNode FeatureNode;
        for (unsigned int idx = 0; idx < test_data.labels().size(); ++idx)
        {
            FeatureNode* f = test_data.features()[idx];
            const int label = test_data.labels()[idx];

            //predict
            values.clear();
            int predicted_class_id = linear_model.predict(f, values);

            if (predicted_class_id == label)
            {
               correct_results += 1;
            }

            os << predicted_class_id << " ";
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                os << values[i];
                if (i < values.size() - 1)
                {
                    os << " ";
                }
            }
            os << "\n";
        }
        os.close();
    }

    std::cout << "Accuracy " << (static_cast<float>(correct_results) / test_data.labels().size()*100)
            << "% (" << correct_results << "/" << test_data.labels().size()
            << ")" << std::endl;

    return 0;
}
