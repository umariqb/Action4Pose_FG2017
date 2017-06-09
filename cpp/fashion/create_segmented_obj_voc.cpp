/*
 * create_voc.cpp
 *
 *  Created on: Oct 11, 2011
 *      Author: lbossard
 */
#define BOOST_FILESYSTEM_NO_DEPRECATED 1

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <tr1/unordered_set>

#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
namespace fs=boost::filesystem;

#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/timer.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "cpp/vision/features/low_level_features.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/utils/file_utils.hpp"

#include "segmentation/SegmentedObject.h"

/**
 * http://c-faq.com/lib/randrange.html
 * @return returns an usinged int in 0 <= ui < max_i
 */
inline unsigned int randrange(int max_i)
{
    return static_cast<unsigned int>(
                    static_cast<double>(rand())
                    / (static_cast<double>(RAND_MAX) + 1)
                    * (max_i));
}

void get_rand_indexes(std::size_t index_count, std::size_t element_count, std::vector<std::size_t>& indexes)
{
    // sanity checks
    indexes.clear();
    if (element_count < index_count)
    {
        std::cerr << "max_idx needs to be greater than index_count and greater 0" << std::endl;
        return;
    }

    // lets generate 0...max_idx if necessary
    static std::vector<std::size_t> range(static_cast<std::size_t>(1), 0);
    if (element_count > range.size())
    {
        std::size_t old_size = range.size();
        range.resize(element_count, 1);
        std::partial_sum(
                range.begin() + old_size - 1,
                range.end(),
                range.begin() + old_size - 1);
    }
    if (element_count == index_count)
    {
        indexes.insert(indexes.begin(), range.begin(), range.begin() + element_count);
        return;
    }

    //http://en.wikipedia.org/wiki/Fisher–Yates_shuffle#The_.22inside-out.22_algorithm
    indexes.resize(element_count);
    indexes[0] = range[0];
    for (std::size_t i = 1; i < element_count; ++i)
    {
        const std::size_t j = randrange(i+1);
        indexes[i] = indexes[j];
        indexes[j] = range[i];
    }
    indexes.resize(index_count);
}


cv::Mat_<float> trainVoc(
        const std::vector<fs::path>& segmented_obj_paths,
        const vision::features::LowLevelFeatureExtractor& extractor,
        const std::size_t vocab_size,
        const std::size_t max_desc_per_img,
        bool use_segmented)
{

    const unsigned int descriptor_dimension = extractor.descriptorLength();

    // create kmeans object
    cv::TermCriteria term_criteria;
    term_criteria.epsilon = 1;
    term_criteria.maxCount = 10;
    term_criteria.type = cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS;
    cv::BOWKMeansTrainer bowTrainer( vocab_size, term_criteria, 3, cv::KMEANS_PP_CENTERS );

    // process each image
    std::cout << "Loading " << segmented_obj_paths.size() << " segmented objects ...." << std::flush;
    cv::Mat_<float> all_descriptors(0, descriptor_dimension);
    {
        //    typedef std::tr1::unordered_set<std::size_t> set_t;
        typedef std::vector<std::size_t> set_t;
        set_t index_set;
        std::vector<cv::Point> locations;
        BOOST_FOREACH(const fs::path& segmented_obj_path, segmented_obj_paths)
        {
            // read segmented object
            fashion::SegmentedObject segmented_obj;
            utils::serialization::read_binary_archive(segmented_obj_path.string(), segmented_obj);
            const cv::Mat& image = segmented_obj.original_image();
            const cv::Mat_<uchar>& mask = segmented_obj.mask();

            locations.clear();
            cv::Mat_<float> descriptors;
            if (use_segmented)
            {
                descriptors = extractor.denseExtract(image, locations, mask);
            }
            else
            {
                descriptors = extractor.denseExtract(image, locations);
            }
            const std::size_t num_descriptors = descriptors.rows;
            if (!descriptors.data || num_descriptors == 0)
            {
                std::cerr << "could not extract any features from  " << segmented_obj_path.string() << std::endl;
                continue;
            }

            const std::size_t descriptors_to_extract = std::min(max_desc_per_img, num_descriptors);

            get_rand_indexes(descriptors_to_extract, num_descriptors, index_set);

            for (set_t::const_iterator it = index_set.begin(); it != index_set.end(); ++it)
            {
                all_descriptors.push_back(descriptors.row(*it));
//                std::cout << descriptors.row(*it) << std::endl;
            }
        }

    }
    std::cout << "...done" << std::endl;

    if (all_descriptors.rows < vocab_size * 100)
    {
        std::cerr << "extracted not enough features:" << all_descriptors.rows  << std::endl;
        return cv::Mat_<float>();
    }

    // train voc
    std::cout << "Training vocabulary with k=" << vocab_size << " and "<< all_descriptors.rows << " features...";
    std::cout.flush();
    cv::Mat_<float> vocabulary = bowTrainer.cluster(all_descriptors);
    std::cout << "...done" << std::endl;
    return vocabulary;
}

DEFINE_string(segmented_obj_dir,
//        "",
        "/home/lbossard/scratch/datasets/google-fashion/attributes_segmented/unsupervised/queries/general/colored",
        "path to segmented objects");
DEFINE_string(vocabulary_file,
//        "",
        "/tmp/foo",
        "outputfile");
DEFINE_string(feature_type,
//        "",
        "ssd",
        "surf hog ssd colorssd");
DEFINE_int32(voc_size, 1024, "voc size");
DEFINE_int32(max_files, 10, "files to get");
DEFINE_bool(use_segmented, true, "use only segmented image regions");

int main(int argc, char** argv)
{
    // get command line args
    google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string image_dir    = FLAGS_segmented_obj_dir;
    const std::string voc_file     = FLAGS_vocabulary_file;
    const std::string feature_typestr = FLAGS_feature_type;
    const unsigned int vocab_size  = FLAGS_voc_size;
    const std::size_t max_files = FLAGS_max_files;
    const bool use_segmented = FLAGS_use_segmented;
    if (image_dir.empty() || voc_file.empty() || feature_typestr.empty())
    {
        google::ShowUsageWithFlags(argv[0]);
        return 1;
    }

    // collect files
    std::vector<fs::path> segmented_objects;
    segmented_objects.reserve(max_files);
    utils::fs::collect_rand_files(image_dir, max_files, ".*\\.segobj", std::back_inserter(segmented_objects));

    // create feature extractor
    const std::size_t max_desc_per_img = 4096;
    vision::features::feature_type::T feature_type = vision::features::feature_type::from_string(feature_typestr);
    if (feature_type == vision::features::feature_type::Lbp)
    {
        std::cerr << "lbp does not need a vocabulary" << std::endl;
        return 1;
    }
    boost::scoped_ptr<vision::features::LowLevelFeatureExtractor> extractor(
            vision::features::ExtractorFactory::createExtractor(feature_type));
    if (feature_type == vision::features::feature_type::Ssd
            || feature_type == vision::features::feature_type::ColorSsd)
    {
        dynamic_cast<vision::features::SelfSimilarityExtractor*>(extractor.get())
                ->setSelfSimilarityParameters(5, 20, 3, 10,25*36*3,1);
    }

    std::cout.flush();
    cv::Mat_<float> vocabulary = trainVoc(
            segmented_objects,
            *extractor,
            vocab_size,
            max_desc_per_img,
            use_segmented);

    if (!vocabulary.data)
    {
        return 1;
    }

    // save vocabulary
    {
        std::ofstream ofs(voc_file.c_str());
        boost::archive::binary_oarchive oa(ofs);
        oa << vocabulary;
    }
    std::cout << "Finished" << std::endl;
}
