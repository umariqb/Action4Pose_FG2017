/*
 * create_libsvm_features.cpp
 *
 *  Created on: Jan 25, 2012
 *      Author: lbossard
 */


#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem/path.hpp>
namespace fs = boost::filesystem;

#include <gflags/gflags.h>

#include "cpp/utils/file_utils.hpp"

#include "cpp/utils/libsvm/libsvm_common.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/vision/features/spmbow_extractor.hpp"
#include "cpp/vision/image_utils.hpp"

#include "segmentation/SegmentedObject.h"





DEFINE_string(segmented_object_dir, "", "path to images");
DEFINE_string(surf_voc_file, "", "surf input voc");
DEFINE_string(hog_voc_file, "", "hog input voc");
DEFINE_string(color_voc_file, "", "color input voc");
DEFINE_string(ssd_voc_file, "", "ssd input voc");
DEFINE_string(class_id, "1", "class +1 / -1");
DEFINE_string(output_file, "", "output svm data");
DEFINE_string(pool_function, "mpool", "pooling function: 'mpool', 'logistic', 'clipping'");
DEFINE_double(m_param, 0, "pooling/logistic parameter");
DEFINE_string(norm_method, "none", "'none', 'l1', 'l2'");
DEFINE_int32(feature_types, 7, "Feature types to extract: surf=1, hog=2, color=4, lbp=8");
DEFINE_int32(pyramid_levels, 3, "Levels of the spatial pyramid");
DEFINE_bool(use_segmented, true, "use only segmented image regions");

// x=0:100; figure; hold on; plot(x,x.^(1./1000),'xr'); plot(x,2*(1./(1+exp(-(x)))-.5),'b+');plot(x,1./(1. + exp(-(x - 10))),'co');
int main(int argc, char** argv)
{
    // get command line args
    google::ParseCommandLineFlags(&argc, &argv, true);

    const std::string segmented_object_dir     = FLAGS_segmented_object_dir;
    const std::string output_file   = FLAGS_output_file;
    const std::string surf_file     = FLAGS_surf_voc_file;
    const std::string hog_file      = FLAGS_hog_voc_file;
    const std::string color_file    = FLAGS_color_voc_file;
    const std::string ssd_file      = FLAGS_ssd_voc_file;
    const std::string class_id      = FLAGS_class_id;
    const double m_param            = FLAGS_m_param;
    const bool do_normalize         = (FLAGS_norm_method != "none");
    const std::string norm_function = FLAGS_norm_method;
    const std::string pool_function = FLAGS_pool_function;
    const vision::features::feature_type::T feature_types = static_cast<vision::features::feature_type::T>(FLAGS_feature_types);
    const bool use_segmented = FLAGS_use_segmented;
    const unsigned int pyramid_levels = FLAGS_pyramid_levels;

    if (segmented_object_dir == ""
            || output_file == ""
            || surf_file == ""
            || hog_file == ""
            || color_file == "")
    {
        google::ShowUsageWithFlags(argv[0]);
        return 1;
    }

    // get segmented_objects
    std::vector<fs::path> segmented_objects;
    utils::fs::collect_files(segmented_object_dir, ".*\\.segobj", std::back_inserter(segmented_objects));
    std::sort(segmented_objects.begin(), segmented_objects.end());
    const std::size_t object_count = segmented_objects.size();

    // load vocabularies
    cv::Mat surf_voc;
    utils::serialization::read_binary_archive(surf_file, surf_voc);
    cv::Mat hog_voc;
    utils::serialization::read_binary_archive(hog_file, hog_voc);
    cv::Mat color_voc;
    utils::serialization::read_binary_archive(color_file, color_voc);
    cv::Mat ssd_voc;
    if (!ssd_file.empty())
    {
        utils::serialization::read_binary_archive(ssd_file, ssd_voc);
    }
    vision::features::SpmBowExtractor spm_extractor(
            pyramid_levels,
            surf_voc,
            hog_voc,
            color_voc,
            ssd_voc
    );


    boost::progress_display show_progress( object_count );
    std::ofstream ofs(output_file.c_str());

    cv::Mat_<int> histograms;
    cv::Mat_<double> histograms_scaled;
    BOOST_FOREACH(const fs::path& segmented_obj_path, segmented_objects)
    {
        ++show_progress;

        // load segmented object
        fashion::SegmentedObject segmented_obj;
        utils::serialization::read_binary_archive(segmented_obj_path.string(), segmented_obj);
        const cv::Mat& image = segmented_obj.original_image();

        if (image.rows < 88 || image.cols < 88) // min dimension for ssd
        {
            continue;
        }

        const cv::Mat_<uchar>& mask = segmented_obj.mask();


        if (use_segmented)
        {
            spm_extractor.extractSpm(image, mask, histograms, feature_types);
        }
        else
        {
            spm_extractor.extractSpm(image, histograms, feature_types);
        }


        // normalize if necessary
        histograms_scaled = histograms;
        if (do_normalize)
        {
            if (norm_function == "l1")
            {
                histograms_scaled /= cv::norm(histograms_scaled, cv::NORM_L1);
            }
            else if (norm_function == "l2")
            {
                histograms_scaled /= cv::norm(histograms_scaled, cv::NORM_L2);
            }
            else
            {
                std::cerr << "uknown norm function" << std::endl;
                return -1;
            }
        }

        // write to text file
        if (!class_id.empty())
        {
            ofs << class_id << " ";
        }
        if (pool_function == "logistic")
        {
            utils::libsvm::write_libsvm_vector(
                    ofs,
                    histograms_scaled,
                    utils::libsvm::SomeLogisticFunction());
        }
        else if (pool_function == "SomeOtherLogisticFunction")
        {
            utils::libsvm::write_libsvm_vector(
                                ofs,
                                histograms_scaled,
                                utils::libsvm::SomeOtherLogisticFunction());
        }
        else if (pool_function == "clipping")
        {
            utils::libsvm::write_libsvm_vector(
                                ofs,
                                histograms_scaled,
                                utils::libsvm::ClippnScaleFunction(m_param));
        }
        else if (pool_function == "mpool")
        {
            if (m_param == 0)
            {
                utils::libsvm::write_libsvm_vector_max_pool(ofs, histograms_scaled);
            }
            else
            {
                utils::libsvm::write_libsvm_vector_mpool(ofs, histograms_scaled, m_param);
            }
        }
        else
        {
            std::cerr << "unknown pooling function. aborting" << std::endl;
            return 1;
        }
        ofs << std::endl;
    }
}
