/*
 * bow_extractor_main.cpp
 *
 *  Created on: Feb 8, 2012
 *      Author: lbossard
 */

#include <vector>

#include <gflags/gflags.h>

#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem/path.hpp>
namespace fs = boost::filesystem;

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cpp/utils/file_utils.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/vision/features/low_level_features.hpp"
#include "cpp/vision/features/bow_extractor.hpp"
#include "cpp/vision/features/bow_histogram.hpp"
#include "cpp/vision/image_transform.hpp"
namespace vf = vision::features;

#include "segmentation/SegmentedObject.h"


DEFINE_string(segmented_object_dir, "", "path to segmented objects");
DEFINE_string(feature_type, "", "Feature types to extract: surf, hog, color, lbp, ssd, colorssd");
DEFINE_string(voc_file, "", "input voc");
DEFINE_string(output_dir, "", "directory to save the features");

DEFINE_int32(pyramid_levels, 3, "Levels of the spatial pyramid");
DEFINE_bool(use_segmented, true, "use only segmented image regions");

DEFINE_bool(on_original_image, true, "yaya");
DEFINE_bool(do_transformations, false, "flipnstuff?");

int main(int argc, char **argv)
{
    // get command line args
    google::ParseCommandLineFlags(&argc, &argv, true);

    const std::string segmented_object_dir  = FLAGS_segmented_object_dir;
    const fs::path output_dir               = FLAGS_output_dir;
    const vf::feature_type::T feature_type  = vf::feature_type::from_string(FLAGS_feature_type);
    const std::string voc_file              = FLAGS_voc_file;
    const bool use_segmented                = FLAGS_use_segmented;
    const unsigned int pyramid_levels       = FLAGS_pyramid_levels;

    boost::ptr_vector<vision::image_transform::ImageTransform> transformations;
    if (FLAGS_on_original_image)
    {
        transformations.push_back(new vision::image_transform::Identity());
    }
    if (FLAGS_do_transformations)
    {
        transformations.push_back(new vision::image_transform::CenterRotate(-20));
        transformations.push_back(new vision::image_transform::CenterRotate(-10));
        transformations.push_back(new vision::image_transform::CenterRotate(+10));
        transformations.push_back(new vision::image_transform::CenterRotate(+20));
        transformations.push_back(new vision::image_transform::Flipper());
    }

//    const std::string pool_function = FLAGS_pool_function;


    if (segmented_object_dir.empty()
                || output_dir.empty()
                || (voc_file.empty() && feature_type != vf::feature_type::Lbp)
                || feature_type == vf::feature_type::None
                )
    {
        google::ShowUsageWithFlags(argv[0]);
        return 1;
    }

    const std::string feature_extension = "." + vf::feature_type::to_string(feature_type);

    // get segmented_objects
    std::vector<fs::path> segmented_objects;
    utils::fs::collect_files(segmented_object_dir, ".*\\.segobj", std::back_inserter(segmented_objects));
    std::sort(segmented_objects.begin(), segmented_objects.end());
    const std::size_t object_count = segmented_objects.size();

    // create extractor
    boost::scoped_ptr<vf::LowLevelFeatureExtractor> extractor(vf::ExtractorFactory::createExtractor(feature_type));

    // creat bow assigner
    cv::Mat bow_voc;
    boost::scoped_ptr<vf::BowExtractor> bow_extractor;
    std::size_t bow_count = 0;
    if (feature_type != vf::feature_type::Lbp)
    {
         // load vocabularies
        utils::serialization::read_binary_archive(voc_file, bow_voc);
        // construct assigner
        bow_extractor.reset(new vf::BowExtractor(bow_voc));
        bow_count = bow_extractor->wordCount();
    }
    else
    {
        dynamic_cast<vf::Lbp*>(extractor.get())->maxWordId();
        bow_count = dynamic_cast<vf::Lbp*>(extractor.get())->wordCount();
    }

    std::cout << "processing " << object_count << " files" << std::endl;
    boost::progress_display show_progress(object_count );

    cv::Mat_<int> histogram;
    static std::vector<cv::Point> locations;
    static std::vector<vf::BowExtractor::WordId> words;
    cv::Mat_<uchar> mask_base;
    BOOST_FOREACH(const fs::path& segmented_obj_path, segmented_objects)
    {
        // load segmented object
        fashion::SegmentedObject segmented_obj;
        utils::serialization::read_binary_archive(segmented_obj_path.string(), segmented_obj);
        const cv::Mat image_base = segmented_obj.original_image();

        if (use_segmented)
        {
            mask_base = segmented_obj.mask();
        }
        else
        {
            mask_base = cv::Mat_<uchar>();
        }

        BOOST_FOREACH(const vision::image_transform::ImageTransform& transformer, transformations)
        {

            cv::Mat image = image_base.clone();
            cv::Mat_<uchar> mask = mask_base.clone();

            transformer(image, mask);
            const cv::Rect roi = cv::Rect(0,0, image.cols, image.rows);

            // lpb is not a bow, its just a histogram, thus we need to treat it
            // differently
            if (feature_type != vf::feature_type::Lbp)
            {
                cv::Mat descriptors = extractor->denseExtract(image, locations);
                if (!descriptors.data || descriptors.rows == 0)
                {
                    continue;
                }
                bow_extractor->match(descriptors, words);
                bow_extractor->sumPool(
                        words,
                        locations,
                        roi,
                        mask,
                        pyramid_levels,
                        histogram);
            }
            else
            {
                vf::Lbp* lbp = dynamic_cast<vf::Lbp*>(extractor.get());
                lbp->sumPool(image,
                        roi,
                        mask,
                        lbp->wordCount(),
                        pyramid_levels,
                        histogram);
            }
            std::string extension =  transformer.name() + feature_extension;
            std::string feature_path = (output_dir / segmented_obj_path.filename()).replace_extension("").string() + extension;
            // serialize
            vf::BowHistogram h(histogram, feature_type, bow_count);
            utils::serialization::write_binary_archive(feature_path, h);
        }

        ++show_progress;
    }
}
