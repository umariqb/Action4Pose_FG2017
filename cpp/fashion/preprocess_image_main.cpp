/*
 * process_fashion_image.cpp
 *
 *  Created on: Jan 17, 2012
 *      Author: lbossard
 */
#define BOOST_FILESYSTEM_VERSION 3

#include <vector>

#include <boost/progress.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>
namespace fs = boost::filesystem;

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "cpp/utils/file_utils.hpp"
#include "cpp/utils/serialization/serialization.hpp"
#include "cpp/utils/serialization/opencv_serialization.hpp"
#include "cpp/vision/unstitch.hpp"
#include "cpp/vision/image_utils.hpp"

#include "body/body_detection.hpp"
#include "skin/SkinSegmentation.h"
#include "segmentation/Segmentation.h"

////////////////////////////////////////////////////////////////////////////////
// some options
namespace options
{
enum Options
{
    Unstitch 		 	= 0x1 << 0,
    NormalizeColor 		= 0x1 << 1,
    RemoveSkinParts		= 0x1 << 2,
    SaveOnlyWithFace 	= 0x1 << 3,
    ShowParts           = 0x1 << 4,
    SaveParts           = 0x1 << 5,
    NoSegmentation      = 0x1 << 6,
    CenteredBody        = 0x1 << 7,

    Default = Unstitch | NormalizeColor | SaveParts | CenteredBody,
};

typedef int T;

}

void processImagePart(
        cv::Mat image_part_scaled,
        const float pixel_fraction_to_discard,
        const int grab_cut_scale_factor,
        const bool do_remove_skin_parts,
        const bool centered_body_assumtion,
        std::vector<fashion::SegmentedObject>* clothes);

DEFINE_string(input_path,
        "",
        //"/home/lbossard/scratch/datasets/google-fashion/attributes/fabric/pattern/checkered",
//        "/home/lbossard/scratch/datasets/google-fashion/attributes/queries/general/colored",
        "folder containing images");
DEFINE_string(output_dir, "", "direcotry to save the parts. if nothing specified, nothing will be saved");
DEFINE_bool(unstitch, true, "should the images be unstitched");
DEFINE_bool(normalize_colors, true, "should the colors be normalized");
DEFINE_bool(remove_skin_parts, false, "should the skin be removed from the segmented image");
DEFINE_bool(show_parts, false, "should the segmentation be shown");
DEFINE_int32(max_side_length, 320, "maximum side length of a part");
DEFINE_bool(centered_body_assumption, true, "if no body could be detected, just take the center of the image");

DEFINE_bool(convert_only, false, "converts the whole image to a segmented object, without segmenting it. unstitch and normalize_colors are considere. the other settings ignored");
int main(int argc, char **argv)
{
    // init logging
    google::InitGoogleLogging(argv[0]);

    // get command line args
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_input_path.empty())
    {
         google::ShowUsageWithFlags(argv[0]);
         return 1;
    }
    // parse flags
    std::string input_path = FLAGS_input_path;
    options::T processing_options = 0;
    if (FLAGS_unstitch)
        processing_options |= options::Unstitch;
    if (FLAGS_normalize_colors)
        processing_options |= options::NormalizeColor;
    if(FLAGS_remove_skin_parts)
        processing_options |= options::RemoveSkinParts;
    if (FLAGS_show_parts)
        processing_options |= options::ShowParts;
    if (FLAGS_convert_only)
        processing_options |= options::NoSegmentation;
    if (FLAGS_centered_body_assumption)
            processing_options |= options::CenteredBody;

    std::string output_dir;
    if (FLAGS_output_dir.empty())
    {
        processing_options |= options::ShowParts;
    }
    else
    {
        output_dir = FLAGS_output_dir;
        processing_options |= options::SaveParts;
    }


    const float pixel_fraction_to_discard = 0.0005;
    const int max_image_side_length = FLAGS_max_side_length;
    const int grab_cut_scale_factor = 2;

    // read image into vector
    std::vector<fs::path> image_paths;
    utils::fs::collect_images(input_path, &image_paths);

    // process each image
    boost::progress_display progress(image_paths.size());
    BOOST_FOREACH(const fs::path& img_path, image_paths)
    {
        ++progress;
        std::string file_name = (output_dir / img_path.filename().stem()).string()
                                                           + "_0"+
                                                           + ".segobj";
        if (fs::exists(file_name))
        {
            std::cout << "skipping " << img_path << std::endl;
            continue;
        }

        // read image
        const cv::Mat image = cv::imread(img_path.string());
        if (image.data == NULL)
        {
            std::cerr << "could not load " << img_path.string() << std::endl;
            continue;
        }

        // split image
        std::vector<cv::Rect> parts;
        if (processing_options & options::Unstitch)
        {
            vision::Unstitch unstitch;
            unstitch.unstitchImage(image, parts);
        }
        else
        {
            parts.push_back(cv::Rect(0,0, image.cols, image.rows));
        }

        for (std::size_t i = 0; i < parts.size(); ++i) {
          const cv::Rect& part = parts[i];

          int id = atoi( img_path.filename().stem().c_str() );
          int id_0 = id/1000000;
          int id_1 = ((id%1000000)/1000);
          std::string dir_name = str( boost::format("%s%03d/%03d/%d/") % output_dir % id_0 % id_1 % id );
          std::string f_name = img_path.filename().stem().string() + "_"+ boost::lexical_cast<std::string>(i) + ".jpg";

          std::string file_name = dir_name + f_name;
//          std::cout << file_name << std::endl;
          cv::Mat image_part =  image(part);
          cv::imwrite(file_name, image_part);
//          cv::imshow(boost::lexical_cast<std::string>(i), image_part);
//          cv::waitKey(0);

        }


        continue;
        // process each image part
        std::vector<fashion::SegmentedObject> clothes;
        BOOST_FOREACH(const cv::Rect& part, parts)
        {
            // scale down if necessary
            cv::Mat image_part_scaled;   // scaled part
            double scale_factor = 1;
            {
                cv::Mat image_part =  image(part);
                scale_factor = vision::image_utils::scale_down_if_bigger(
                        image_part,
                        max_image_side_length,
                        image_part_scaled);
            }

            // we normalize each part independently
            if (options::NormalizeColor & processing_options)
            {
                vision::image_utils::normalizeColors(
                        image_part_scaled,
                        pixel_fraction_to_discard);
            }

            // segment or just take the whole image
            if (options::NoSegmentation & processing_options)
            {
                clothes.push_back(
                        fashion::SegmentedObject(
                                image_part_scaled,
                                cv::Mat_<uchar>(),
                                cv::Rect(
                                        part.x * scale_factor, // offset
                                        part.y * scale_factor,
                                        image_part_scaled.cols,
                                        image_part_scaled.rows)));
            }
            else
            {
                std::vector<fashion::SegmentedObject> bodies;
                processImagePart(
                        image_part_scaled,
                        pixel_fraction_to_discard,
                        grab_cut_scale_factor,
                        processing_options & options::RemoveSkinParts,
                        processing_options & options::CenteredBody,
                        &bodies);
                // add offset of part
                BOOST_FOREACH(fashion::SegmentedObject& body, bodies)
                {
                    body.location().x *= scale_factor;
                    body.location().y *= scale_factor;
                    clothes.push_back(body);
                }
            }
        }

        // save all parts
        for (std::size_t i = 0; i < clothes.size(); ++i)
        {
            std::string file_name = (output_dir / img_path.filename().stem()).string()
                                                   + "_"+ boost::lexical_cast<std::string>(i)
                                                   + ".segobj";

            const fashion::SegmentedObject& segmented_object = clothes[i];

            // save if necessary
            if (options::SaveParts & processing_options )
            {
                utils::serialization::write_binary_archive(file_name, segmented_object);
            }
            // show if necessary
            if (options::ShowParts & processing_options)
            {
                cv::imshow(
                        file_name,
                        segmented_object.getMaskedImageForDisplay());
            }
        }
        if (clothes.size() && (options::ShowParts & processing_options))
        {
            cv::waitKey();
            cv::destroyAllWindows();
        }
    }
}


//------------------------------------------------------------------------------
void processImagePart(
        cv::Mat image_part_scaled,
        const float pixel_fraction_to_discard,
        const int grab_cut_scale_factor,
        const bool do_remove_skin_parts,
        const bool centered_body_assumtion,
        std::vector<fashion::SegmentedObject>* clothes)
{
    // get grey version
    cv::Mat image_part_scaled_gray;
    cv::cvtColor(image_part_scaled, image_part_scaled_gray, CV_BGR2GRAY);
    cv::equalizeHist(image_part_scaled_gray, image_part_scaled_gray);

    // detect skin
    fashion::SkinSegmentation skin_segmentation;
    cv::Mat skin_map_scaled;
    skin_segmentation.createSkinMap(image_part_scaled, &skin_map_scaled);

    // detect bodies
    fashion::BodyDetection body_detection;
    std::vector<fashion::Body> bodies;
    body_detection.detectAllBodies(
            image_part_scaled,
            image_part_scaled_gray,
            skin_map_scaled,
            &bodies);

    // use grabcut to segment the bodies from the background
    fashion::Segmentation segmentation;
    for (std::size_t i = 0; i < bodies.size(); ++i)
    {
        // skip centered bounding box
        if (!centered_body_assumtion && bodies[i].method == fashion::Body::CenterProbability)
        {
            continue;
        }

        segmentation.getAllObjects(
                image_part_scaled,
                image_part_scaled_gray,
                cv::Mat(),
                grab_cut_scale_factor,
                skin_map_scaled,
                do_remove_skin_parts,
                bodies[i],
                clothes);
    }
}
