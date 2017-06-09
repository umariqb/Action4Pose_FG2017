/*
 * felz_pix_main.cpp
 *
 *  Created on: Sep 13, 2013
 *      Author: lbossard
 */

#include <boost/foreach.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
namespace fs = boost::filesystem;

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/highgui/highgui.hpp>


#include "cpp/utils/file_utils.hpp"
#include "cpp/utils/image_file_utils.hpp"
#include "cpp/vision/image_utils.hpp"
#include "cpp/third_party/felz_pix/felz_pix.hpp"



double next_halton(uint32_t index, uint32_t prime_base){
  double result = 0;
  double f = 1./prime_base;
  while (index > 0){
    result += f * (index % prime_base);
    index /= prime_base;
    f /= prime_base;
  }
  return result;
}


void superpixel_to_rgb(const cv::Mat_<int32_t>& labels, cv::Mat* img_){

  img_->create(labels.rows, labels.cols, cv::DataType<cv::Vec3b>::type);
  cv::Mat_<cv::Vec3b> img = *img_;

  for (int r = 0; r < labels.rows; ++r){
    for (int c = 0; c < labels.cols; ++c){
      cv::Vec3b& pixel = img(r,c);
      pixel[0] =  127 * next_halton(labels(r,c), 2);
      pixel[1] = 255 * (.5 + next_halton(labels(r,c), 3) * .5);
      pixel[2] = 255 * (.7 + next_halton(labels(r,c), 5) * .3);
    }
  }
  cv::cvtColor(img, img, cv::COLOR_HSV2BGR);
}



DEFINE_string(images, "", "image directory, single image file or or file with image paths");
DEFINE_double(sigma, .1, "sigma");
DEFINE_double(k, 20, "k");
DEFINE_int32(min_size, 20, "superpixel min size");
DEFINE_int32(max_side_length, 0, "");
DEFINE_double(alpha, .6, "alpha blending");
DEFINE_bool(visualize, true, "if should visualize");
DEFINE_string(output_dir, "","dir for outptup");
int main(int argc, char** argv){

  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  // get command line args
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_images.empty()) {
    google::ShowUsageWithFlags(argv[0]);
    return 1;
  }

  const float sigma = FLAGS_sigma;
  const int k = FLAGS_k;
  const int min_size = FLAGS_min_size;
  const fs::path images = FLAGS_images;
  const int max_side_length = FLAGS_max_side_length;
  const float alpha = FLAGS_alpha;
  const bool do_visualize = FLAGS_visualize;
  const bool do_save = !FLAGS_output_dir.empty();
  const fs::path output_base = FLAGS_output_dir;

  std::vector<fs::path> image_paths;
  if (fs::is_directory(images)){
    utils::fs::collect_images(images.string(), &image_paths);
  }
  else {
    const std::string suffix = images.extension().string();
    if (suffix == ".jpg"
        || suffix == ".jpeg"
        || suffix == ".gif"
        || suffix == ".png"
        || suffix == ".bmp"
        || suffix == ".tiff"){
      image_paths.push_back(images);
    }
    else {
      CHECK(utils::image_file::load_paths(images.string(), image_paths));
    }
  }
  std::sort(image_paths.begin(), image_paths.end());

  cv::Mat_<int32_t> labels;
  BOOST_FOREACH(const fs::path& image_path, image_paths){
     // load image and scale it to max side legnth
     cv::Mat image_orig = cv::imread(image_path.string());
     if (image_orig.data == NULL) {
       LOG(ERROR) << "could not load " << image_path;
       continue;
     }
     cv::Mat image = image_orig;
     double scale_factor = 1;
     if (max_side_length > 0){
       scale_factor = vision::image_utils::scale_down_if_bigger(
           image_orig,
           max_side_length,
           image);
     }

     const int num_components = awesomeness::third_party::felz_pix::segment_image(image, sigma, k, min_size, &labels);
     LOG(INFO) << "num_pixels= " << num_components << ": " << image_path;
     if (scale_factor < 1){
       cv::resize(labels, labels, image_orig.size(), /*fx*/ 0, /*fy*/ 0, /*interpolation*/ cv::INTER_NEAREST);
     }

     cv::Mat colored_labels;
     superpixel_to_rgb(labels, &colored_labels);
     cv::Mat blended;
     cv::addWeighted(image_orig, alpha, colored_labels, (1-alpha), 0.0, blended);

     if (do_save){
       std::string file_name = image_path.filename().replace_extension("").string() + "_seg.jpg";
       fs::path output_path = output_base / file_name;
       cv::imwrite(output_path.string(), blended);
     }
     if (do_visualize){
       cv::imshow("original", image_orig);
       cv::imshow("supperpixls", colored_labels);
       cv::imshow("blended", blended);
       cv::waitKey();
       cv::destroyAllWindows();
     }
  }
}

