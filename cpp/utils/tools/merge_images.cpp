#include <istream>

#include <opencv2/opencv.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "cpp/utils/image_file_utils.hpp"

DEFINE_string(index_file, "", "Index of test images");
DEFINE_string(out_dir, "", "Output folder");

int main(int argc, char** argv) {
	google::ParseCommandLineFlags(&argc, &argv, true);
	CHECK(!FLAGS_index_file.empty()) << "No index file provided.";
	CHECK(!FLAGS_out_dir.empty()) << "No out dir provided.";

	utils::image_file::create_big_images(FLAGS_index_file, FLAGS_out_dir,
																			 40, 500);

	return 0;
}
